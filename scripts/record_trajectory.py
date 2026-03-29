"""
Trajectory recording script — runs policy for N steps and saves state data.

Usage
-----
    cd /workspace/IsaacLab
    /workspace/run_training.sh \\
        /workspace/isaac-lab-locomotion/scripts/record_trajectory.py \\
        --checkpoint /workspace/checkpoints/flat/best_model.pt \\
        --terrain flat \\
        --steps 500

Outputs
-------
    /workspace/isaac-lab-locomotion/results/trajectories/trajectory_{terrain}.npz

Saved arrays (all indexed by step):
    base_position       [steps, 3]   - world-frame XYZ
    base_orientation    [steps, 4]   - quaternion (w, x, y, z)
    joint_positions     [steps, N_dof]
    foot_contact_forces [steps, 4, 3] - zeros if sensor unavailable
    base_linear_velocity [steps, 3]
    base_angular_velocity [steps, 3]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record AnymalC policy trajectory.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--terrain",
    type=str,
    choices=["flat", "slopes", "stairs", "contact_aware"],
    default="flat",
)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.env import VecEnv
from tensordict import TensorDict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TERRAIN_ENV_MAP = {
    "flat": "Isaac-Velocity-Flat-Anymal-C-v0",
    "slopes": "Isaac-Velocity-Rough-Anymal-C-v0",
    "stairs": "Isaac-Velocity-Rough-Anymal-C-v0",
    "contact_aware": "Isaac-Velocity-Rough-Anymal-C-v0",
}


class TensorDictVecEnvWrapper(VecEnv):
    def __init__(self, env):
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = getattr(env, "cfg", {})
        self.extras = {}

    def _to_td(self, obs):
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self):
        obs, _ = self._env.get_observations()
        self.episode_length_buf = self._env.episode_length_buf
        return self._to_td(obs)

    def step(self, actions):
        obs, rewards, dones, extras = self._env.step(actions)
        self.episode_length_buf = self._env.episode_length_buf
        self.extras = extras
        return self._to_td(obs), rewards, dones, extras

    def reset(self):
        obs, extras = self._env.reset()
        return self._to_td(obs), extras

    def close(self):
        self._env.close()


def load_actor(checkpoint_path: str, obs_size: int, action_size: int, device: str):
    import os
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden=(512, 256, 128)):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ELU()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    actor = MLP(obs_size, action_size).to(device)
    if os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, weights_only=False, map_location=device)
        if "actor_state_dict" in saved:
            actor.load_state_dict(saved["actor_state_dict"], strict=False)
            print(f"Loaded actor from {checkpoint_path}", flush=True)
    actor.eval()
    return actor


def record(checkpoint: str, terrain: str, num_steps: int, num_envs: int = 1, device: str = "cuda"):
    env_id = TERRAIN_ENV_MAP[terrain]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None

    gym_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    rsl_env = RslRlVecEnvWrapper(gym_env)
    env = TensorDictVecEnvWrapper(rsl_env)

    actor = load_actor(checkpoint, rsl_env.num_obs, env.num_actions, device)

    base_positions = []
    base_orientations = []
    joint_positions = []
    foot_contact_forces = []
    base_linear_velocities = []
    base_angular_velocities = []

    obs_td, _ = env.reset()

    with torch.no_grad():
        for _ in range(num_steps):
            obs = obs_td["policy"]
            actions = actor(obs)
            obs_td, _, _, _ = env.step(actions)

            try:
                isaac_env = env._env.unwrapped
                rd = isaac_env.scene["robot"].data

                base_positions.append(rd.root_pos_w[0].cpu().numpy())
                base_orientations.append(rd.root_quat_w[0].cpu().numpy())
                joint_positions.append(rd.joint_pos[0].cpu().numpy())
                base_linear_velocities.append(rd.root_lin_vel_w[0].cpu().numpy())
                base_angular_velocities.append(rd.root_ang_vel_w[0].cpu().numpy())

                # Contact forces — try multiple sensor keys
                cf_found = False
                for sensor_key in ["contact_forces", "feet_contact", "foot_contact"]:
                    try:
                        sensor = isaac_env.scene[sensor_key]
                        cf = sensor.data.net_forces_w[0].cpu().numpy()  # [4, 3]
                        foot_contact_forces.append(cf)
                        cf_found = True
                        break
                    except (KeyError, AttributeError):
                        continue
                if not cf_found:
                    foot_contact_forces.append(np.zeros((4, 3), dtype=np.float32))

            except Exception:
                # If state extraction fails, fill with zeros
                base_positions.append(np.zeros(3, dtype=np.float32))
                base_orientations.append(np.array([1, 0, 0, 0], dtype=np.float32))
                joint_positions.append(np.zeros(12, dtype=np.float32))
                foot_contact_forces.append(np.zeros((4, 3), dtype=np.float32))
                base_linear_velocities.append(np.zeros(3, dtype=np.float32))
                base_angular_velocities.append(np.zeros(3, dtype=np.float32))

    env.close()

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trajectory_{terrain}.npz"

    np.savez(
        out_path,
        base_position=np.array(base_positions),
        base_orientation=np.array(base_orientations),
        joint_positions=np.array(joint_positions),
        foot_contact_forces=np.array(foot_contact_forces),
        base_linear_velocity=np.array(base_linear_velocities),
        base_angular_velocity=np.array(base_angular_velocities),
    )
    print(f"Saved trajectory → {out_path}", flush=True)
    return str(out_path)


if __name__ == "__main__":
    record(
        checkpoint=args_cli.checkpoint,
        terrain=args_cli.terrain,
        num_steps=args_cli.steps,
        num_envs=args_cli.num_envs,
    )
    simulation_app.close()
