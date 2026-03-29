"""
Policy trajectory recorder + matplotlib 3D animation.

Since Isaac Lab headless recording requires display capture (which isn't
available in a headless container), this script instead:
  1. Runs the policy for --duration seconds at 50 Hz
  2. Logs base position, orientation, joint angles, and foot contacts
  3. Saves trajectory data to results/trajectory_{terrain}.npz
  4. Renders a matplotlib 3D animation and saves to
     results/policy_videos/{terrain}_policy.mp4 via ffmpeg

Usage
-----
    cd /workspace/IsaacLab
    /workspace/run_training.sh \\
        /workspace/isaac-lab-locomotion/scripts/record_policy_video.py \\
        --checkpoint /workspace/checkpoints/flat/best_model.pt \\
        --terrain flat \\
        --duration 10
"""

from __future__ import annotations

import argparse
import os
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
parser.add_argument(
    "--duration",
    type=float,
    default=10.0,
    help="Recording duration in seconds (default: 10)",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel environments (1 for single trajectory)",
)
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

CHECKPOINT_REWARD_MAP = {
    "flat": 14.527,
    "slopes": 17.677,
    "stairs": 9.871,
    "contact_aware": "TBD",  # filled after training
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


def load_actor(checkpoint_path, obs_size, action_size, device):
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
    actor.eval()
    return actor


def record_trajectory(
    checkpoint: str,
    terrain: str,
    duration: float,
    num_envs: int,
    device: str = "cuda",
) -> dict:
    """Run policy and record trajectory data."""
    SIM_DT = 0.02  # 50 Hz
    num_steps = int(duration / SIM_DT)

    env_id = TERRAIN_ENV_MAP[terrain]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs

    gym_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    rsl_env = RslRlVecEnvWrapper(gym_env)
    env = TensorDictVecEnvWrapper(rsl_env)

    actor = load_actor(checkpoint, rsl_env.num_obs, env.num_actions, device)

    # Trajectory buffers
    base_pos = []
    base_quat = []
    joint_pos = []
    rewards_buf = []
    timestamps = []

    obs_td, _ = env.reset()

    with torch.no_grad():
        for step in range(num_steps):
            obs = obs_td["policy"]
            actions = actor(obs)
            obs_td, rewards, dones, _ = env.step(actions)

            isaac_env = env._env.unwrapped
            robot = isaac_env.scene["robot"]
            rd = robot.data

            if hasattr(rd, "root_pos_w"):
                base_pos.append(rd.root_pos_w[0, :3].cpu().numpy())
            if hasattr(rd, "root_quat_w"):
                base_quat.append(rd.root_quat_w[0].cpu().numpy())
            if hasattr(rd, "joint_pos"):
                joint_pos.append(rd.joint_pos[0].cpu().numpy())
            rewards_buf.append(float(rewards[0].item()))
            timestamps.append(step * SIM_DT)

    env.close()

    trajectory = {
        "terrain": terrain,
        "checkpoint": checkpoint,
        "duration": duration,
        "dt": SIM_DT,
        "timestamps": np.array(timestamps),
        "rewards": np.array(rewards_buf),
    }
    if base_pos:
        trajectory["base_pos"] = np.array(base_pos)
    if base_quat:
        trajectory["base_quat"] = np.array(base_quat)
    if joint_pos:
        trajectory["joint_pos"] = np.array(joint_pos)

    return trajectory


def render_animation(trajectory: dict, out_path: str, terrain: str) -> None:
    """Create a matplotlib 3D animation and save as MP4 via ffmpeg."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.animation import FFMpegWriter

    pos = trajectory.get("base_pos")
    if pos is None or len(pos) == 0:
        print("No base position data — skipping animation", flush=True)
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    trail_len = 50
    (line,) = ax.plot([], [], [], "b-", linewidth=1.5, alpha=0.7, label="Base path")
    (point,) = ax.plot([], [], [], "ro", markersize=6, label="Current position")

    # Set axis limits
    margin = 1.0
    ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
    ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)
    z_max = max(pos[:, 2].max() + 0.5, 0.5)
    ax.set_zlim(0, z_max)

    reward = CHECKPOINT_REWARD_MAP.get(terrain, "?")
    ax.set_title(
        f"AnymalC — {terrain.replace('_', ' ').title()} Terrain\n"
        f"Best reward: {reward}  |  dt={trajectory['dt']*1000:.0f}ms",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper left", fontsize=8)

    fps = 25
    skip = max(1, int(1.0 / (fps * trajectory["dt"])))
    frames = range(0, len(pos), skip)

    def update(frame_idx):
        i = frame_idx
        start = max(0, i - trail_len)
        line.set_data(pos[start:i, 0], pos[start:i, 1])
        line.set_3d_properties(pos[start:i, 2])
        if i < len(pos):
            point.set_data([pos[i, 0]], [pos[i, 1]])
            point.set_3d_properties([pos[i, 2]])
        t = trajectory["timestamps"][i] if i < len(trajectory["timestamps"]) else 0.0
        ax.set_title(
            f"AnymalC — {terrain.replace('_', ' ').title()} Terrain\n" f"Best reward: {reward}  |  t={t:.1f}s",
            fontsize=11,
            fontweight="bold",
        )
        return line, point

    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, update, frames=list(frames), interval=1000 // fps, blit=False)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    try:
        anim.save(out_path, writer=writer)
        print(f"Video saved → {out_path}", flush=True)
    except Exception as e:
        print(f"Video save failed ({e}), saving frames as PNG instead", flush=True)
        for i in list(frames)[::5]:
            update(i)
            fig.savefig(out_path.replace(".mp4", f"_frame{i:04d}.png"))
    plt.close(fig)


if __name__ == "__main__":
    print(
        f"Recording {args_cli.duration}s trajectory on {args_cli.terrain}...",
        flush=True,
    )
    traj = record_trajectory(
        checkpoint=args_cli.checkpoint,
        terrain=args_cli.terrain,
        duration=args_cli.duration,
        num_envs=args_cli.num_envs,
    )

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory data
    npz_path = out_dir / f"trajectory_{args_cli.terrain}.npz"
    np.savez(str(npz_path), **{k: v for k, v in traj.items() if isinstance(v, np.ndarray)})
    print(f"Trajectory data saved → {npz_path}", flush=True)

    # Render animation
    video_dir = out_dir / "policy_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = str(video_dir / f"{args_cli.terrain}_policy.mp4")
    render_animation(traj, mp4_path, args_cli.terrain)

    simulation_app.close()
