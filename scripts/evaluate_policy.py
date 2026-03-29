"""
Policy evaluation script — runs N episodes and reports statistics.

Usage
-----
    cd /workspace/IsaacLab
    /workspace/run_training.sh \\
        /workspace/isaac-lab-locomotion/scripts/evaluate_policy.py \\
        --checkpoint /workspace/checkpoints/flat/best_model.pt \\
        --terrain flat \\
        --episodes 100

Outputs
-------
    /workspace/isaac-lab-locomotion/results/evaluation_{terrain}.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# ── AppLauncher must come before any omni imports ─────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained AnymalC locomotion policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the best_model.pt checkpoint")
parser.add_argument(
    "--terrain",
    type=str,
    choices=["flat", "slopes", "stairs", "contact_aware"],
    default="flat",
    help="Terrain type to evaluate on",
)
parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import gymnasium as gym
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
    """Load just the actor network from a checkpoint for greedy evaluation."""
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

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: checkpoint not found at {checkpoint_path}", flush=True)
        return actor

    saved = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_key = "actor_state_dict"
    if state_key in saved:
        actor.load_state_dict(saved[state_key], strict=False)
        print(f"Loaded actor from {checkpoint_path}", flush=True)
    else:
        print(
            f"WARNING: key '{state_key}' not found in checkpoint. Keys: {list(saved.keys())}",
            flush=True,
        )
    actor.eval()
    return actor


def evaluate(
    checkpoint: str,
    terrain: str,
    num_episodes: int,
    num_envs: int,
    device: str = "cuda",
) -> dict:
    env_id = TERRAIN_ENV_MAP[terrain]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    # Disable curriculum during evaluation
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None

    gym_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    rsl_env = RslRlVecEnvWrapper(gym_env)
    env = TensorDictVecEnvWrapper(rsl_env)

    obs_size = rsl_env.num_obs
    action_size = env.num_actions

    actor = load_actor(checkpoint, obs_size, action_size, device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    obs_td, _ = env.reset()
    episode_reward = torch.zeros(num_envs, device=device)
    episode_length = torch.zeros(num_envs, dtype=torch.int, device=device)

    episodes_done = 0
    max_steps = num_episodes * env.max_episode_length * 2  # safety cap

    with torch.no_grad():
        for step in range(max_steps):
            obs = obs_td["policy"]
            actions = actor(obs)
            obs_td, rewards, dones, _ = env.step(actions)

            episode_reward += rewards
            episode_length += 1

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                for idx in done_indices:
                    episode_rewards.append(float(episode_reward[idx].item()))
                    episode_lengths.append(int(episode_length[idx].item()))
                    episode_reward[idx] = 0.0
                    episode_length[idx] = 0
                    episodes_done += 1

            if episodes_done >= num_episodes:
                break

    env.close()

    if not episode_rewards:
        episode_rewards = [0.0]
        episode_lengths = [0]

    tail = episode_rewards[:num_episodes]
    return {
        "terrain": terrain,
        "checkpoint": checkpoint,
        "num_episodes": len(tail),
        "mean_reward": float(sum(tail) / len(tail)),
        "std_reward": float((sum((r - sum(tail) / len(tail)) ** 2 for r in tail) / len(tail)) ** 0.5),
        "max_reward": float(max(tail)),
        "min_reward": float(min(tail)),
        "episode_length_mean": float(sum(episode_lengths[:num_episodes]) / len(episode_lengths[:num_episodes])),
    }


if __name__ == "__main__":
    results = evaluate(
        checkpoint=args_cli.checkpoint,
        terrain=args_cli.terrain,
        num_episodes=args_cli.episodes,
        num_envs=args_cli.num_envs,
    )

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"evaluation_{args_cli.terrain}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  EVALUATION: {args_cli.terrain.upper()}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("=" * 55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:.3f}")
        else:
            print(f"  {k:<28}: {v}")
    print(f"\n  Saved → {out_path}")
    print("=" * 55)

    simulation_app.close()
