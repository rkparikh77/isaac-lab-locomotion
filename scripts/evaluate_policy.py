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
parser.add_argument(
    "--eval_terrain",
    type=str,
    choices=["flat", "slopes", "stairs", "contact_aware"],
    default=None,
    help="Override terrain for cross-terrain evaluation (default: same as --terrain)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.runner_utils import create_env, load_policy  # noqa: E402


def evaluate(
    checkpoint: str,
    terrain: str,
    num_episodes: int,
    num_envs: int,
    eval_terrain: str | None = None,
    device: str = "cuda",
) -> dict:
    # eval_terrain allows cross-terrain evaluation: use a different env than training terrain
    env_terrain = eval_terrain if eval_terrain is not None else terrain

    env = create_env(env_terrain, num_envs)
    policy = load_policy(checkpoint, env, device=device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    obs_td, _ = env.reset()
    episode_reward = torch.zeros(num_envs, device=device)
    episode_length = torch.zeros(num_envs, dtype=torch.int, device=device)

    episodes_done = 0
    max_steps = num_episodes * env.max_episode_length * 2  # safety cap

    with torch.no_grad():
        for step in range(max_steps):
            actions = policy(obs_td)
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
        "eval_terrain": env_terrain,
        "checkpoint": checkpoint,
        "num_episodes": len(tail),
        "mean_reward": float(sum(tail) / len(tail)),
        "std_reward": float((sum((r - sum(tail) / len(tail)) ** 2 for r in tail) / len(tail)) ** 0.5),
        "max_reward": float(max(tail)),
        "min_reward": float(min(tail)),
        "episode_length_mean": float(sum(episode_lengths[:num_episodes]) / len(episode_lengths[:num_episodes])),
    }


if __name__ == "__main__":
    eval_terrain = args_cli.eval_terrain  # None means same as --terrain
    results = evaluate(
        checkpoint=args_cli.checkpoint,
        terrain=args_cli.terrain,
        num_episodes=args_cli.episodes,
        num_envs=args_cli.num_envs,
        eval_terrain=eval_terrain,
    )

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use eval_terrain for filename when doing cross-terrain evaluation
    out_terrain = eval_terrain if eval_terrain is not None else args_cli.terrain
    out_path = out_dir / f"evaluation_{out_terrain}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  EVALUATION: {args_cli.terrain.upper()} policy on {out_terrain.upper()} terrain")
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
