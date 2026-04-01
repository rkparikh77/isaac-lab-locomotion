"""
Flat-terrain AnymalC velocity-tracking training script.

Usage:
    python experiments/01_flat_terrain/train.py
    python experiments/01_flat_terrain/train.py --num_envs 1024 --max_iterations 500
    python experiments/01_flat_terrain/train.py --headless  # no GUI

All tunable numbers are in config.py — do not add numeric literals here.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

# ── Isaac Lab app launcher must come before any omni imports ──────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train AnymalC on flat terrain with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=None, help="Override TrainingConfig.num_envs")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Override TrainingConfig.max_iterations",
)
parser.add_argument("--run_name", type=str, default=None, help="Override wandb run name")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Remaining imports (after app launch) ─────────────────────────────────────
import gymnasium as gym
import torch
import wandb

# Isaac Lab task registry — registers all bundled envs including AnymalC variants
import isaaclab_tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import (
    NetworkConfig,
    PPOConfig,
    RewardWeights,
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# rsl_rl v5 bridge: wrap the old-API RslRlVecEnvWrapper to return TensorDicts
# ─────────────────────────────────────────────────────────────────────────────


class ExtrasTrackingWrapper(VecEnv):
    """Thin wrapper around RslRlVecEnvWrapper that stores the latest extras
    dict for metric extraction between training iterations."""

    def __init__(self, env: RslRlVecEnvWrapper) -> None:
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = getattr(env, "cfg", {})
        self.extras = {}

    def get_observations(self) -> TensorDict:
        obs_td = self._env.get_observations()
        self.episode_length_buf = self._env.episode_length_buf
        return obs_td

    def step(self, actions: torch.Tensor):
        obs_td, rewards, dones, extras = self._env.step(actions)
        self.episode_length_buf = self._env.episode_length_buf
        self.extras = extras
        return obs_td, rewards, dones, extras

    def reset(self):
        obs_td = self._env.get_observations()
        return obs_td, {}

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def build_runner_cfg(cfg: TrainingConfig) -> dict:
    """Build the rsl_rl v2 training config dict."""
    p: PPOConfig = cfg.ppo
    n: NetworkConfig = cfg.network
    return {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": p.value_loss_coef,
            "use_clipped_value_loss": True,
            "clip_param": p.clip_param,
            "entropy_coef": p.entropy_coef,
            "num_learning_epochs": p.num_learning_epochs,
            "num_mini_batches": p.num_mini_batches,
            "learning_rate": p.learning_rate,
            "schedule": "adaptive",
            "gamma": p.gamma,
            "lam": p.gae_lambda,
            "desired_kl": 0.01,
            "max_grad_norm": p.max_grad_norm,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": n.actor_hidden_dims,
            "activation": n.activation,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": n.init_noise_std,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": n.critic_hidden_dims,
            "activation": n.activation,
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": p.num_steps_per_env,
        "max_iterations": cfg.max_iterations,
        "save_interval": cfg.checkpoint_interval,
        "experiment_name": cfg.wandb_run_name or "flat_terrain",
        "run_name": cfg.wandb_run_name or "",
        "logger": "tensorboard",  # built-in logger disabled (log_dir=None)
        "wandb_project": cfg.wandb_project,
    }


def make_env(cfg: TrainingConfig) -> ExtrasTrackingWrapper:
    """Create and wrap the Isaac Lab vectorised environment."""
    env_cfg = load_cfg_from_registry(cfg.env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs
    env = gym.make(cfg.env_id, cfg=env_cfg, render_mode=None)
    return ExtrasTrackingWrapper(RslRlVecEnvWrapper(env))


def apply_reward_weights(env: ExtrasTrackingWrapper, rw: RewardWeights) -> None:
    """Patch the environment's reward manager with our config weights."""
    reward_manager = env._env.unwrapped.reward_manager
    weight_map = {
        "track_lin_vel_xy_exp": rw.lin_vel_tracking,
        "track_ang_vel_z_exp": rw.ang_vel_tracking,
        "lin_vel_z_l2": rw.lin_vel_z_penalty,
        "ang_vel_xy_l2": rw.ang_vel_xy_penalty,
        "dof_torques_l2": rw.joint_torque_penalty,
        "dof_acc_l2": rw.joint_accel_penalty,
        "action_rate_l2": rw.action_rate_penalty,
        "feet_air_time": rw.feet_air_time_bonus,
        "undesired_contacts": rw.undesired_contact_penalty,
        "flat_orientation_l2": rw.flat_orientation_penalty,
    }
    for term_name, weight in weight_map.items():
        if hasattr(reward_manager, term_name):
            getattr(reward_manager, term_name).weight = weight


# ─────────────────────────────────────────────────────────────────────────────
# Early-stopping state
# ─────────────────────────────────────────────────────────────────────────────


class EarlyStopper:
    """Fires when mean episode reward exceeds *threshold* for *patience* consecutive iters."""

    def __init__(self, threshold: float, patience: int) -> None:
        self.threshold = threshold
        self.patience = patience
        self._consecutive: int = 0

    def update(self, mean_reward: float) -> bool:
        """Return True if training should stop."""
        if mean_reward > self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0
        return self._consecutive >= self.patience


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(cfg: TrainingConfig) -> None:
    pathlib.Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── wandb init ────────────────────────────────────────────────────────────
    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        config={
            "num_envs": cfg.num_envs,
            "terrain_type": cfg.terrain_type,
            "max_iterations": cfg.max_iterations,
            **vars(cfg.ppo),
            **vars(cfg.network),
            **vars(cfg.reward_weights),
        },
    )

    # ── Environment ───────────────────────────────────────────────────────────
    env = make_env(cfg)
    apply_reward_weights(env, cfg.reward_weights)

    # ── RSL-RL runner ─────────────────────────────────────────────────────────
    runner_cfg = build_runner_cfg(cfg)
    rsl_log_dir = os.path.join(cfg.checkpoint_dir, "rsl_rl_logs")
    os.makedirs(rsl_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=rsl_log_dir, device=cfg.device)

    stopper = EarlyStopper(cfg.early_stop_reward_threshold, cfg.early_stop_patience)
    best_reward: float = float("-inf")
    wall_start: float = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(cfg.max_iterations):
        # One PPO update step
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

        # ── Metrics extraction ────────────────────────────────────────────────
        log_data: dict = {}

        # Episode rewards from the runner's logger buffer
        if hasattr(runner, "logger") and len(runner.logger.rewbuffer) > 0:
            mean_reward = sum(runner.logger.rewbuffer) / len(runner.logger.rewbuffer)
            log_data["episode_reward"] = mean_reward
        if hasattr(runner, "logger") and len(runner.logger.lenbuffer) > 0:
            log_data["episode_length"] = sum(runner.logger.lenbuffer) / len(runner.logger.lenbuffer)

        # Velocity tracking error from the env info dict
        if hasattr(env, "extras") and env.extras and "log" in env.extras:
            env_log = env.extras["log"]
            if "tracking_lin_vel" in env_log:
                log_data["velocity_tracking_error"] = 1.0 - env_log["tracking_lin_vel"].mean().item()

        log_data["iteration"] = iteration
        log_data["wall_time_hours"] = (time.time() - wall_start) / 3600.0

        # ── wandb logging (every log_interval iters) ──────────────────────────
        if iteration % cfg.log_interval == 0:
            wandb.log(log_data, step=iteration)

        # ── Periodic checkpointing ────────────────────────────────────────────
        if (iteration + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_{iteration + 1}.pt")
            runner.save(ckpt_path)
            print(f"[iter {iteration + 1:>5}] checkpoint saved → {ckpt_path}", flush=True)

        # ── Best-model tracking ───────────────────────────────────────────────
        mean_ep_reward = log_data.get("episode_reward", float("-inf"))
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            runner.save(cfg.best_model_path)

        # ── Early stopping ────────────────────────────────────────────────────
        if stopper.update(mean_ep_reward):
            print(
                f"[iter {iteration + 1}] Early stopping triggered — "
                f"mean reward {mean_ep_reward:.3f} > {cfg.early_stop_reward_threshold} "
                f"for {cfg.early_stop_patience} consecutive iters."
            )
            runner.save(cfg.best_model_path)
            wandb.summary["early_stop_iteration"] = iteration + 1
            wandb.summary["final_reward"] = mean_ep_reward
            break

        if iteration % cfg.log_interval == 0:
            print(
                f"[iter {iteration + 1:>5}/{cfg.max_iterations}] "
                f"reward={mean_ep_reward:.3f}  "
                f"best={best_reward:.3f}",
                flush=True,
            )

    wandb.summary["best_reward"] = best_reward
    wandb.summary["wall_time_hours"] = (time.time() - wall_start) / 3600.0
    run.finish()
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TrainingConfig()

    # Apply CLI overrides
    if args_cli.num_envs is not None:
        cfg.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        cfg.wandb_run_name = args_cli.run_name

    train(cfg)
    simulation_app.close()
