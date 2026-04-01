"""
Stair-terrain AnymalC velocity-tracking training script.

Warm-starts from the slope-terrain best checkpoint and fine-tunes for 1 000
iterations on pyramid stairs (step height 5–20 cm, step width 25–40 cm).

Usage:
    python /workspace/isaac-lab-locomotion/experiments/03_stairs/train.py --headless
    python /workspace/isaac-lab-locomotion/experiments/03_stairs/train.py --headless \\
        --load-checkpoint /workspace/checkpoints/slopes/best_model.pt

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

parser = argparse.ArgumentParser(description="Train AnymalC on stair terrain with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument(
    "--load-checkpoint",
    type=str,
    default=None,
    dest="load_checkpoint",
    help="Path to pretrained checkpoint (overrides config default)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Remaining imports (after app launch) ─────────────────────────────────────
import gymnasium as gym
import torch
import wandb

import isaaclab_tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import NetworkConfig, PPOConfig, RewardWeights, TrainingConfig

# ─────────────────────────────────────────────────────────────────────────────
# rsl_rl v5 TensorDict bridge (identical to 01_flat_terrain)
# ─────────────────────────────────────────────────────────────────────────────


class ExtrasTrackingWrapper(VecEnv):
    """Thin wrapper around RslRlVecEnvWrapper that stores the latest extras."""

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
    p, n = cfg.ppo, cfg.network
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
        "experiment_name": cfg.wandb_run_name or "stair_terrain",
        "run_name": cfg.wandb_run_name or "",
        "logger": "tensorboard",
        "wandb_project": cfg.wandb_project,
    }


def make_env(cfg: TrainingConfig) -> ExtrasTrackingWrapper:
    """Create the Rough env with terrain generator patched to stair sub-terrains only."""
    import isaaclab.terrains as terrain_gen

    env_cfg = load_cfg_from_registry(cfg.env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs

    # Replace mixed rough terrain with pyramid stairs only.
    # step_height_range in metres; step_width takes the midpoint of config range.
    t = cfg.terrain
    step_width_mid = (t.step_width_min_m + t.step_width_max_m) / 2.0
    env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(t.step_height_min_m, t.step_height_max_m),
            step_width=step_width_mid,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(t.step_height_min_m, t.step_height_max_m),
            step_width=step_width_mid,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    }

    env = gym.make(cfg.env_id, cfg=env_cfg, render_mode=None)
    return ExtrasTrackingWrapper(RslRlVecEnvWrapper(env))


def apply_reward_weights(env: ExtrasTrackingWrapper, rw: RewardWeights) -> None:
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


def extract_stair_metrics(env: ExtrasTrackingWrapper) -> dict:
    """
    Pull stair-specific diagnostics from env.extras["log"].

    Expected keys (populated by the terrain/event systems):
      "foot_clearance_above_step" — per-env foot clearance in metres
      "fall_events"               — per-env cumulative falls this episode
      "episode_count"             — per-env episodes completed
    """
    metrics: dict = {}
    if not (hasattr(env, "extras") and env.extras and "log" in env.extras):
        return metrics

    env_log = env.extras["log"]

    clearance = env_log.get("foot_clearance_above_step")
    if clearance is not None and clearance.numel() > 0:
        metrics["foot_clearance_above_step"] = clearance.float().mean().item()

    fall_events = env_log.get("fall_events")
    episode_count = env_log.get("episode_count")
    if fall_events is not None and episode_count is not None and fall_events.numel() > 0 and episode_count.numel() > 0:
        safe_ep = episode_count.float().clamp(min=1.0)
        metrics["fall_rate"] = ((fall_events.float() / safe_ep) * 100.0).mean().item()

    return metrics


def _partial_load_checkpoint(runner: OnPolicyRunner, path: str, device: str) -> None:
    """Load weights, skipping tensors whose shapes differ (safe cross-stage transfer)."""
    saved = torch.load(path, weights_only=False, map_location=device)
    transferred, skipped = [], []

    for model_name, state_key in [
        ("actor", "actor_state_dict"),
        ("critic", "critic_state_dict"),
    ]:
        model = getattr(runner.alg, model_name)
        current_sd = model.state_dict()
        saved_sd = saved.get(state_key, {})
        filtered = {}
        for k, v in saved_sd.items():
            if k in current_sd and current_sd[k].shape == v.shape:
                filtered[k] = v
                transferred.append(k)
            else:
                skipped.append(f"{model_name}.{k}")
        current_sd.update(filtered)
        model.load_state_dict(current_sd, strict=True)

    print(
        f"  Transferred {len(transferred)} tensors, skipped {len(skipped)} (shape mismatch).",
        flush=True,
    )


class EarlyStopper:
    def __init__(self, threshold: float, patience: int) -> None:
        self.threshold = threshold
        self.patience = patience
        self._consecutive = 0

    def update(self, mean_reward: float) -> bool:
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

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        config={
            "num_envs": cfg.num_envs,
            "terrain_type": cfg.terrain_type,
            "max_iterations": cfg.max_iterations,
            "load_checkpoint": cfg.load_checkpoint,
            **vars(cfg.ppo),
            **vars(cfg.network),
            **vars(cfg.reward_weights),
            **vars(cfg.terrain),
        },
    )

    env = make_env(cfg)
    apply_reward_weights(env, cfg.reward_weights)

    runner_cfg = build_runner_cfg(cfg)
    rsl_log_dir = os.path.join(cfg.checkpoint_dir, "rsl_rl_logs")
    os.makedirs(rsl_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=rsl_log_dir, device=cfg.device)

    # ── Warm-start ────────────────────────────────────────────────────────────
    if cfg.load_checkpoint and os.path.exists(cfg.load_checkpoint):
        print(f"Loading checkpoint: {cfg.load_checkpoint}", flush=True)
        _partial_load_checkpoint(runner, cfg.load_checkpoint, cfg.device)
    elif cfg.load_checkpoint:
        print(
            f"WARNING: checkpoint '{cfg.load_checkpoint}' not found — training from scratch.",
            flush=True,
        )

    stopper = EarlyStopper(cfg.early_stop_reward_threshold, cfg.early_stop_patience)
    best_reward: float = float("-inf")
    wall_start = time.time()

    for iteration in range(cfg.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

        log_data: dict = {}

        if hasattr(runner, "logger") and len(runner.logger.rewbuffer) > 0:
            log_data["episode_reward"] = sum(runner.logger.rewbuffer) / len(runner.logger.rewbuffer)
        if hasattr(runner, "logger") and len(runner.logger.lenbuffer) > 0:
            log_data["episode_length"] = sum(runner.logger.lenbuffer) / len(runner.logger.lenbuffer)

        if hasattr(env, "extras") and env.extras and "log" in env.extras:
            env_log = env.extras["log"]
            if "tracking_lin_vel" in env_log:
                log_data["velocity_tracking_error"] = 1.0 - env_log["tracking_lin_vel"].mean().item()

        log_data.update(extract_stair_metrics(env))
        log_data["iteration"] = iteration
        log_data["wall_time_hours"] = (time.time() - wall_start) / 3600.0

        if iteration % cfg.log_interval == 0:
            wandb.log(log_data, step=iteration)

        if (iteration + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_{iteration + 1}.pt")
            runner.save(ckpt_path)
            print(f"[iter {iteration + 1:>5}] checkpoint saved → {ckpt_path}", flush=True)

        mean_ep_reward = log_data.get("episode_reward", float("-inf"))
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            runner.save(cfg.best_model_path)

        if stopper.update(mean_ep_reward):
            print(
                f"[iter {iteration + 1}] Early stopping triggered — "
                f"reward {mean_ep_reward:.3f} > {cfg.early_stop_reward_threshold} "
                f"for {cfg.early_stop_patience} consecutive iters.",
                flush=True,
            )
            runner.save(cfg.best_model_path)
            wandb.summary["early_stop_iteration"] = iteration + 1
            wandb.summary["final_reward"] = mean_ep_reward
            break

        if iteration % cfg.log_interval == 0:
            fall_str = f"  fall_rate={log_data['fall_rate']:.1f}/100ep" if "fall_rate" in log_data else ""
            print(
                f"[iter {iteration + 1:>5}/{cfg.max_iterations}] "
                f"reward={mean_ep_reward:.3f}  best={best_reward:.3f}{fall_str}",
                flush=True,
            )

    wandb.summary["best_reward"] = best_reward
    wandb.summary["wall_time_hours"] = (time.time() - wall_start) / 3600.0
    run.finish()
    env.close()


if __name__ == "__main__":
    cfg = TrainingConfig()
    if args_cli.num_envs is not None:
        cfg.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        cfg.wandb_run_name = args_cli.run_name
    if args_cli.load_checkpoint is not None:
        cfg.load_checkpoint = args_cli.load_checkpoint

    train(cfg)
    simulation_app.close()
