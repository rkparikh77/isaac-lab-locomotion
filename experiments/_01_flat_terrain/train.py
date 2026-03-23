"""Train AnymalC on flat terrain using RSL-RL PPO (Isaac Lab).

Run from the IsaacLab repo root so that Isaac Lab's Python path is set up:

    cd /workspace/IsaacLab
    python /workspace/isaac-lab-locomotion/experiments/01_flat_terrain/train.py \\
        --headless 2>&1 | tee /workspace/results/flat_terrain_training.log

All hyperparameters live in config.py — do not add magic numbers here.
"""

from __future__ import annotations

# ── AppLauncher MUST be imported and started before any Isaac / gym imports ────
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train AnymalC flat terrain locomotion")
# AppLauncher contributes --headless, --num_envs, --device, --enable_cameras, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# ── Post-launch imports (Isaac Sim / Gym must load after AppLauncher) ──────────
import os
import time

import gymnasium as gym
import torch
import wandb

# Register all Isaac Lab task environments (side-effect import).
import isaaclab_tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCFlatPPORunnerCfg,
)
from rsl_rl.runners import OnPolicyRunner

# ── Config import — add project root to sys.path ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._01_flat_terrain.config import FlatTerrainConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_agent_cfg(flat_cfg: FlatTerrainConfig) -> AnymalCFlatPPORunnerCfg:
    """Construct the RSL-RL runner config entirely from FlatTerrainConfig.

    No values are hard-coded here: every field is read from flat_cfg.
    """
    cfg = AnymalCFlatPPORunnerCfg()

    # Training schedule
    cfg.max_iterations = flat_cfg.max_iterations
    cfg.num_steps_per_env = flat_cfg.num_steps_per_env
    cfg.save_interval = flat_cfg.save_interval
    cfg.seed = flat_cfg.seed
    cfg.device = flat_cfg.device

    # Experiment / logging
    cfg.experiment_name = flat_cfg.experiment_name
    cfg.logger = "wandb"
    cfg.wandb_project = flat_cfg.wandb_project

    # Policy network
    cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=flat_cfg.init_noise_std,
        actor_hidden_dims=flat_cfg.actor_hidden_dims,
        critic_hidden_dims=flat_cfg.critic_hidden_dims,
        activation=flat_cfg.activation,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    # PPO algorithm
    cfg.algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=flat_cfg.value_loss_coef,
        use_clipped_value_loss=True,
        clip_param=flat_cfg.clip_param,
        entropy_coef=flat_cfg.entropy_coef,
        num_learning_epochs=flat_cfg.num_learning_epochs,
        num_mini_batches=flat_cfg.num_mini_batches,
        learning_rate=flat_cfg.learning_rate,
        schedule="adaptive",
        gamma=flat_cfg.gamma,
        lam=flat_cfg.gae_lambda,
        desired_kl=flat_cfg.desired_kl,
        max_grad_norm=flat_cfg.max_grad_norm,
    )

    return cfg


def _get_mean_episode_reward(env_wrapper: RslRlVecEnvWrapper) -> float | None:
    """Extract mean episode reward from the most recent batch of completed episodes.

    Isaac Lab's ManagerBasedRLEnv accumulates per-episode statistics in
    ``env.extras["log"]`` whenever episodes terminate. The exact key names
    depend on the reward config; we try several fallback keys.

    Returns None if no episode data is available yet.
    """
    try:
        extras = env_wrapper.extras
        log: dict = extras.get("log", {})

        # Isaac Lab locomotion envs typically expose a total episode reward
        # under one of these keys (case-sensitive, so we try both styles).
        for key in (
            "Episode Reward",
            "episode_reward",
            "EpisodeReward",
            "rew",
            "reward",
        ):
            if key in log:
                val = log[key]
                # Might be a scalar tensor, a plain float, or a numpy array.
                if hasattr(val, "mean"):
                    return float(val.mean().item())
                return float(val)
    except Exception:
        pass
    return None


def _get_velocity_tracking_error(env_wrapper: RslRlVecEnvWrapper) -> float | None:
    """Return mean velocity-tracking error from env log extras, or None."""
    try:
        log: dict = env_wrapper.extras.get("log", {})
        for key in (
            "track_lin_vel_xy_exp",
            "velocity_tracking_error",
            "vel_tracking_error",
        ):
            if key in log:
                val = log[key]
                # The reward term is a reward, not an error directly, so we
                # invert the sign so that "lower is better" makes intuitive sense.
                raw = float(val.mean().item()) if hasattr(val, "mean") else float(val)
                return 1.0 - raw  # 0 → perfect tracking, 1 → no tracking
    except Exception:
        pass
    return None


def _get_mean_episode_length(env_wrapper: RslRlVecEnvWrapper) -> float | None:
    """Return mean episode length (steps) from env log extras, or None."""
    try:
        log: dict = env_wrapper.extras.get("log", {})
        for key in ("Episode Length", "episode_length", "ep_len", "l"):
            if key in log:
                val = log[key]
                return float(val.mean().item()) if hasattr(val, "mean") else float(val)
    except Exception:
        pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    flat_cfg = FlatTerrainConfig()

    # ── Directories ─────────────────────────────────────────────────────────
    checkpoint_dir = Path(flat_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("/workspace/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment ──────────────────────────────────────────────────────────
    # Grab the env config class registered by Isaac Lab, update num_envs, then
    # pass it back to gym.make so the scene is configured before Omniverse loads.
    env_spec = gym.spec(flat_cfg.task_name)
    env_cfg_cls = env_spec.kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = flat_cfg.num_envs

    print(f"\n[Train] Task          : {flat_cfg.task_name}")
    print(f"[Train] Num envs      : {flat_cfg.num_envs}")
    print(f"[Train] Max iterations: {flat_cfg.max_iterations}")
    print(f"[Train] Checkpoint dir: {checkpoint_dir}\n")

    env = gym.make(flat_cfg.task_name, cfg=env_cfg)
    env_wrapper = RslRlVecEnvWrapper(env)

    # ── RSL-RL runner ─────────────────────────────────────────────────────────
    agent_cfg = _build_agent_cfg(flat_cfg)

    # Pass our custom checkpoint directory as the runner's log dir so that
    # the runner writes its automatic saves directly there.
    runner = OnPolicyRunner(
        env_wrapper,
        agent_cfg.to_dict(),
        log_dir=str(checkpoint_dir),
        device=agent_cfg.device,
    )

    # ── wandb initialisation ──────────────────────────────────────────────────
    # The RSL-RL runner initialises wandb internally when logger="wandb".
    # We also init here to ensure the run object is accessible for custom logs.
    if wandb.run is None:
        wandb.init(
            project=flat_cfg.wandb_project,
            name=flat_cfg.experiment_name,
            config={
                "num_envs":       flat_cfg.num_envs,
                "max_iterations": flat_cfg.max_iterations,
                "learning_rate":  flat_cfg.learning_rate,
                "gamma":          flat_cfg.gamma,
                "gae_lambda":     flat_cfg.gae_lambda,
                "clip_param":     flat_cfg.clip_param,
                "terrain_type":   flat_cfg.terrain_type,
                "reward_weights": flat_cfg.reward_weights,
            },
        )

    # ── Training loop with early stopping ─────────────────────────────────────
    # We run training in chunks of `wandb_log_every` iterations so we can:
    #   1. Log custom wandb metrics at a fine granularity.
    #   2. Check the early-stopping condition between chunks.
    #
    # The runner's own logging (tensorboard + wandb) is unaffected: it runs on
    # its internal schedule regardless of how we slice the outer loop.

    LOG_CHUNK = flat_cfg.wandb_log_every           # iterations per chunk
    consecutive_above_threshold = 0                # consecutive iters above reward threshold
    best_mean_reward = float("-inf")
    best_model_saved = False
    training_start = time.time()

    print(f"[Train] Starting training — early stop at {flat_cfg.early_stop_reward:.1f} "
          f"reward for {flat_cfg.early_stop_consecutive_iters} consecutive iters.\n")

    iters_done = 0
    while iters_done < flat_cfg.max_iterations:
        chunk = min(LOG_CHUNK, flat_cfg.max_iterations - iters_done)
        is_first_chunk = iters_done == 0

        # Run one chunk of PPO updates.
        runner.learn(
            num_learning_iterations=chunk,
            init_at_random_ep_len=is_first_chunk,
        )
        iters_done += chunk

        # ── Gather metrics ───────────────────────────────────────────────────
        mean_reward = _get_mean_episode_reward(env_wrapper)
        ep_length   = _get_mean_episode_length(env_wrapper)
        vel_error   = _get_velocity_tracking_error(env_wrapper)

        # ── Custom wandb log ─────────────────────────────────────────────────
        if wandb.run is not None:
            log_dict: dict = {"iteration": iters_done}
            if mean_reward is not None:
                log_dict["episode_reward"] = mean_reward
            if ep_length is not None:
                log_dict["episode_length"] = ep_length
            if vel_error is not None:
                log_dict["velocity_tracking_error"] = vel_error
            wandb.log(log_dict, step=iters_done)

        # ── Console heartbeat ────────────────────────────────────────────────
        elapsed = time.time() - training_start
        rew_str = f"{mean_reward:.3f}" if mean_reward is not None else "n/a"
        print(
            f"[iter {iters_done:>5}/{flat_cfg.max_iterations}]  "
            f"reward={rew_str}  "
            f"consecutive_good={consecutive_above_threshold}  "
            f"elapsed={elapsed:.0f}s"
        )

        # ── Periodic checkpoint (mirrors runner's save_interval) ─────────────
        # The runner saves automatically via save_interval; we add a named copy
        # here for convenience.
        if iters_done % flat_cfg.save_interval == 0:
            ckpt_path = checkpoint_dir / f"model_{iters_done:05d}.pt"
            runner.save(str(ckpt_path))
            print(f"[Train] Checkpoint → {ckpt_path}")

        # ── Early stopping ───────────────────────────────────────────────────
        if mean_reward is not None:
            if mean_reward > flat_cfg.early_stop_reward:
                consecutive_above_threshold += chunk
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
            else:
                consecutive_above_threshold = 0

            if consecutive_above_threshold >= flat_cfg.early_stop_consecutive_iters:
                best_path = checkpoint_dir / "best_model.pt"
                runner.save(str(best_path))
                best_model_saved = True
                print(
                    f"\n[EarlyStopping] Triggered at iteration {iters_done}.\n"
                    f"  Mean reward : {mean_reward:.3f}  (threshold: {flat_cfg.early_stop_reward})\n"
                    f"  Best model  → {best_path}\n"
                )
                if wandb.run is not None:
                    wandb.summary["early_stop_iteration"] = iters_done
                    wandb.summary["best_mean_reward"] = best_mean_reward
                break

    # ── Final save if early stopping was not triggered ─────────────────────────
    if not best_model_saved:
        final_path = checkpoint_dir / "final_model.pt"
        runner.save(str(final_path))
        print(f"\n[Train] Training complete. Final model → {final_path}")
        if wandb.run is not None:
            wandb.summary["best_mean_reward"] = best_mean_reward

    total_time = time.time() - training_start
    print(f"[Train] Total training time: {total_time / 3600:.2f} h")

    if wandb.run is not None:
        wandb.finish()

    # ── Shutdown ───────────────────────────────────────────────────────────────
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
