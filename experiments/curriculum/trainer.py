"""
Curriculum trainer — flat → slopes → stairs sequential training.

Advances to the next terrain stage when the current stage meets its
advancement thresholds (min_reward AND max_reward_std), or when the
stage's max_iterations budget is exhausted.

Outputs
-------
results/curriculum_log.json
    [{stage, start_iter, end_iter, advancement_trigger, timestamp}, ...]

Usage
-----
    python -m experiments.curriculum.trainer
    python -m experiments.curriculum.trainer --headless
    python -m experiments.curriculum.trainer --headless --num_envs 256 --max_iterations_per_stage 50
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from datetime import datetime, timezone

# ── Isaac Lab app launcher must come before any omni imports ──────────────────
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sequential curriculum trainer: flat → slopes → stairs.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override num_envs for all stages",
)
parser.add_argument(
    "--max_iterations_per_stage",
    type=int,
    default=None,
    help="Override max_iterations for every stage (useful for smoke tests)",
)
parser.add_argument(
    "--skip_to_stage",
    type=str,
    default=None,
    choices=["flat", "slopes", "stairs"],
    help="Start curriculum at this stage (assumes earlier checkpoints already exist)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Remaining imports (after app launch) ─────────────────────────────────────
import gymnasium as gym
import torch
import wandb

import omni.isaac.lab_tasks  # noqa: F401

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner

from experiments.curriculum.config import (
    ADVANCEMENT_THRESHOLDS,
    STAGE_ORDER,
    STAGE_CONFIG_MAP,
    AdvancementThreshold,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared training helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_runner_cfg(cfg) -> RslRlOnPolicyRunnerCfg:
    p = cfg.ppo
    n = cfg.network
    return RslRlOnPolicyRunnerCfg(
        algorithm=dict(
            class_name="PPO",
            clip_param=p.clip_param,
            entropy_coef=p.entropy_coef,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            learning_rate=p.learning_rate,
            max_grad_norm=p.max_grad_norm,
            num_learning_epochs=p.num_learning_epochs,
            num_mini_batches=p.num_mini_batches,
            schedule="adaptive",
            value_loss_coef=p.value_loss_coef,
        ),
        policy=dict(
            class_name=("ActorCriticRecurrent" if "lstm" in n.activation.lower() else "ActorCritic"),
            init_noise_std=n.init_noise_std,
            actor_hidden_dims=n.actor_hidden_dims,
            critic_hidden_dims=n.critic_hidden_dims,
            activation=n.activation,
        ),
        num_steps_per_env=p.num_steps_per_env,
        max_iterations=cfg.max_iterations,
        save_interval=cfg.checkpoint_interval,
        experiment_name=cfg.wandb_run_name or f"{cfg.terrain_type}_curriculum",
        run_name=cfg.wandb_run_name or "",
        logger="wandb",
        neptune_project="",
        wandb_project=cfg.wandb_project,
        resume=False,
        load_run="",
        load_checkpoint="",
    )


def _apply_reward_weights(env: RslRlVecEnvWrapper, rw) -> None:
    reward_manager = env.unwrapped.reward_manager
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
# Per-stage runner
# ─────────────────────────────────────────────────────────────────────────────


def run_stage(
    cfg,
    threshold: AdvancementThreshold,
    load_checkpoint: str | None = None,
) -> tuple[int, str]:
    """
    Run one curriculum stage until the threshold is met or the iteration budget runs out.

    Returns
    -------
    end_iter : int
        Iteration at which the stage ended (1-indexed).
    advancement_trigger : str
        ``"threshold_met"`` if both advancement criteria were satisfied,
        ``"budget_exhausted"`` otherwise.
    """
    pathlib.Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags + ["curriculum"],
        config={
            "num_envs": cfg.num_envs,
            "terrain_type": cfg.terrain_type,
            "max_iterations": cfg.max_iterations,
            "load_checkpoint": load_checkpoint or getattr(cfg, "load_checkpoint", None),
            "advancement_min_reward": threshold.min_reward,
            "advancement_max_reward_std": threshold.max_reward_std,
            **vars(cfg.ppo),
            **vars(cfg.network),
            **vars(cfg.reward_weights),
        },
    )

    env = gym.make(cfg.env_id, num_envs=cfg.num_envs, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    _apply_reward_weights(env, cfg.reward_weights)

    runner_cfg = _build_runner_cfg(cfg)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=cfg.device)

    # Warm-start
    ckpt_to_load = load_checkpoint or getattr(cfg, "load_checkpoint", None)
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"  Loading checkpoint: {ckpt_to_load}")
        runner.load(ckpt_to_load)
    elif ckpt_to_load:
        print(f"  WARNING: checkpoint '{ckpt_to_load}' not found — starting from scratch.")

    best_reward: float = float("-inf")
    # Rolling window for std estimation (last 20 reward samples)
    reward_window: list[float] = []
    _WINDOW = 20
    advancement_trigger = "budget_exhausted"
    end_iter = cfg.max_iterations
    wall_start = time.time()

    for iteration in range(cfg.max_iterations):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

        log_data: dict = {}
        mean_reward = float("-inf")
        reward_std = float("inf")

        if hasattr(runner, "alg") and hasattr(runner.alg, "storage"):
            storage = runner.alg.storage
            if hasattr(storage, "rewards") and storage.rewards.numel() > 0:
                mean_reward = storage.rewards.mean().item()
                reward_std = storage.rewards.std().item()
                log_data["episode_reward"] = mean_reward
                log_data["episode_reward_std"] = reward_std
            if hasattr(storage, "ep_length"):
                log_data["episode_length"] = storage.ep_length.float().mean().item()

        if hasattr(env, "extras") and "log" in env.extras:
            env_log = env.extras["log"]
            if "tracking_lin_vel" in env_log:
                log_data["velocity_tracking_error"] = 1.0 - env_log["tracking_lin_vel"].mean().item()

        log_data["iteration"] = iteration
        log_data["wall_time_hours"] = (time.time() - wall_start) / 3600.0

        if iteration % cfg.log_interval == 0:
            wandb.log(log_data, step=iteration)

        if (iteration + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_{iteration + 1}.pt")
            runner.save(ckpt_path)

        if mean_reward > best_reward:
            best_reward = mean_reward
            runner.save(cfg.best_model_path)

        # ── Advancement check ─────────────────────────────────────────────────
        reward_window.append(mean_reward)
        if len(reward_window) > _WINDOW:
            reward_window.pop(0)

        if len(reward_window) == _WINDOW:
            window_mean = sum(reward_window) / _WINDOW
            window_std = (sum((r - window_mean) ** 2 for r in reward_window) / _WINDOW) ** 0.5
            if window_mean >= threshold.min_reward and window_std <= threshold.max_reward_std:
                print(
                    f"  [{cfg.terrain_type}] Advancement threshold met at iter {iteration + 1} "
                    f"(window_mean={window_mean:.3f} >= {threshold.min_reward}, "
                    f"window_std={window_std:.3f} <= {threshold.max_reward_std})"
                )
                runner.save(cfg.best_model_path)
                advancement_trigger = "threshold_met"
                end_iter = iteration + 1
                wandb.summary["advancement_trigger"] = advancement_trigger
                wandb.summary["advancement_iteration"] = end_iter
                break

        if iteration % cfg.log_interval == 0:
            print(
                f"  [{cfg.terrain_type}] iter {iteration + 1:>5}/{cfg.max_iterations}  "
                f"reward={mean_reward:.3f}  best={best_reward:.3f}"
            )

    wandb.summary["best_reward"] = best_reward
    wandb.summary["wall_time_hours"] = (time.time() - wall_start) / 3600.0
    run.finish()
    env.close()

    return end_iter, advancement_trigger


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum log helpers
# ─────────────────────────────────────────────────────────────────────────────

_LOG_PATH = pathlib.Path("results/curriculum_log.json")


def _load_log() -> list[dict]:
    if _LOG_PATH.exists():
        with open(_LOG_PATH) as f:
            return json.load(f)
    return []


def _append_log(entry: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_log()
    entries.append(entry)
    with open(_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"  curriculum log updated → {_LOG_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # Determine which stages to run
    skip_to = args_cli.skip_to_stage
    stages_to_run = STAGE_ORDER if skip_to is None else STAGE_ORDER[STAGE_ORDER.index(skip_to) :]

    # Cumulative iteration counter (for curriculum log start_iter)
    global_iter = 0

    for stage_name in stages_to_run:
        ConfigClass = STAGE_CONFIG_MAP[stage_name]
        cfg = ConfigClass()

        # Apply CLI overrides
        if args_cli.num_envs is not None:
            cfg.num_envs = args_cli.num_envs
        if args_cli.max_iterations_per_stage is not None:
            cfg.max_iterations = args_cli.max_iterations_per_stage

        threshold = ADVANCEMENT_THRESHOLDS[stage_name]

        print(f"\n{'='*60}")
        print(f"  STAGE: {stage_name.upper()}")
        print(
            f"  max_iterations={cfg.max_iterations}  "
            f"threshold=reward>={threshold.min_reward} std<={threshold.max_reward_std}"
        )
        print(f"{'='*60}\n")

        start_iter = global_iter
        end_iter_relative, trigger = run_stage(cfg, threshold)
        global_iter += end_iter_relative

        _append_log(
            {
                "stage": stage_name,
                "start_iter": start_iter,
                "end_iter": global_iter,
                "advancement_trigger": trigger,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        print(f"\n  Stage '{stage_name}' complete: {trigger}  " f"(iters {start_iter}–{global_iter})\n")

    print(f"\nCurriculum complete. Log saved to {_LOG_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
