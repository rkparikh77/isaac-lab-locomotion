"""
Contact-aware AnymalC training script.

Builds on the stair-terrain training pipeline and adds a contact-aware
RewardManager whose five terms are computed at each env.step() call and
added to the base Isaac Lab reward.  Every term's mean value is logged to
wandb at each log_interval so ablation results can be analysed post-hoc.

Usage:
    python /workspace/isaac-lab-locomotion/experiments/04_contact_aware/train.py --headless
    python /workspace/isaac-lab-locomotion/experiments/04_contact_aware/train.py \
        --headless --enabled_terms velocity_tracking foot_slip_penalty
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

# ── Isaac Lab app launcher must come before any omni imports ──────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train AnymalC on stairs with contact-aware reward.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Path to pretrained checkpoint (overrides config default)",
)
parser.add_argument(
    "--enabled_terms",
    nargs="*",
    default=None,
    help="Space-separated list of contact-aware term names to enable "
    "(default: all). E.g. --enabled_terms velocity_tracking foot_slip_penalty",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Override checkpoint output directory",
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from environments.contact_aware_reward import build_contact_aware_manager

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from config import (
    ContactRewardWeights,
    NetworkConfig,
    PPOConfig,
    RewardWeights,
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# TensorDict VecEnv bridge (same as previous stages)
# ─────────────────────────────────────────────────────────────────────────────


class TensorDictVecEnvWrapper(VecEnv):
    def __init__(self, env: RslRlVecEnvWrapper) -> None:
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.device = env.device
        self.cfg = getattr(env, "cfg", {})
        self.extras = {}

    def _to_td(self, obs: torch.Tensor) -> TensorDict:
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        obs, _ = self._env.get_observations()
        self.episode_length_buf = self._env.episode_length_buf
        return self._to_td(obs)

    def step(self, actions: torch.Tensor):
        obs, rewards, dones, extras = self._env.step(actions)
        self.episode_length_buf = self._env.episode_length_buf
        self.extras = extras
        return self._to_td(obs), rewards, dones, extras

    def reset(self):
        obs, extras = self._env.reset()
        return self._to_td(obs), extras

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Contact-aware wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _extract_contact_data(isaac_env) -> dict:
    """
    Extract contact forces, foot kinematics, and joint torques from the
    underlying Isaac Lab environment.

    Returns an empty dict for any data that cannot be read — reward terms
    will fall back gracefully.
    """
    data: dict = {}
    try:
        robot = isaac_env.scene["robot"]
        rd = robot.data

        # --- joint torques [N, num_dof] ---
        if hasattr(rd, "applied_torque"):
            data["joint_torques"] = rd.applied_torque

        # --- base velocity in body frame [N, 3] ---
        if hasattr(rd, "root_lin_vel_b"):
            data["base_velocity"] = rd.root_lin_vel_b

        # --- foot body positions + velocities ---
        foot_ids = None
        for pattern in [
            [".*foot", ".*FOOT"],
            [".*_FOOT"],
            ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
        ]:
            try:
                ids, _ = robot.find_bodies(pattern)
                if len(ids) >= 4:
                    foot_ids = ids[:4]
                    break
            except Exception:
                continue

        if foot_ids is not None:
            if hasattr(rd, "body_pos_w"):
                data["foot_positions"] = rd.body_pos_w[:, foot_ids, :]
            if hasattr(rd, "body_lin_vel_w"):
                data["foot_velocities"] = rd.body_lin_vel_w[:, foot_ids, :]
            elif hasattr(rd, "body_state_w"):
                # state_w: [N, bodies, 13] — pos(3), quat(4), lin_vel(3), ang_vel(3)
                data["foot_velocities"] = rd.body_state_w[:, foot_ids, 7:10]

    except Exception:
        pass

    # --- contact forces (foot bodies only) ---
    # The velocity env cfg registers the sensor as scene.sensors["contact_forces"]
    # covering all robot bodies.  We filter to the 4 foot bodies via find_bodies().
    try:
        sensor = isaac_env.scene.sensors["contact_forces"]
        foot_ids, _ = sensor.find_bodies(".*FOOT")
        data["contact_forces"] = sensor.data.net_forces_w[:, foot_ids, :]  # [N, 4, 3]
    except Exception:
        pass

    # --- velocity commands ---
    try:
        data["commands"] = isaac_env.command_manager.get_command("base_velocity")
    except Exception:
        pass

    return data


class ContactAwareVecEnvWrapper(TensorDictVecEnvWrapper):
    """
    Extends TensorDictVecEnvWrapper to inject contact-aware reward terms.

    At each step the RewardManager computes additional reward from contact
    force / foot kinematics data extracted from the Isaac Lab scene.  The
    per-term breakdown is stored in ``self.last_breakdown`` for logging.
    """

    def __init__(self, env: RslRlVecEnvWrapper, reward_manager) -> None:
        super().__init__(env)
        self.reward_manager = reward_manager
        self.last_breakdown: dict[str, float] = {}

    def step(self, actions: torch.Tensor):
        obs_td, rewards, dones, extras = super().step(actions)

        try:
            isaac_env = self._env.unwrapped
            contacts = _extract_contact_data(isaac_env)
            obs_tensor = obs_td["policy"]
            extra_reward, breakdown = self.reward_manager.compute(obs_tensor, actions, contacts)
            rewards = rewards + extra_reward.to(rewards.device, rewards.dtype)
            self.last_breakdown = breakdown
        except Exception:
            self.last_breakdown = {}

        return obs_td, rewards, dones, extras


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
        "experiment_name": cfg.wandb_run_name or "contact_aware",
        "run_name": cfg.wandb_run_name or "",
        "logger": "tensorboard",
        "wandb_project": cfg.wandb_project,
    }


def make_env(cfg: TrainingConfig) -> ContactAwareVecEnvWrapper:
    """Create stair env with contact-aware reward wrapper."""
    import isaaclab.terrains as terrain_gen

    env_cfg = load_cfg_from_registry(cfg.env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs

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
    rsl_env = RslRlVecEnvWrapper(env)

    # Build reward manager
    crw = cfg.contact_reward_weights
    weight_overrides = {
        "velocity_tracking": crw.velocity_tracking,
        "foot_slip_penalty": crw.foot_slip_penalty,
        "terrain_clearance": crw.terrain_clearance,
        "contact_timing": crw.contact_timing,
        "energy_penalty": crw.energy_penalty,
    }
    enabled = set(cfg.enabled_contact_terms) if cfg.enabled_contact_terms is not None else None
    reward_manager = build_contact_aware_manager(
        enabled_terms=enabled,
        weight_overrides=weight_overrides,
    )

    return ContactAwareVecEnvWrapper(rsl_env, reward_manager)


def apply_base_reward_weights(env: ContactAwareVecEnvWrapper, rw: RewardWeights) -> None:
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


def _partial_load_checkpoint(runner: OnPolicyRunner, path: str, device: str) -> None:
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
        f"  Transferred {len(transferred)} tensors, skipped {len(skipped)}.",
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


def train(cfg: TrainingConfig) -> float:
    """Run training, return best episode reward achieved."""
    pathlib.Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    active_terms = cfg.enabled_contact_terms or [
        "velocity_tracking",
        "foot_slip_penalty",
        "terrain_clearance",
        "contact_timing",
        "energy_penalty",
    ]

    run = wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        config={
            "num_envs": cfg.num_envs,
            "terrain_type": cfg.terrain_type,
            "max_iterations": cfg.max_iterations,
            "load_checkpoint": cfg.load_checkpoint,
            "enabled_contact_terms": active_terms,
            **vars(cfg.ppo),
            **vars(cfg.network),
            **vars(cfg.reward_weights),
            **vars(cfg.contact_reward_weights),
            **vars(cfg.terrain),
        },
    )

    env = make_env(cfg)
    apply_base_reward_weights(env, cfg.reward_weights)

    runner_cfg = build_runner_cfg(cfg)
    rsl_log_dir = os.path.join(cfg.checkpoint_dir, "rsl_rl_logs")
    os.makedirs(rsl_log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=rsl_log_dir, device=cfg.device)

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

        # Per-term contact reward breakdown (from last step of this iteration)
        log_data.update(env.last_breakdown)

        log_data["iteration"] = iteration
        log_data["wall_time_hours"] = (time.time() - wall_start) / 3600.0

        if iteration % cfg.log_interval == 0:
            wandb.log(log_data, step=iteration)

        if (iteration + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"model_{iteration + 1}.pt")
            runner.save(ckpt_path)
            print(f"[iter {iteration + 1:>5}] checkpoint → {ckpt_path}", flush=True)

        mean_ep_reward = log_data.get("episode_reward", float("-inf"))
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            runner.save(cfg.best_model_path)

        if stopper.update(mean_ep_reward):
            print(
                f"[iter {iteration + 1}] Early stopping — "
                f"reward {mean_ep_reward:.3f} > {cfg.early_stop_reward_threshold} "
                f"for {cfg.early_stop_patience} iters.",
                flush=True,
            )
            runner.save(cfg.best_model_path)
            wandb.summary["early_stop_iteration"] = iteration + 1
            wandb.summary["final_reward"] = mean_ep_reward
            break

        if iteration % cfg.log_interval == 0:
            # Build a short breakdown string for console
            bd = env.last_breakdown
            bd_str = "  ".join(
                f"{k.split('/')[-1]}={v:.3f}"
                for k, v in bd.items()
                if k.startswith("reward/") and not k.endswith("total_contact_aware")
            )
            print(
                f"[iter {iteration + 1:>5}/{cfg.max_iterations}] "
                f"reward={mean_ep_reward:.3f}  best={best_reward:.3f}  {bd_str}",
                flush=True,
            )

    wandb.summary["best_reward"] = best_reward
    wandb.summary["wall_time_hours"] = (time.time() - wall_start) / 3600.0
    run.finish()
    env.close()

    return best_reward


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    if args_cli.enabled_terms is not None:
        cfg.enabled_contact_terms = args_cli.enabled_terms
    if args_cli.checkpoint_dir is not None:
        cfg.checkpoint_dir = args_cli.checkpoint_dir
        cfg.best_model_path = os.path.join(args_cli.checkpoint_dir, "best_model.pt")

    train(cfg)
    simulation_app.close()
