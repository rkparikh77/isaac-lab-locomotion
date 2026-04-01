"""
Shared utilities for evaluation, trajectory recording, and video scripts.

Provides correct checkpoint loading via rsl_rl OnPolicyRunner, eliminating
the state_dict key mismatch bug that caused previous scripts to silently
run random policies.
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

TERRAIN_ENV_MAP = {
    "flat": "Isaac-Velocity-Flat-Anymal-C-v0",
    "slopes": "Isaac-Velocity-Rough-Anymal-C-v0",
    "stairs": "Isaac-Velocity-Rough-Anymal-C-v0",
    "contact_aware": "Isaac-Velocity-Rough-Anymal-C-v0",
}


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


def build_eval_runner_cfg() -> dict:
    """Build runner config matching the training architecture."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "save_interval": 999999,
        "experiment_name": "eval",
        "run_name": "eval",
        "logger": "tensorboard",
    }


def _patch_terrain(env_cfg, terrain: str) -> None:
    """Patch terrain generator sub_terrains to match training configuration."""
    import math
    import isaaclab.terrains as terrain_gen

    if terrain == "slopes":
        # Match experiments/02_slopes/train.py: pyramid slopes only, 0-15 degrees
        slope_min_rad = math.radians(0.0)
        slope_max_rad = math.radians(15.0)
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.5,
                slope_range=(slope_min_rad, slope_max_rad),
                platform_width=2.0,
                border_width=0.25,
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.5,
                slope_range=(slope_min_rad, slope_max_rad),
                platform_width=2.0,
                border_width=0.25,
            ),
        }
    elif terrain in ("stairs", "contact_aware"):
        # Match experiments/03_stairs/train.py and 04_contact_aware/train.py:
        # pyramid stairs only, step height 5-20cm, step width midpoint of 25-40cm = 32.5cm
        step_width_mid = (0.25 + 0.40) / 2.0
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.05, 0.20),
                step_width=step_width_mid,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.5,
                step_height_range=(0.05, 0.20),
                step_width=step_width_mid,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        }
    # flat: no patch needed, uses default flat terrain


def create_env(terrain: str, num_envs: int) -> ExtrasTrackingWrapper:
    """Create and wrap the Isaac Lab vectorised environment."""
    env_id = TERRAIN_ENV_MAP[terrain]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    _patch_terrain(env_cfg, terrain)
    gym_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    rsl_env = RslRlVecEnvWrapper(gym_env)
    return ExtrasTrackingWrapper(rsl_env)


def load_policy(checkpoint_path: str, env: ExtrasTrackingWrapper, device: str = "cuda"):
    """
    Load a trained policy from a checkpoint using OnPolicyRunner.

    Returns a callable: obs_tensor → action_tensor (deterministic, no noise).
    """
    runner_cfg = build_eval_runner_cfg()
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # Sanity check: verify the policy produces non-trivial actions
    # NOTE: MLPModel.forward() expects a TensorDict (obs["policy"]), not a raw tensor
    with torch.no_grad():
        obs_td = env.get_observations()
        test_actions = policy(obs_td)
        action_mag = test_actions.abs().mean().item()
        print(f"SANITY CHECK: mean |action| = {action_mag:.4f}", flush=True)
        if action_mag < 0.01:
            raise RuntimeError(
                f"Policy outputs near-zero actions ({action_mag:.6f}) — " "checkpoint may not have loaded correctly!"
            )

    return policy
