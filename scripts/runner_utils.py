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


class TensorDictVecEnvWrapper(VecEnv):
    """Adapts Isaac Lab's RslRlVecEnvWrapper (flat tensors) to rsl_rl v5
    VecEnv interface which requires TensorDicts."""

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


def create_env(terrain: str, num_envs: int) -> TensorDictVecEnvWrapper:
    """Create and wrap the Isaac Lab vectorised environment."""
    env_id = TERRAIN_ENV_MAP[terrain]
    env_cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    gym_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    rsl_env = RslRlVecEnvWrapper(gym_env)
    return TensorDictVecEnvWrapper(rsl_env)


def load_policy(checkpoint_path: str, env: TensorDictVecEnvWrapper, device: str = "cuda"):
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
