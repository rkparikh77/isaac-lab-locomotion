"""
Hyperparameter configuration for contact-aware AnymalC training.

Terrain : pyramid stairs (5–20 cm steps) — same as Stage 3.
Loads   : /workspace/checkpoints/stairs/best_model.pt
Saves   : /workspace/checkpoints/contact_aware/

All contact-aware reward weights live here so ablation conditions can be
parameterised by importing this config and overriding fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Base Isaac Lab reward scaling factors (inherited from stairs stage)."""

    lin_vel_tracking: float = 1.0
    ang_vel_tracking: float = 0.5
    lin_vel_z_penalty: float = -2.0
    ang_vel_xy_penalty: float = -0.02
    joint_torque_penalty: float = -1.0e-5
    joint_accel_penalty: float = -2.5e-7
    action_rate_penalty: float = -0.01
    feet_air_time_bonus: float = 0.5
    undesired_contact_penalty: float = -1.0
    flat_orientation_penalty: float = -2.0


@dataclass
class ContactRewardWeights:
    """Weights for the additional contact-aware reward terms."""

    velocity_tracking: float = 1.0
    foot_slip_penalty: float = -0.5
    terrain_clearance: float = 0.3
    contact_timing: float = -0.2
    energy_penalty: float = -1.0e-4


@dataclass
class PPOConfig:
    """RSL-RL PPO algorithm hyperparameters."""

    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    num_steps_per_env: int = 24


@dataclass
class NetworkConfig:
    """Actor-critic MLP architecture (identical to previous stages)."""

    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class StairTerrainConfig:
    """Stair terrain parameters (same as Stage 3)."""

    step_height_min_m: float = 0.05
    step_height_max_m: float = 0.20
    step_width_min_m: float = 0.25
    step_width_max_m: float = 0.40


@dataclass
class TrainingConfig:
    """Top-level configuration for contact-aware training stage."""

    # --- Environment ---
    num_envs: int = 4096
    terrain_type: str = "stairs"
    env_id: str = "Isaac-Velocity-Rough-Anymal-C-v0"
    device: str = "cuda"

    # --- Checkpoint loading ---
    load_checkpoint: str = "/workspace/checkpoints/stairs/best_model.pt"

    # --- Training schedule ---
    max_iterations: int = 300
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: str = "/workspace/checkpoints/contact_aware"
    best_model_path: str = "/workspace/checkpoints/contact_aware/best_model.pt"

    # --- Early stopping ---
    # Threshold is in rsl_rl rewbuffer units (undiscounted episode returns,
    # typically 500–2000 for stair terrain starting from a pretrained checkpoint).
    early_stop_reward_threshold: float = 2000.0
    early_stop_patience: int = 50

    # --- Contact-aware terms: which are enabled ---
    # None = all enabled; set of names = only those active
    enabled_contact_terms: list[str] | None = None  # None → all

    # --- wandb ---
    wandb_project: str = "isaac-lab-locomotion"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["anymal-c", "stairs", "contact-aware", "ppo"])

    # --- Sub-configs ---
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    contact_reward_weights: ContactRewardWeights = field(default_factory=ContactRewardWeights)
    terrain: StairTerrainConfig = field(default_factory=StairTerrainConfig)
