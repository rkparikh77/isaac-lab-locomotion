"""
Hyperparameter configuration for stair-terrain AnymalC velocity-tracking experiment.

Terrain: pyramid stairs — step height 5–20 cm, step width 25–40 cm.
Loads from: /workspace/checkpoints/slopes/best_model.pt
Trains for 1 000 additional iterations.

All magic numbers live here. No numeric literals should appear in train.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Per-term reward scaling factors tuned for stair traversal."""

    lin_vel_tracking: float = 1.0
    ang_vel_tracking: float = 0.5
    lin_vel_z_penalty: float = -2.0
    # Softened — some pitch is expected when climbing stairs
    ang_vel_xy_penalty: float = -0.02
    joint_torque_penalty: float = -1.0e-5
    joint_accel_penalty: float = -2.5e-7
    action_rate_penalty: float = -0.01
    # Boosted — high foot lift is critical for stair clearance
    feet_air_time_bonus: float = 0.5
    undesired_contact_penalty: float = -1.0
    flat_orientation_penalty: float = -2.0


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
    """Actor-critic MLP architecture."""

    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0


@dataclass
class StairTerrainConfig:
    """Parameters for the pyramid-stair terrain sub-terrains."""

    step_height_min_m: float = 0.05  # 5 cm
    step_height_max_m: float = 0.20  # 20 cm
    step_width_min_m: float = 0.25  # 25 cm (midpoint passed to Isaac Lab)
    step_width_max_m: float = 0.40  # 40 cm


@dataclass
class TrainingConfig:
    """Top-level training configuration — single source of truth for stair stage."""

    # --- Environment ---
    num_envs: int = 4096
    terrain_type: str = "stairs"
    # Use Rough env — terrain generator patched to stairs-only in train.py
    env_id: str = "Isaac-Velocity-Rough-Anymal-C-v0"
    device: str = "cuda"

    # --- Checkpoint loading ---
    load_checkpoint: str = "/workspace/checkpoints/slopes/best_model.pt"

    # --- Training schedule ---
    max_iterations: int = 1000
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_dir: str = "/workspace/checkpoints/stairs"
    best_model_path: str = "/workspace/checkpoints/stairs/best_model.pt"

    # --- Early stopping ---
    early_stop_reward_threshold: float = 6.5
    early_stop_patience: int = 50

    # --- wandb ---
    wandb_project: str = "isaac-lab-locomotion"
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["anymal-c", "stairs", "ppo"])

    # --- Sub-configs ---
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    terrain: StairTerrainConfig = field(default_factory=StairTerrainConfig)
