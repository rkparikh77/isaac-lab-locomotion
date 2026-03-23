"""All hyperparameters for the flat-terrain AnymalC locomotion experiment.

Every number that controls training lives here.  train.py reads this file
and must not hard-code any magic numbers.

Usage:
    from experiments._01_flat_terrain.config import FlatTerrainConfig
    cfg = FlatTerrainConfig()
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlatTerrainConfig:
    # ── Environment ────────────────────────────────────────────────────────────
    task_name: str = "Isaac-Velocity-Flat-Anymal-C-v0"
    terrain_type: str = "flat"
    num_envs: int = 4096

    # ── Training schedule ─────────────────────────────────────────────────────
    max_iterations: int = 1500
    num_steps_per_env: int = 24        # horizon length per rollout (N_steps in PPO)

    # ── PPO algorithm ─────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE λ
    clip_param: float = 0.2            # PPO ε-clip
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5       # gradient updates per rollout
    num_mini_batches: int = 4
    desired_kl: float = 0.01           # adaptive learning-rate target KL
    max_grad_norm: float = 1.0         # gradient clipping

    # ── Policy network ────────────────────────────────────────────────────────
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128])
    activation: str = "elu"
    init_noise_std: float = 1.0

    # ── Reward weights ────────────────────────────────────────────────────────
    # Keys match the reward term names in the AnymalCFlat env config.
    # Set to 0.0 to disable a term without modifying the env config.
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "track_lin_vel_xy_exp":   1.0,     # forward / lateral velocity tracking
        "track_ang_vel_z_exp":    0.5,     # yaw-rate tracking
        "lin_vel_z_l2":          -2.0,     # penalise vertical CoM velocity
        "ang_vel_xy_l2":         -0.05,    # penalise roll/pitch rate
        "dof_torques_l2":        -1.0e-5,  # energy efficiency
        "dof_acc_l2":            -2.5e-7,  # joint jerk
        "action_rate_l2":        -0.01,    # action smoothness
        "feet_air_time":          0.25,    # encourage lifting feet
        "undesired_contacts":    -1.0,     # penalise shin/thigh contacts
        "flat_orientation_l2":   -0.0,     # (unused on flat terrain — set via env cfg)
    })

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "/workspace/checkpoints/flat"
    save_interval: int = 100           # save checkpoint every N iterations

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stop_reward: float = 8.0         # mean episode reward threshold
    early_stop_consecutive_iters: int = 50 # must exceed threshold for this many iters

    # ── Logging ───────────────────────────────────────────────────────────────
    experiment_name: str = "anymal_c_flat"
    wandb_project: str = "motionmd-locomotion"
    wandb_log_every: int = 10              # log to wandb every N iterations

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda:0"
