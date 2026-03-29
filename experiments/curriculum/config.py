"""
Curriculum configuration — advancement thresholds and stage config aggregation.

Advancement rule per stage: advance when
    mean_episode_reward >= min_reward  AND  reward_std <= max_reward_std

Usage (import in trainer.py or notebooks):
    from experiments.curriculum.config import ADVANCEMENT_THRESHOLDS, STAGE_ORDER
    from experiments.curriculum.config import FlatConfig, SlopesConfig, StairsConfig
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────────
# Advancement thresholds
# ─────────────────────────────────────────────────────────────────────────────


class AdvancementThreshold(NamedTuple):
    """Criteria that must *both* be satisfied to leave a stage."""

    min_reward: float  # mean episode reward must reach this value
    max_reward_std: float  # reward std across envs must fall below this value


# (min_reward, max_reward_std) per stage
ADVANCEMENT_THRESHOLDS: dict[str, AdvancementThreshold] = {
    "flat": AdvancementThreshold(min_reward=7.5, max_reward_std=1.0),
    "slopes": AdvancementThreshold(min_reward=7.0, max_reward_std=1.2),
    "stairs": AdvancementThreshold(min_reward=6.5, max_reward_std=1.5),
}

# Canonical stage order for sequential training
STAGE_ORDER: list[str] = ["flat", "slopes", "stairs"]


# ─────────────────────────────────────────────────────────────────────────────
# Stage config imports
#
# Directories beginning with digits are not valid Python identifiers, so we
# use importlib rather than a bare "from" statement.
# ─────────────────────────────────────────────────────────────────────────────

_flat_mod = importlib.import_module("experiments.01_flat_terrain.config")
_slopes_mod = importlib.import_module("experiments.02_slopes.config")
_stairs_mod = importlib.import_module("experiments.03_stairs.config")

#: TrainingConfig class for the flat stage
FlatConfig = _flat_mod.TrainingConfig
#: TrainingConfig class for the slope stage
SlopesConfig = _slopes_mod.TrainingConfig
#: TrainingConfig class for the stair stage
StairsConfig = _stairs_mod.TrainingConfig

# Convenience map: stage name → config class
STAGE_CONFIG_MAP: dict[str, type] = {
    "flat": FlatConfig,
    "slopes": SlopesConfig,
    "stairs": StairsConfig,
}
