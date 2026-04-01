"""
Contact-aware reward module for AnymalC locomotion training.

Provides a RewardManager with toggleable reward terms that operate *additively*
on top of the base Isaac Lab reward system. All terms are computed from
observation, action, and contact tensors passed at each step.

Interface
---------
    mgr = RewardManager()
    mgr.add_term("foot_slip_penalty", foot_slip_penalty, weight=-0.5)
    total_reward, breakdown = mgr.compute(obs, actions, contacts)
    wandb.log(breakdown, step=iteration)

contacts dict keys (all optional — terms degrade gracefully if absent):
    contact_forces   : Tensor [num_envs, num_feet, 3]
    foot_velocities  : Tensor [num_envs, num_feet, 3]
    foot_positions   : Tensor [num_envs, num_feet, 3]  (world-frame z used)
    joint_torques    : Tensor [num_envs, num_dof]
    commands         : Tensor [num_envs, >=2]   (vx, vy commands)
    base_velocity    : Tensor [num_envs, >=2]   (actual vx, vy)
"""

from __future__ import annotations

from typing import Callable, Set

import torch
from torch import Tensor

# ─────────────────────────────────────────────────────────────────────────────
# Reward Manager
# ─────────────────────────────────────────────────────────────────────────────


class RewardManager:
    """
    Composable, weight-scaled reward manager.

    Each registered term is a callable:
        fn(obs, actions, contacts) -> Tensor[num_envs]

    Terms are toggled off by setting weight=0.0.
    """

    def __init__(self) -> None:
        self._terms: dict[str, tuple[Callable, float]] = {}

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_term(self, name: str, fn: Callable, weight: float) -> None:
        """Register or overwrite a reward term."""
        self._terms[name] = (fn, weight)

    def remove_term(self, name: str) -> None:
        self._terms.pop(name, None)

    def set_weight(self, name: str, weight: float) -> None:
        if name in self._terms:
            fn, _ = self._terms[name]
            self._terms[name] = (fn, weight)

    @property
    def active_terms(self) -> list[str]:
        return [n for n, (_, w) in self._terms.items() if w != 0.0]

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(
        self,
        obs: Tensor,
        actions: Tensor,
        contacts: dict,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Compute total contact-aware reward and per-term breakdown.

        Returns
        -------
        total_reward : Tensor [num_envs]
        breakdown    : dict  {term_name: mean_float, ...}
        """
        num_envs = obs.shape[0]
        total = torch.zeros(num_envs, device=obs.device, dtype=obs.dtype)
        breakdown: dict[str, float] = {}

        for name, (fn, weight) in self._terms.items():
            if weight == 0.0:
                breakdown[f"reward/{name}"] = 0.0
                continue
            try:
                raw = fn(obs, actions, contacts)  # [num_envs]
                raw = raw.to(device=obs.device, dtype=obs.dtype)
                weighted = weight * raw
                total = total + weighted
                breakdown[f"reward/{name}"] = float(raw.mean().item())
                breakdown[f"reward_weighted/{name}"] = float(weighted.mean().item())
            except Exception:
                # Graceful degradation when tensors are unavailable
                breakdown[f"reward/{name}"] = float("nan")

        breakdown["reward/total_contact_aware"] = float(total.mean().item())
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Reward term functions
# (all have signature: fn(obs, actions, contacts) -> Tensor[num_envs])
# ─────────────────────────────────────────────────────────────────────────────


def velocity_tracking_reward(
    obs: Tensor,
    actions: Tensor,
    contacts: dict,
    *,
    sigma: float = 0.25,
) -> Tensor:
    """
    Exponential reward for tracking commanded linear velocity.

    Uses ``contacts["commands"]`` and ``contacts["base_velocity"]`` when
    available; falls back to observation indices 0–1 (command) and 3–4
    (actual velocity) as a proxy.
    """
    commands = contacts.get("commands")
    base_vel = contacts.get("base_velocity")

    if commands is not None and base_vel is not None:
        cmd_vel = commands[:, :2].float()
        act_vel = base_vel[:, :2].float()
    else:
        cmd_vel = obs[:, 0:2].float()
        act_vel = obs[:, 3:5].float()

    vel_error_sq = ((cmd_vel - act_vel) ** 2).sum(dim=-1)
    return torch.exp(-vel_error_sq / (sigma**2))


def foot_slip_penalty(
    obs: Tensor,
    actions: Tensor,
    contacts: dict,
    *,
    force_threshold: float = 1.0,
) -> Tensor:
    """
    Penalise horizontal foot velocity whenever contact force > threshold.

    Returns a positive quantity; weight is negative so the net contribution penalises slip.
    """
    cf = contacts.get("contact_forces")  # [N, feet, 3]
    fv = contacts.get("foot_velocities")  # [N, feet, 3]
    if cf is None or fv is None:
        return torch.zeros(obs.shape[0], device=obs.device)

    force_mag = cf.float().norm(dim=-1)  # [N, feet]
    in_contact = (force_mag > force_threshold).float()

    foot_vel_sq = (fv[..., :2].float() ** 2).sum(dim=-1)  # [N, feet]
    return (in_contact * foot_vel_sq).sum(dim=-1)  # [N]  positive; weight=-0.5 → penalty


def terrain_clearance_reward(
    obs: Tensor,
    actions: Tensor,
    contacts: dict,
    *,
    min_clearance: float = 0.05,
    force_threshold: float = 1.0,
) -> Tensor:
    """
    Reward swing-phase foot height above ``min_clearance``.

    A foot is in swing when its contact force magnitude < ``force_threshold``.
    """
    cf = contacts.get("contact_forces")  # [N, feet, 3]
    fp = contacts.get("foot_positions")  # [N, feet, 3]
    if cf is None or fp is None:
        return torch.zeros(obs.shape[0], device=obs.device)

    force_mag = cf.float().norm(dim=-1)  # [N, feet]
    in_swing = (force_mag < force_threshold).float()

    foot_height = fp[..., 2].float()  # [N, feet]  world-frame z
    clearance = torch.clamp(foot_height - min_clearance, min=0.0)
    return (in_swing * clearance).sum(dim=-1)  # [N]


def contact_timing_penalty(
    obs: Tensor,
    actions: Tensor,
    contacts: dict,
    *,
    force_threshold: float = 1.0,
) -> Tensor:
    """
    Penalise asymmetric contact timing between diagonal leg pairs.

    AnymalC foot order assumed: FL=0, FR=1, RL=2, RR=3.

    Diagonal pairs:
      Pair A: FL (0) ↔ RR (3)
      Pair B: FR (1) ↔ RL (2)

    penalty = |contact[FL] - contact[RR]| + |contact[FR] - contact[RL]|

    This encourages a trot gait where diagonal legs contact in phase.
    Returns a positive quantity; weight is negative so the net contribution penalises asymmetry.
    """
    cf = contacts.get("contact_forces")
    if cf is None or cf.shape[1] < 4:
        return torch.zeros(obs.shape[0], device=obs.device)

    force_mag = cf.float().norm(dim=-1)  # [N, 4]
    c = (force_mag > force_threshold).float()  # [N, 4]

    pair_a = (c[:, 0] - c[:, 3]).abs()  # FL vs RR
    pair_b = (c[:, 1] - c[:, 2]).abs()  # FR vs RL
    return pair_a + pair_b  # [N]  positive; weight=-0.2 → penalty


def energy_penalty(
    obs: Tensor,
    actions: Tensor,
    contacts: dict,
) -> Tensor:
    """
    Penalise joint torque squared (proxy for mechanical energy expenditure).

    Falls back to action-magnitude penalty when torques are unavailable.
    """
    jt = contacts.get("joint_torques")
    if jt is not None:
        return (jt.float() ** 2).sum(dim=-1)  # positive; weight=-1e-4 makes this a penalty
    # Fallback: penalise large actions as a proxy
    return (actions.float() ** 2).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_TERM_REGISTRY: dict[str, tuple[Callable, float]] = {
    "velocity_tracking": (velocity_tracking_reward, 1.0),
    "foot_slip_penalty": (foot_slip_penalty, -0.5),
    "terrain_clearance": (terrain_clearance_reward, 0.3),
    "contact_timing": (contact_timing_penalty, -0.2),
    "energy_penalty": (energy_penalty, -1e-6),
}


def build_contact_aware_manager(
    enabled_terms: Set[str] | None = None,
    weight_overrides: dict[str, float] | None = None,
) -> RewardManager:
    """
    Build a RewardManager with all five contact-aware terms.

    Parameters
    ----------
    enabled_terms
        If given, only terms in this set are active (others get weight=0).
        Pass ``None`` to enable all terms.
    weight_overrides
        Optional dict to override per-term weights, e.g.
        ``{"energy_penalty": -5e-5}``.
    """
    mgr = RewardManager()
    overrides = weight_overrides or {}

    for name, (fn, default_weight) in _TERM_REGISTRY.items():
        weight = overrides.get(name, default_weight)
        if enabled_terms is not None and name not in enabled_terms:
            weight = 0.0
        mgr.add_term(name, fn, weight)

    return mgr
