"""
Trajectory comparison visualization.

Loads all trajectory_{terrain}.npz files and produces:
  - results/trajectory_comparison.png  (2x2 panel figure, 300 DPI)
  - results/trajectory_{terrain}.png   (per-terrain individual plots)

Usage
-----
    python /workspace/isaac-lab-locomotion/scripts/visualize_trajectory.py

No simulator required — operates on pre-recorded .npz files.
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAJ_DIR = REPO_ROOT / "results" / "trajectories"
OUT_DIR = REPO_ROOT / "results"

TERRAIN_ORDER = ["flat", "slopes", "stairs", "contact_aware"]
TERRAIN_LABELS = {
    "flat": "Flat",
    "slopes": "Slopes (0–15°)",
    "stairs": "Stairs (5–20 cm)",
    "contact_aware": "Contact-Aware",
}
TERRAIN_COLORS = {
    "flat": "#2196F3",
    "slopes": "#4CAF50",
    "stairs": "#FF9800",
    "contact_aware": "#9C27B0",
}

FOOT_NAMES = ["FL", "FR", "RL", "RR"]
FOOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def load_trajectories() -> dict:
    trajs = {}
    for terrain in TERRAIN_ORDER:
        path = TRAJ_DIR / f"trajectory_{terrain}.npz"
        if path.exists():
            trajs[terrain] = dict(np.load(path))
            print(f"Loaded {path.name}")
        else:
            print(f"MISSING: {path} — skipping {terrain}")
    return trajs


def make_comparison_figure(trajs: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Quadruped Locomotion — Terrain Curriculum Progression",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # ── Panel 1: Base XY trajectory (top-down) ───────────────────────────────
    ax1 = axes[0, 0]
    for terrain, data in trajs.items():
        pos = data["base_position"]
        ax1.plot(pos[:, 0], pos[:, 1], color=TERRAIN_COLORS[terrain], label=TERRAIN_LABELS[terrain], linewidth=1.5)
        ax1.plot(pos[0, 0], pos[0, 1], "o", color=TERRAIN_COLORS[terrain], markersize=5)
    ax1.set_xlabel("X position (m)", fontsize=11)
    ax1.set_ylabel("Y position (m)", fontsize=11)
    ax1.set_title("Base XY Trajectory (top-down)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Base Z height over time ─────────────────────────────────────
    ax2 = axes[0, 1]
    for terrain, data in trajs.items():
        pos = data["base_position"]
        steps = np.arange(len(pos))
        ax2.plot(steps, pos[:, 2], color=TERRAIN_COLORS[terrain], label=TERRAIN_LABELS[terrain], linewidth=1.5)
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Base height (m)", fontsize=11)
    ax2.set_title("Base Height over Time", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Joint velocity magnitude (energy usage proxy) ───────────────
    ax3 = axes[1, 0]
    for terrain, data in trajs.items():
        jv_proxy = np.linalg.norm(data.get("base_linear_velocity", np.zeros((1, 3))), axis=-1)
        steps = np.arange(len(jv_proxy))
        ax3.plot(
            steps, jv_proxy, color=TERRAIN_COLORS[terrain], label=TERRAIN_LABELS[terrain], linewidth=1.0, alpha=0.85
        )
    ax3.set_xlabel("Step", fontsize=11)
    ax3.set_ylabel("Base speed (m/s)", fontsize=11)
    ax3.set_title("Base Speed over Time (energy proxy)", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Foot contact gait diagram ───────────────────────────────────
    ax4 = axes[1, 1]
    terrain_with_contact = None
    for terrain in TERRAIN_ORDER:
        if terrain in trajs:
            cf = trajs[terrain]["foot_contact_forces"]  # [steps, 4, 3]
            if np.any(cf != 0):
                terrain_with_contact = terrain
                break
    if terrain_with_contact is None and trajs:
        terrain_with_contact = next(iter(trajs))

    if terrain_with_contact is not None:
        cf = trajs[terrain_with_contact]["foot_contact_forces"]
        force_mag = np.linalg.norm(cf, axis=-1)  # [steps, 4]
        in_contact = (force_mag > 1.0).astype(float)  # [steps, 4]
        steps = np.arange(len(in_contact))
        display_steps = min(200, len(steps))
        for foot_idx in range(4):
            c = in_contact[:display_steps, foot_idx]
            ax4.fill_between(
                steps[:display_steps],
                foot_idx + c * 0.8,
                foot_idx,
                color=FOOT_COLORS[foot_idx],
                alpha=0.7,
                label=FOOT_NAMES[foot_idx],
            )
        ax4.set_yticks([0.4, 1.4, 2.4, 3.4])
        ax4.set_yticklabels(FOOT_NAMES, fontsize=10)
        ax4.set_xlabel("Step", fontsize=11)
        ax4.set_title(f"Foot Contact Pattern — {TERRAIN_LABELS[terrain_with_contact]}", fontsize=12)
        ax4.legend(fontsize=9, loc="upper right")
        ax4.grid(True, axis="x", alpha=0.3)
    else:
        ax4.text(
            0.5, 0.5, "No trajectory data\navailable", ha="center", va="center", transform=ax4.transAxes, fontsize=12
        )
        ax4.set_title("Foot Contact Pattern", fontsize=12)

    plt.tight_layout()
    return fig


def make_individual_figure(terrain: str, data: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    color = TERRAIN_COLORS[terrain]
    label = TERRAIN_LABELS[terrain]
    fig.suptitle(f"Trajectory — {label}", fontsize=13, fontweight="bold")

    pos = data["base_position"]
    steps = np.arange(len(pos))

    ax = axes[0, 0]
    ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.5)
    ax.plot(pos[0, 0], pos[0, 1], "go", markersize=6, label="Start")
    ax.plot(pos[-1, 0], pos[-1, 1], "rs", markersize=6, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Path")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, pos[:, 2], color=color, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Z (m)")
    ax.set_title("Base Height")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    jp = data.get("joint_positions", np.zeros((len(steps), 12)))
    jv_mag = np.sqrt((np.diff(jp, axis=0) ** 2).sum(axis=-1))
    ax.plot(np.arange(len(jv_mag)), jv_mag, color=color, linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("|Δjoint_pos|")
    ax.set_title("Joint Velocity Magnitude")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    cf = data["foot_contact_forces"]
    force_mag = np.linalg.norm(cf, axis=-1)
    in_contact = (force_mag > 1.0).astype(float)
    display_steps = min(200, len(steps))
    for foot_idx in range(4):
        c = in_contact[:display_steps, foot_idx]
        ax.fill_between(
            np.arange(display_steps),
            foot_idx + c * 0.8,
            foot_idx,
            color=FOOT_COLORS[foot_idx],
            alpha=0.7,
            label=FOOT_NAMES[foot_idx],
        )
    ax.set_yticks([0.4, 1.4, 2.4, 3.4])
    ax.set_yticklabels(FOOT_NAMES, fontsize=10)
    ax.set_xlabel("Step")
    ax.set_title("Foot Contact Pattern")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    trajs = load_trajectories()
    if not trajs:
        print("No trajectory files found in", TRAJ_DIR)
        print("Run record_trajectory.py for each terrain first.")
        return

    # Comparison figure
    fig = make_comparison_figure(trajs)
    out_path = OUT_DIR / "trajectory_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Individual figures
    for terrain, data in trajs.items():
        fig = make_individual_figure(terrain, data)
        out_path = OUT_DIR / f"trajectory_{terrain}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
