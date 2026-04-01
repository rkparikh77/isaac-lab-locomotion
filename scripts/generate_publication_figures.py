#!/usr/bin/env python3
"""Generate publication-quality figures."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_gait import extract_foot_contact, FOOT_NAMES

plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150


def generate_trajectory_comparison():
    """4-panel trajectory comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "AnymalC Quadruped Locomotion with Trotting Gait", fontsize=14, fontweight="bold"
    )

    terrains = ["flat", "slopes", "stairs", "contact_aware"]
    colors = {
        "flat": "#1f77b4",
        "slopes": "#2ca02c",
        "stairs": "#ff7f0e",
        "contact_aware": "#9467bd",
    }
    labels = {
        "flat": "Flat",
        "slopes": "Slopes (0-15 deg)",
        "stairs": "Stairs (5-20cm)",
        "contact_aware": "Contact-Aware",
    }
    traj_dir = Path("results/trajectories")

    # XY Trajectory
    ax = axes[0, 0]
    for terrain in terrains:
        traj_path = traj_dir / f"trajectory_{terrain}.npz"
        if traj_path.exists():
            d = np.load(traj_path)
            pos = d["base_position"]
            ax.plot(
                pos[:, 0] - pos[0, 0],
                pos[:, 1] - pos[0, 1],
                color=colors[terrain],
                label=labels[terrain],
                linewidth=1.5,
            )
    ax.set_xlabel("X displacement (m)")
    ax.set_ylabel("Y displacement (m)")
    ax.set_title("Base XY Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Height over time
    ax = axes[0, 1]
    for terrain in terrains:
        traj_path = traj_dir / f"trajectory_{terrain}.npz"
        if traj_path.exists():
            d = np.load(traj_path)
            ax.plot(
                d["base_position"][:, 2],
                color=colors[terrain],
                label=labels[terrain],
                linewidth=1,
                alpha=0.8,
            )
    ax.axhline(0.58, color="gray", linestyle="--", alpha=0.5, label="Nominal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Height (m)")
    ax.set_title("Base Height")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Speed over time
    ax = axes[1, 0]
    for terrain in terrains:
        traj_path = traj_dir / f"trajectory_{terrain}.npz"
        if traj_path.exists():
            d = np.load(traj_path)
            vel = d["base_linear_velocity"]
            speed = np.linalg.norm(vel[:, :2], axis=-1)
            # Smooth with rolling average
            kernel = np.ones(10) / 10
            speed_smooth = np.convolve(speed, kernel, mode="same")
            ax.plot(
                speed_smooth,
                color=colors[terrain],
                label=labels[terrain],
                linewidth=1,
                alpha=0.8,
            )
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Base Speed (10-step rolling avg)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Duty cycle comparison
    ax = axes[1, 1]
    gait_file = Path("results/gait_analysis.json")
    if gait_file.exists():
        with open(gait_file) as f:
            gait_data = json.load(f)

        x = np.arange(len(terrains))
        duty_cycles = [
            gait_data.get(t, {}).get("mean_duty_cycle", 1.0) for t in terrains
        ]
        bars = ax.bar(x, duty_cycles, color=[colors[t] for t in terrains], alpha=0.85)
        ax.axhline(0.7, color="red", linestyle="--", alpha=0.7, label="Trot threshold (0.7)")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[t] for t in terrains], rotation=15)
        ax.set_ylabel("Mean Duty Cycle")
        ax.set_title("Gait Quality (< 0.7 = trotting)")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, dc in zip(bars, duty_cycles):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                dc + 0.02,
                f"{dc:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig("results/trajectory_comparison_publication.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/trajectory_comparison_publication.png")


def generate_gait_comparison():
    """Side-by-side gait diagrams for all terrains."""
    terrains = ["flat", "slopes", "stairs", "contact_aware"]
    labels = {
        "flat": "Flat",
        "slopes": "Slopes",
        "stairs": "Stairs",
        "contact_aware": "Contact-Aware",
    }
    traj_dir = Path("results/trajectories")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Gait Patterns Across Terrains\n(Colored = stance, White = swing)",
        fontsize=14,
        fontweight="bold",
    )

    leg_names = FOOT_NAMES
    leg_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
    steps = 200

    gait_data = {}
    if Path("results/gait_analysis.json").exists():
        with open("results/gait_analysis.json") as f:
            gait_data = json.load(f)

    for idx, terrain in enumerate(terrains):
        ax = axes[idx]
        traj_path = traj_dir / f"trajectory_{terrain}.npz"

        if not traj_path.exists():
            ax.text(
                0.5, 0.5, f"No data for {terrain}", ha="center", va="center", transform=ax.transAxes
            )
            continue

        try:
            contact, d = extract_foot_contact(str(traj_path))
        except Exception:
            ax.text(
                0.5, 0.5, f"Error loading {terrain}", ha="center", va="center", transform=ax.transAxes
            )
            continue

        contacts = contact[:steps]

        for leg_idx, (leg_name, color) in enumerate(zip(leg_names, leg_colors)):
            y_pos = len(leg_names) - leg_idx - 1
            c = contacts[:, leg_idx]

            in_stance = False
            stance_start = 0
            for t in range(len(c)):
                if c[t] and not in_stance:
                    stance_start = t
                    in_stance = True
                elif not c[t] and in_stance:
                    rect = Rectangle(
                        (stance_start, y_pos + 0.1),
                        t - stance_start,
                        0.8,
                        facecolor=color,
                        edgecolor="black",
                        linewidth=0.3,
                    )
                    ax.add_patch(rect)
                    in_stance = False
            if in_stance:
                rect = Rectangle(
                    (stance_start, y_pos + 0.1),
                    len(c) - stance_start,
                    0.8,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3,
                )
                ax.add_patch(rect)

        ax.set_xlim(0, steps)
        ax.set_ylim(-0.5, 4.5)
        ax.set_yticks([i + 0.5 for i in range(4)])
        ax.set_yticklabels(leg_names[::-1])
        ax.set_ylabel(labels[terrain], fontsize=11, fontweight="bold")

        duty = gait_data.get(terrain, {}).get("mean_duty_cycle", -1)
        corr = gait_data.get(terrain, {}).get("diagonal_correlation", 0)
        if duty >= 0:
            gait_type = "Trot" if duty < 0.7 and corr < -0.3 else "Mixed"
            ax.text(
                steps + 5,
                2,
                f"{gait_type}\nduty={duty:.2f}\ncorr={corr:.2f}",
                fontsize=9,
                va="center",
            )

    axes[-1].set_xlabel("Simulation Step")
    plt.tight_layout()
    plt.savefig("results/gait_comparison_publication.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/gait_comparison_publication.png")


def generate_ablation_figure():
    """Publication-quality ablation bar chart with gait metrics."""
    traj_dir = Path("results/trajectories")
    conditions = [
        "all_terms",
        "no_contact_timing",
        "no_terrain_clearance",
        "no_energy_penalty",
        "no_foot_slip",
    ]
    labels = [
        "All Terms\n(baseline)",
        "No Contact\nTiming",
        "No Terrain\nClearance",
        "No Energy\nPenalty",
        "No Foot\nSlip",
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    results = []
    for cond in conditions:
        traj_path = traj_dir / f"trajectory_ablation_{cond}.npz"
        if traj_path.exists():
            try:
                contact, d = extract_foot_contact(str(traj_path))
                pos = d["base_position"]
                vel = d["base_linear_velocity"]
                xy = float(
                    np.sqrt(
                        (pos[-1, 0] - pos[0, 0]) ** 2 + (pos[-1, 1] - pos[0, 1]) ** 2
                    )
                )
                speed = float(np.linalg.norm(vel[30:, :2], axis=-1).mean())
                duty = float(contact.mean())
                diag1 = contact[:, 0].astype(float) + contact[:, 3].astype(float)
                diag2 = contact[:, 1].astype(float) + contact[:, 2].astype(float)
                corr = float(np.corrcoef(diag1, diag2)[0, 1])
                results.append(
                    {"xy": xy, "speed": speed, "duty": duty, "corr": corr, "exists": True}
                )
            except Exception:
                results.append({"xy": 0, "speed": 0, "duty": 1, "corr": 0, "exists": False})
        else:
            results.append({"xy": 0, "speed": 0, "duty": 1, "corr": 0, "exists": False})

    if not any(r["exists"] for r in results):
        print("No ablation data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Contact-Aware Reward Ablation - Physical + Gait Metrics",
        fontsize=14,
        fontweight="bold",
    )

    x = np.arange(len(conditions))

    # XY displacement
    ax = axes[0]
    xy_vals = [r["xy"] for r in results]
    bars = ax.bar(x, xy_vals, color=colors, alpha=0.85)
    if results[0]["exists"]:
        ax.axhline(results[0]["xy"], color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("XY Displacement (m)")
    ax.set_title("Distance Traveled")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, xy_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", fontsize=8)

    # Speed
    ax = axes[1]
    speed_vals = [r["speed"] for r in results]
    bars = ax.bar(x, speed_vals, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Speed (m/s)")
    ax.set_title("Locomotion Speed")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, speed_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # Duty cycle (gait quality)
    ax = axes[2]
    duty_vals = [r["duty"] for r in results]
    bars = ax.bar(x, duty_vals, color=colors, alpha=0.85)
    ax.axhline(0.7, color="red", linestyle="--", alpha=0.7, label="Trot threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean Duty Cycle")
    ax.set_title("Gait Quality (< 0.7 = trotting)")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, duty_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/ablation_publication.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/ablation_publication.png")


def generate_reward_ablation_chart():
    """Bar chart from ablation CSV results."""
    csv_path = Path("/workspace/results/ablation_results.csv")
    if not csv_path.exists():
        print("No ablation CSV found")
        return

    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labels_map = {
        "all_terms": "All Terms\n(baseline)",
        "no_contact_timing": "No Contact\nTiming",
        "no_terrain_clearance": "No Terrain\nClearance",
        "no_energy_penalty": "No Energy\nPenalty",
        "no_foot_slip": "No Foot\nSlip",
    }
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]

    conds = [r["condition"] for r in rows]
    means = [float(r["mean_reward_last50"]) for r in rows]
    stds = [float(r["std_reward_last50"]) for r in rows]
    labels = [labels_map.get(c, c) for c in conds]

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=6, error_kw={"elinewidth": 1.5})

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + s + 10,
            f"{m:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Episode Reward (last 50 iters)", fontsize=11)
    ax.set_title(
        "Contact-Aware Reward Ablation - Leave-One-Out\n"
        "AnymalC on Pyramid Stairs (500 iterations, 1024 envs)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("results/ablation_reward_publication.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/ablation_reward_publication.png")


if __name__ == "__main__":
    generate_trajectory_comparison()
    generate_gait_comparison()
    generate_ablation_figure()
    generate_reward_ablation_chart()
