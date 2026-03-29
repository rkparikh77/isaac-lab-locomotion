"""
Publication-quality ablation bar chart.

Reads results/ablation_results.csv and produces:
  - results/ablation_figure.png  (300 DPI, publication-ready)

Usage
-----
    python /workspace/isaac-lab-locomotion/scripts/generate_ablation_figure.py

No simulator required.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "results" / "ablation_results.csv"
OUT_PATH = REPO_ROOT / "results" / "ablation_figure.png"

# Clean display labels (no underscores)
LABEL_MAP = {
    "all_terms": "All Terms\n(baseline)",
    "no_contact_timing": "No Contact\nTiming",
    "no_terrain_clearance": "No Terrain\nClearance",
    "no_energy_penalty": "No Energy\nPenalty",
    "no_foot_slip": "No Foot\nSlip",
}

# Display order
CONDITION_ORDER = [
    "all_terms",
    "no_contact_timing",
    "no_terrain_clearance",
    "no_energy_penalty",
    "no_foot_slip",
]

BASELINE_COLOR = "#2196F3"
ABLATED_COLOR = "#90CAF9"
CRITICAL_COLOR = "#F44336"


def load_csv(path: pathlib.Path) -> dict:
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["condition"]
            rows[name] = {
                "mean": float(row["mean_reward_last50"]),
                "std": float(row["std_reward_last50"]),
                "n": int(row["num_data_points"]),
            }
    return rows


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    data = load_csv(CSV_PATH)

    conditions = [c for c in CONDITION_ORDER if c in data]
    means = [data[c]["mean"] for c in conditions]
    stds = [data[c]["std"] for c in conditions]
    labels = [LABEL_MAP.get(c, c) for c in conditions]

    # Color bars: baseline = blue, critical (no_energy_penalty) = red, others = light blue
    colors = []
    for c in conditions:
        if c == "all_terms":
            colors.append(BASELINE_COLOR)
        elif c == "no_energy_penalty":
            colors.append(CRITICAL_COLOR)
        else:
            colors.append(ABLATED_COLOR)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(conditions))
    bars = ax.bar(
        x,
        means,
        color=colors,
        width=0.55,
        yerr=stds,
        capsize=5,
        error_kw={"linewidth": 1.5, "ecolor": "#555"},
        zorder=3,
    )

    # Horizontal dashed reference line at all_terms mean
    if "all_terms" in data:
        baseline_mean = data["all_terms"]["mean"]
        ax.axhline(
            baseline_mean,
            color=BASELINE_COLOR,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"All-terms baseline ({baseline_mean:.0f})",
            zorder=2,
        )

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 30,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Reward (last 50 iters)", fontsize=12)
    ax.set_title("Contact-Aware Reward Ablation — Stairs Terrain", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation for critical result
    if "no_energy_penalty" in conditions:
        idx = conditions.index("no_energy_penalty")
        ax.annotate(
            "Bug: double-negative\n(see EVALUATION_NOTES)",
            xy=(idx, data["no_energy_penalty"]["mean"] + data["no_energy_penalty"]["std"] + 50),
            xytext=(idx + 0.7, data["no_energy_penalty"]["mean"] + 400),
            fontsize=8,
            color=CRITICAL_COLOR,
            arrowprops={"arrowstyle": "->", "color": CRITICAL_COLOR, "lw": 1.2},
        )

    # Legend patches for color coding
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, label="Baseline (all terms)"),
        Patch(facecolor=ABLATED_COLOR, label="Ablated condition"),
        Patch(facecolor=CRITICAL_COLOR, label="No energy penalty (dominant term)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
