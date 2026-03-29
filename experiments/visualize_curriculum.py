"""
Curriculum training curve visualizer.

Loads results/curriculum_log.json + matching wandb runs and produces a
3-panel matplotlib figure — one panel per terrain stage — with vertical
dashed lines marking curriculum advancement points.

Output: results/curriculum_training_curves.png

Usage
-----
    python experiments/visualize_curriculum.py
    python experiments/visualize_curriculum.py --project isaac-lab-locomotion
    python experiments/visualize_curriculum.py --log results/curriculum_log.json --out results/my_plot.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize curriculum training curves.")
    p.add_argument(
        "--log",
        type=str,
        default="results/curriculum_log.json",
        help="Path to curriculum_log.json",
    )
    p.add_argument(
        "--project",
        type=str,
        default="isaac-lab-locomotion",
        help="wandb project name",
    )
    p.add_argument(
        "--entity",
        type=str,
        default=None,
        help="wandb entity (team or username); omit to use default",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="episode_reward",
        help="wandb metric to plot on y-axis",
    )
    p.add_argument(
        "--out",
        type=str,
        default="results/curriculum_training_curves.png",
        help="Output PNG path",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# wandb helpers
# ─────────────────────────────────────────────────────────────────────────────


def fetch_terrain_runs(project: str, entity: str | None, terrain_type: str) -> list:
    """Return all wandb runs tagged with *terrain_type* from the given project."""
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed — cannot fetch run history.", file=sys.stderr)
        return []

    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(
            project_path,
            filters={"config.terrain_type": terrain_type},
        )
        return list(runs)
    except Exception as exc:
        print(
            f"WARNING: wandb query failed for terrain '{terrain_type}': {exc}",
            file=sys.stderr,
        )
        return []


def fetch_history(run, metric: str) -> tuple[list[int], list[float]]:
    """Return (steps, values) for *metric* from a wandb run."""
    try:
        history = run.history(keys=[metric], pandas=False)
        steps = [row.get("_step", i) for i, row in enumerate(history)]
        values = [row.get(metric) for row in history]
        # Drop None entries
        pairs = [(s, v) for s, v in zip(steps, values) if v is not None]
        if not pairs:
            return [], []
        steps, values = zip(*pairs)
        return list(steps), list(values)
    except Exception as exc:
        print(f"WARNING: could not fetch history for run {run.id}: {exc}", file=sys.stderr)
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette per terrain stage
_STAGE_COLOURS = {
    "flat": "#2196F3",  # blue
    "slopes": "#4CAF50",  # green
    "stairs": "#9C27B0",  # purple
}

_STAGE_LABELS = {
    "flat": "Stage 1 — Flat terrain",
    "slopes": "Stage 2 — Slopes (0–15°)",
    "stairs": "Stage 3 — Stairs (5–20 cm)",
}

_STAGE_ORDER = ["flat", "slopes", "stairs"]


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Simple moving-average smoother; returns same length as input."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return list(np.convolve(padded, kernel, mode="valid"))


def plot_curriculum(
    log_entries: list[dict],
    runs_by_stage: dict[str, list],
    metric: str,
    out_path: str,
) -> None:
    """
    Draw a 3-panel figure: one panel per stage.

    For each panel:
    - Training curve(s) from wandb (smoothed + raw shaded band if multiple runs)
    - Vertical dashed line at the advancement iteration (from curriculum_log.json)
    """
    n_stages = len(_STAGE_ORDER)
    fig, axes = plt.subplots(
        1,
        n_stages,
        figsize=(6 * n_stages, 5),
        sharey=False,
    )
    if n_stages == 1:
        axes = [axes]

    # Build a map from stage name → log entry for quick look-up
    log_map: dict[str, dict] = {e["stage"]: e for e in log_entries}

    for ax, stage in zip(axes, _STAGE_ORDER):
        colour = _STAGE_COLOURS[stage]
        label = _STAGE_LABELS[stage]
        runs = runs_by_stage.get(stage, [])

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

        if not runs:
            ax.text(
                0.5,
                0.5,
                "No wandb data\nfound for this stage",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )
        else:
            all_steps: list[list[int]] = []
            all_values: list[list[float]] = []

            for run in runs:
                steps, values = fetch_history(run, metric)
                if steps:
                    all_steps.append(steps)
                    all_values.append(smooth(values))

            if all_values:
                # Plot each run as a faint line, plus a bold mean
                for steps, values in zip(all_steps, all_values):
                    ax.plot(steps, values, color=colour, alpha=0.3, linewidth=0.8)

                if len(all_values) > 1:
                    # Interpolate to a common x grid for the mean
                    max_step = max(max(s) for s in all_steps)
                    grid = np.linspace(0, max_step, 500)
                    interp_vals = [np.interp(grid, s, v) for s, v in zip(all_steps, all_values)]
                    mean_vals = np.mean(interp_vals, axis=0)
                    ax.plot(grid, mean_vals, color=colour, linewidth=2.0, label="mean")
                else:
                    ax.plot(
                        all_steps[0],
                        all_values[0],
                        color=colour,
                        linewidth=2.0,
                        label="reward",
                    )

        # ── Advancement marker ─────────────────────────────────────────────────
        if stage in log_map:
            entry = log_map[stage]
            adv_iter = entry.get("end_iter")
            trigger = entry.get("advancement_trigger", "")
            if adv_iter is not None:
                ax.axvline(
                    x=adv_iter,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"advance ({trigger})",
                    zorder=5,
                )
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Curriculum Training Progress — AnymalC", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Load curriculum log
    log_path = pathlib.Path(args.log)
    if not log_path.exists():
        print(f"ERROR: curriculum log not found at '{log_path}'.", file=sys.stderr)
        print("Run  python -m experiments.curriculum.trainer  first.", file=sys.stderr)
        sys.exit(1)

    with open(log_path) as f:
        log_entries: list[dict] = json.load(f)

    print(f"Loaded {len(log_entries)} stage entries from {log_path}")

    # Fetch wandb runs per stage
    runs_by_stage: dict[str, list] = {}
    for stage in _STAGE_ORDER:
        print(f"Fetching wandb runs for stage '{stage}' …")
        runs = fetch_terrain_runs(args.project, args.entity, terrain_type=stage)
        # Keep only runs tagged "curriculum" if possible; fall back to all
        curriculum_runs = [r for r in runs if "curriculum" in (r.tags or [])]
        runs_by_stage[stage] = curriculum_runs if curriculum_runs else runs
        print(f"  found {len(runs_by_stage[stage])} run(s)")

    plot_curriculum(log_entries, runs_by_stage, args.metric, args.out)


if __name__ == "__main__":
    main()
