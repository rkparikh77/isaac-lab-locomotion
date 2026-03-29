"""
Aggregate all wandb runs in the project and produce:
  results/training_summary.csv   — one row per run
  results/training_curves.png    — reward curves coloured by terrain type

Usage:
    python experiments/log_results.py
    python experiments/log_results.py --project isaac-lab-locomotion --out_dir results
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless — no display required on a remote GPU server
import matplotlib.pyplot as plt
import pandas as pd
import wandb

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise wandb runs.")
    p.add_argument(
        "--project",
        default="isaac-lab-locomotion",
        help="wandb project name (default: isaac-lab-locomotion)",
    )
    p.add_argument("--entity", default=None, help="wandb entity / team (omit to use default)")
    p.add_argument(
        "--out_dir",
        default="results",
        help="Directory for output files (default: results/)",
    )
    p.add_argument(
        "--metric",
        default="episode_reward",
        help="History metric to plot (default: episode_reward)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────


def fetch_runs(project: str, entity: str | None) -> list[wandb.apis.public.Run]:
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path))
    if not runs:
        raise RuntimeError(
            f"No runs found in wandb project '{path}'. "
            "Check --project / --entity arguments and that you are logged in."
        )
    print(f"Found {len(runs)} run(s) in '{path}'.")
    return runs


def extract_summary_row(run: wandb.apis.public.Run) -> dict:
    """Pull the scalar summary values we care about from a single run."""
    cfg = run.config or {}
    summary = run.summary or {}

    # Convergence iteration: prefer the explicit early-stop summary key,
    # fall back to the last logged iteration.
    convergence_iter = summary.get(
        "early_stop_iteration",
        summary.get("iteration", None),
    )

    return {
        "run_name": run.name,
        "run_id": run.id,
        "terrain_type": cfg.get("terrain_type", "unknown"),
        "final_reward": summary.get("best_reward", summary.get("episode_reward", None)),
        "convergence_iteration": convergence_iter,
        # wall_time stored in hours by train.py
        "wall_time_hours": summary.get("wall_time_hours", None),
        "num_envs": cfg.get("num_envs", None),
        "state": run.state,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV summary
# ─────────────────────────────────────────────────────────────────────────────


def write_summary_csv(runs: list[wandb.apis.public.Run], out_path: pathlib.Path) -> pd.DataFrame:
    rows = [extract_summary_row(r) for r in runs]
    df = pd.DataFrame(
        rows,
        columns=[
            "run_name",
            "terrain_type",
            "final_reward",
            "convergence_iteration",
            "wall_time_hours",
            "num_envs",
            "state",
            "run_id",
        ],
    )
    df.sort_values("terrain_type", inplace=True)
    df.to_csv(out_path, index=False)
    print(f"Summary CSV → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Training curves plot
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for terrain types
_TERRAIN_COLOURS: dict[str, str] = {
    "flat": "#2196F3",
    "rough": "#FF5722",
    "slope": "#4CAF50",
    "stairs": "#9C27B0",
    "unknown": "#9E9E9E",
}


def fetch_history(run: wandb.apis.public.Run, metric: str) -> pd.DataFrame:
    """Return a DataFrame with columns [step, <metric>] for one run."""
    history = run.history(keys=[metric, "_step"], pandas=True)
    if history.empty or metric not in history.columns:
        return pd.DataFrame(columns=["step", metric])
    history = history.rename(columns={"_step": "step"})
    history = history[["step", metric]].dropna()
    return history


def plot_curves(
    runs: list[wandb.apis.public.Run],
    metric: str,
    out_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    terrain_handles: dict[str, plt.Line2D] = {}

    for run in runs:
        terrain = (run.config or {}).get("terrain_type", "unknown")
        colour = _TERRAIN_COLOURS.get(terrain, "#9E9E9E")
        history = fetch_history(run, metric)
        if history.empty:
            print(f"  [warn] No '{metric}' history for run '{run.name}' — skipping.")
            continue

        (line,) = ax.plot(
            history["step"],
            history[metric],
            color=colour,
            alpha=0.8,
            linewidth=1.2,
            label=terrain if terrain not in terrain_handles else "_nolegend_",
        )
        if terrain not in terrain_handles:
            terrain_handles[terrain] = line

    ax.axhline(
        y=8.0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="Early-stop threshold (8.0)",
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Isaac Lab — Locomotion Training Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = fetch_runs(args.project, args.entity)

    write_summary_csv(runs, out_dir / "training_summary.csv")
    plot_curves(runs, args.metric, out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
