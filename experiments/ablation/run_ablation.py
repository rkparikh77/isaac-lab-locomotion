"""
Contact-aware reward ablation study — leave-one-out design.

Runs 5 training conditions by launching each as a separate subprocess.
Each condition removes exactly one contact-aware reward term from the full model.
This avoids the Isaac Lab / PhysX GPU re-init crash that occurs when creating
multiple simulation instances in the same process.

Conditions (leave-one-out):
  1. all_terms          : all 5 contact-aware terms (full model)
  2. no_contact_timing  : remove foot contact timing penalty
  3. no_terrain_clearance: remove terrain clearance / foot lift reward
  4. no_energy_penalty  : remove energy/GRF proxy penalty
  5. no_foot_slip       : remove foot slip penalty

Each condition runs for --max_iterations (default 300) starting from the
stairs best_model.pt checkpoint.

Outputs
-------
  /workspace/results/ablation_results.csv
  /workspace/results/ablation_bar_chart.png

Usage
-----
    cd /workspace/IsaacLab
    python3.10 /workspace/isaac-lab-locomotion/experiments/ablation/run_ablation.py \\
        --headless [--num_envs 1024] [--max_iterations 300]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time

# ── Parse args before any Isaac Lab imports ───────────────────────────────────
# (AppLauncher must be the first thing that runs for the *sub-process* train.py;
#  this orchestrator script does NOT launch Isaac Lab itself.)
parser = argparse.ArgumentParser(description="Contact-aware reward ablation study.")
parser.add_argument(
    "--headless",
    action="store_true",
    default=True,
    help="Pass --headless to each training subprocess",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1024,
    help="Environments per condition (default: 1024)",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=300,
    help="Iterations per ablation condition (default: 300)",
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default="/workspace/checkpoints/stairs/best_model.pt",
)
parser.add_argument(
    "--train_script",
    type=str,
    default=str(pathlib.Path(__file__).resolve().parent.parent / "04_contact_aware" / "train.py"),
    help="Path to the train.py script to invoke per condition",
)
parser.add_argument(
    "--start_from",
    type=int,
    default=0,
    help="Skip the first N conditions (0 = run all). Useful for resuming after a crash.",
)
args_cli = parser.parse_args()

# ── All five contact-aware terms ──────────────────────────────────────────────
ALL_TERMS = [
    "velocity_tracking",
    "foot_slip_penalty",
    "terrain_clearance",
    "contact_timing",
    "energy_penalty",
]

# Leave-one-out ablation conditions:
#   label  →  set of ENABLED terms  (None = all)
ABLATION_CONDITIONS: list[tuple[str, list[str] | None]] = [
    ("all_terms", None),  # full model — baseline
    ("no_contact_timing", [t for t in ALL_TERMS if t != "contact_timing"]),
    ("no_terrain_clearance", [t for t in ALL_TERMS if t != "terrain_clearance"]),
    ("no_energy_penalty", [t for t in ALL_TERMS if t != "energy_penalty"]),
    ("no_foot_slip", [t for t in ALL_TERMS if t != "foot_slip_penalty"]),
]

LAST_N_ITERS = 50  # window for mean/std in CSV
WANDB_LOG_PATH = "/workspace/results/ablation_wandb_logs"


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner
# ─────────────────────────────────────────────────────────────────────────────


def run_condition_subprocess(
    condition_label: str,
    enabled_terms: list[str] | None,
    num_envs: int,
    max_iterations: int,
    load_checkpoint: str,
    train_script: str,
) -> tuple[int, str]:
    """
    Launch one ablation condition as a subprocess.

    Returns
    -------
    returncode : int  (0 = success)
    log_path   : str  path to the condition's stdout/stderr log
    """
    ckpt_dir = f"/workspace/checkpoints/ablation/{condition_label}"
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(WANDB_LOG_PATH).mkdir(parents=True, exist_ok=True)

    log_path = f"/workspace/results/ablation_{condition_label}.log"

    cmd = [
        sys.executable,
        train_script,
        "--headless",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
        "--load_checkpoint",
        load_checkpoint,
        "--checkpoint_dir",
        ckpt_dir,
        "--run_name",
        f"ablation_{condition_label}",
    ]
    if enabled_terms is not None:
        cmd += ["--enabled_terms"] + enabled_terms

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"

    # Replicate the LD_LIBRARY_PATH set by /workspace/run_training.sh so that
    # libhdx.so and other Isaac Sim shared libraries are found by the subprocess.
    import subprocess as _sp
    import shlex

    _ld_script = (
        "EXT=$(find /home/claudeuser/.local/share/ov/data/exts/v2 -maxdepth 2 -name bin -type d 2>/dev/null | paste -sd ':' -);"
        "ISIM=$(find /home/claudeuser/.local/lib/python3.10/site-packages/omni/data -name bin -type d 2>/dev/null | paste -sd ':' -);"
        "EXTRA=/tmp/libsm6_extracted/usr/lib/x86_64-linux-gnu:/tmp/libxt6_extracted/usr/lib/x86_64-linux-gnu"
        ":/tmp/libxrender1_extracted/usr/lib/x86_64-linux-gnu:/tmp/libice6_extracted/usr/lib/x86_64-linux-gnu;"
        "echo ${EXTRA}:${EXT}:${ISIM}"
    )
    _ld_result = _sp.run(["bash", "-c", _ld_script], capture_output=True, text=True)
    _extra_ld = _ld_result.stdout.strip()
    if _extra_ld:
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{_extra_ld}:{existing_ld}" if existing_ld else _extra_ld

    print(f"\n  Launching: {' '.join(cmd[:6])} ...", flush=True)
    print(f"  Log: {log_path}", flush=True)

    with open(log_path, "w") as logfile:
        proc = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/workspace/IsaacLab",
        )
        proc.wait()

    return proc.returncode, log_path


# ─────────────────────────────────────────────────────────────────────────────
# Parse per-condition log to extract reward history
# ─────────────────────────────────────────────────────────────────────────────


def parse_rewards_from_log(log_path: str) -> list[float]:
    """
    Extract mean episode rewards from a train.py log file.

    Looks for lines of the form:
        [iter   N/300] reward=XXX.XXX  best=YYY.YYY ...
    """
    rewards: list[float] = []
    if not os.path.exists(log_path):
        return rewards
    with open(log_path) as f:
        for line in f:
            if "[iter " in line and "reward=" in line:
                try:
                    # Extract first reward= token
                    for token in line.split():
                        if token.startswith("reward="):
                            val = float(token.split("=")[1])
                            rewards.append(val)
                            break
                except ValueError:
                    pass
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Results export
# ─────────────────────────────────────────────────────────────────────────────


def export_csv(
    conditions: list[str],
    reward_histories: list[list[float]],
    last_n: int,
    csv_path: str,
) -> list[tuple[str, float, float]]:
    """Write ablation_results.csv, return [(label, mean, std), ...]."""
    import statistics

    rows: list[tuple[str, float, float]] = []
    for label, history in zip(conditions, reward_histories):
        tail = [r for r in history[-last_n:] if math.isfinite(r)]  # drop NaN and ±inf
        if not tail:
            mean_r, std_r = float("nan"), float("nan")
        else:
            mean_r = statistics.mean(tail)
            std_r = statistics.stdev(tail) if len(tail) > 1 else 0.0
        rows.append((label, mean_r, std_r))

    pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                f"mean_reward_last{last_n}",
                f"std_reward_last{last_n}",
                "num_data_points",
            ]
        )
        for label, mean_r, std_r in rows:
            idx = conditions.index(label)
            n = len([r for r in reward_histories[idx][-last_n:] if r == r])
            writer.writerow([label, f"{mean_r:.4f}", f"{std_r:.4f}", n])

    print(f"\nAblation CSV → {csv_path}", flush=True)
    return rows


def export_bar_chart(
    rows: list[tuple[str, float, float]],
    png_path: str,
    last_n: int,
) -> None:
    """Generate and save the ablation bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Human-readable labels
    label_map = {
        "all_terms": "Full Model\n(all 5 terms)",
        "no_contact_timing": "No Contact\nTiming",
        "no_terrain_clearance": "No Terrain\nClearance",
        "no_energy_penalty": "No Energy\n(GRF proxy)",
        "no_foot_slip": "No Slip\nPenalty",
    }

    labels = [label_map.get(r[0], r[0]) for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]

    x = np.arange(len(labels))
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colours[: len(labels)],
        alpha=0.85,
        width=0.55,
        capsize=6,
        error_kw={"elinewidth": 1.5, "ecolor": "black"},
    )

    for bar, mean, std in zip(bars, means, stds):
        if mean == mean:  # not NaN
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(mean + std + 5, mean * 1.02),
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel(
        f"Mean Episode Reward\n(last {last_n} iters, within-run std)",
        fontsize=11,
    )
    ax.set_title(
        "Contact-Aware Reward Ablation — AnymalC Stair Terrain\n"
        "Leave-one-out: each bar removes one reward term from the full model",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    pathlib.Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation bar chart → {png_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    num_envs = args_cli.num_envs
    max_iterations = args_cli.max_iterations
    load_ckpt = args_cli.load_checkpoint
    train_script = args_cli.train_script

    print(f"\n{'='*65}")
    print(f"  ABLATION STUDY  ({len(ABLATION_CONDITIONS)} conditions × {max_iterations} iters)")
    print("  Design: leave-one-out (each condition removes one term)")
    print(f"  num_envs={num_envs}  load_checkpoint={load_ckpt}")
    print(f"  train_script={train_script}")
    print(f"{'='*65}\n")

    all_labels: list[str] = []
    all_histories: list[list[float]] = []
    results: list[dict] = []

    start_from = args_cli.start_from

    # Pre-load completed condition logs when resuming from --start_from N
    for idx in range(start_from):
        label = ABLATION_CONDITIONS[idx][0]
        log_path = f"/workspace/results/ablation_{label}.log"
        history = parse_rewards_from_log(log_path)
        all_labels.append(label)
        all_histories.append(history)
        tail = [r for r in history[-LAST_N_ITERS:] if math.isfinite(r)]
        mean_r = sum(tail) / len(tail) if tail else float("nan")
        std_r = 0.0
        if len(tail) > 1:
            std_r = statistics.stdev(tail)
        best_r = max((r for r in history if math.isfinite(r)), default=float("nan"))
        results.append({"label": label, "mean": mean_r, "std": std_r, "best": best_r, "elapsed_min": 0.0, "retcode": 0})
        print(f"  [resume] Loaded {label} from {log_path}  best={best_r:.1f}  mean_last{LAST_N_ITERS}={mean_r:.1f}")

    for idx, (label, enabled_terms) in enumerate(ABLATION_CONDITIONS):
        if idx < start_from:
            continue
        removed = "none (full model)" if enabled_terms is None else (set(ALL_TERMS) - set(enabled_terms))
        print(f"\n{'─'*60}")
        print(f"  Condition {idx + 1}/{len(ABLATION_CONDITIONS)}: {label}")
        print(f"  Removed term: {removed}")
        print(f"  Active terms: {enabled_terms or ALL_TERMS}")
        print(f"{'─'*60}")

        t_start = time.time()
        retcode, log_path = run_condition_subprocess(
            condition_label=label,
            enabled_terms=enabled_terms,
            num_envs=num_envs,
            max_iterations=max_iterations,
            load_checkpoint=load_ckpt,
            train_script=train_script,
        )
        elapsed = time.time() - t_start

        history = parse_rewards_from_log(log_path)
        all_labels.append(label)
        all_histories.append(history)

        tail = [r for r in history[-LAST_N_ITERS:] if math.isfinite(r)]  # drop NaN and ±inf
        if tail:
            mean_r = statistics.mean(tail)
            std_r = statistics.stdev(tail) if len(tail) > 1 else 0.0
            best_r = max(history) if history else float("nan")
        else:
            mean_r, std_r, best_r = float("nan"), float("nan"), float("nan")

        results.append(
            {
                "label": label,
                "mean": mean_r,
                "std": std_r,
                "best": best_r,
                "elapsed_min": elapsed / 60.0,
                "retcode": retcode,
            }
        )

        print(
            f"\n  [{label}] DONE  returncode={retcode}  "
            f"elapsed={elapsed / 60:.1f}min  "
            f"best={best_r:.1f}  "
            f"mean_last{LAST_N_ITERS}={mean_r:.1f} ±{std_r:.1f}",
            flush=True,
        )

        if retcode != 0:
            print(
                f"  WARNING: condition '{label}' exited with code {retcode}. " f"Check {log_path}",
                flush=True,
            )

    # ── Export results ────────────────────────────────────────────────────────
    csv_path = "/workspace/results/ablation_results.csv"
    png_path = "/workspace/results/ablation_bar_chart.png"

    rows = export_csv(all_labels, all_histories, LAST_N_ITERS, csv_path)
    export_bar_chart(rows, png_path, LAST_N_ITERS)

    # ── Print final table ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  ABLATION RESULTS  (mean ± std of last {LAST_N_ITERS} iters, within-run std)")
    print("  Note: error bars reflect training variability, not across-seed std.")
    print("  Limitation: single seed per condition — treat as exploratory.")
    print(f"{'─' * 65}")
    print(f"  {'Condition':<25}  {'Mean Reward':>12}  {'± Std':>8}  {'Best':>9}")
    print(f"{'─'*65}")
    for r in results:
        print(f"  {r['label']:<25}  {r['mean']:>12.1f}  {r['std']:>8.1f}  {r['best']:>9.1f}")
    print(f"{'='*65}\n")

    print("All ablation conditions complete.")
    print(f"  CSV  → {csv_path}")
    print(f"  Plot → {png_path}")


if __name__ == "__main__":
    main()
