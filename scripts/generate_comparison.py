"""
Generate a 2x2 subplot comparison animation from trajectory NPZ files.

Reads results/trajectory_{terrain}.npz for each terrain and creates
a side-by-side comparison animation.

Usage
-----
    python3.10 scripts/generate_comparison.py

Outputs
-------
    results/curriculum_demo.mp4
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"

TERRAINS = ["flat", "slopes", "stairs", "contact_aware"]
TERRAIN_TITLES = {
    "flat": "Phase 1: Flat Terrain (reward=14.53)",
    "slopes": "Phase 2a: Slopes 0-15° (reward=17.68)",
    "stairs": "Phase 2b: Stairs 5-20cm (reward=9.87)",
    "contact_aware": "Phase 3: Contact-Aware (stairs)",
}


def load_trajectory(terrain: str) -> dict | None:
    path = RESULTS_DIR / f"trajectory_{terrain}.npz"
    if not path.exists():
        print(f"  Missing: {path}", flush=True)
        return None
    data = dict(np.load(str(path)))
    data["terrain"] = terrain
    return data


def make_comparison_animation(trajectories: list[dict], out_path: str) -> None:
    n = len(trajectories)
    if n == 0:
        print("No trajectory data found.", flush=True)
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "Isaac Lab Quadruped Locomotion — Terrain Curriculum + Contact-Aware Reward\n"
        "Pre-arrival research project for Penn GRASP Lab (Fall 2026)  •  Rahil Parikh",
        fontsize=11,
        fontweight="bold",
    )

    positions_list = []
    axes = []
    lines = []
    points = []

    rows, cols = 2, 2
    for idx, traj in enumerate(trajectories):
        terrain = traj["terrain"]
        pos = traj.get("base_pos")
        positions_list.append(pos)

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        axes.append(ax)

        if pos is not None and len(pos) > 0:
            margin = 1.0
            ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
            ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)
            ax.set_zlim(0, max(pos[:, 2].max() + 0.5, 0.5))

        title = TERRAIN_TITLES.get(terrain, terrain)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)

        if pos is not None:
            (line,) = ax.plot([], [], [], "b-", linewidth=1.2, alpha=0.7)
            (pt,) = ax.plot([], [], [], "ro", markersize=5)
        else:
            (line,) = ax.plot([], [], [], "gray")
            (pt,) = ax.plot([], [], [], "gray")
            ax.text(
                0.5,
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
            )

        lines.append(line)
        points.append(pt)

    # Find max length
    max_frames = max(
        (len(pos) for pos in positions_list if pos is not None),
        default=1,
    )
    trail_len = 50
    fps = 20
    skip = max(1, max_frames // (fps * 10))  # target ~10s video
    frame_indices = list(range(0, max_frames, skip))

    def update(frame_i):
        for idx, (pos, line, pt) in enumerate(zip(positions_list, lines, points)):
            if pos is None or frame_i >= len(pos):
                continue
            start = max(0, frame_i - trail_len)
            line.set_data(pos[start:frame_i, 0], pos[start:frame_i, 1])
            line.set_3d_properties(pos[start:frame_i, 2])
            pt.set_data([pos[frame_i, 0]], [pos[frame_i, 1]])
            pt.set_3d_properties([pos[frame_i, 2]])
        return lines + points

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=1000 // fps, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    try:
        anim.save(out_path, writer=writer)
        print(f"Comparison video saved → {out_path}", flush=True)
    except Exception as e:
        print(f"ffmpeg failed ({e}). Saving static comparison image instead.", flush=True)
        png_path = out_path.replace(".mp4", ".png")
        # Update to last frame
        update(frame_indices[-1] if frame_indices else 0)
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Static image saved → {png_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    print("Loading trajectory files...", flush=True)
    trajectories = []
    for terrain in TERRAINS:
        traj = load_trajectory(terrain)
        if traj is not None:
            trajectories.append(traj)
            print(
                f"  Loaded: {terrain} ({len(traj.get('base_pos', []))} frames)",
                flush=True,
            )

    if not trajectories:
        print("No trajectories found. Run record_policy_video.py first.")
        sys.exit(1)

    out_path = str(RESULTS_DIR / "curriculum_demo.mp4")
    print(f"\nRendering comparison animation → {out_path}", flush=True)
    make_comparison_animation(trajectories, out_path)
