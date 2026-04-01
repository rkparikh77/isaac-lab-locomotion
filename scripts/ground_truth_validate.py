#!/usr/bin/env python3
"""Ground truth validation using ONLY trajectory data."""
import numpy as np
import json
import sys
from pathlib import Path


def validate_trajectory(traj_path, terrain_name, verbose=True):
    d = np.load(traj_path)
    pos = d["base_position"]
    vel = d["base_linear_velocity"]
    total_steps = len(pos)

    heights = pos[:, 2]
    height_ok_mask = (heights >= 0.35) & (heights <= 0.85)
    stable_steps = int(height_ok_mask.sum())
    height_pass = stable_steps >= min(400, int(total_steps * 0.8))

    max_height = float(heights.max())
    min_height = float(heights.min())
    launched = max_height > 1.2
    crashed = min_height < 0.25

    xy_displacement = float(np.sqrt((pos[-1, 0] - pos[0, 0]) ** 2 + (pos[-1, 1] - pos[0, 1]) ** 2))
    max_xy = float(np.sqrt((pos[:, 0] - pos[0, 0]) ** 2 + (pos[:, 1] - pos[0, 1]) ** 2).max())
    xy_threshold = 2.0 if terrain_name == "flat" else 0.8
    xy_pass = max_xy > xy_threshold

    stable_indices = np.where(height_ok_mask)[0]
    if len(stable_indices) > 50:
        stable_vel = vel[stable_indices[50:]]
        speeds = np.linalg.norm(stable_vel[:, :2], axis=-1)
        mean_speed = float(speeds.mean())
    else:
        mean_speed = 0.0
    speed_pass = mean_speed > 0.1

    overall_pass = height_pass and xy_pass and speed_pass

    result = {
        "terrain": terrain_name,
        "total_steps": total_steps,
        "stable_steps": stable_steps,
        "height_pass": height_pass,
        "height_range": [min_height, max_height],
        "launched": launched,
        "crashed": crashed,
        "xy_displacement": xy_displacement,
        "max_xy": max_xy,
        "xy_pass": xy_pass,
        "mean_speed": mean_speed,
        "speed_pass": speed_pass,
        "OVERALL_PASS": overall_pass,
    }

    if verbose:
        status = "PASS" if overall_pass else "FAIL"
        print(f"\n{terrain_name.upper()}: {status}")
        print(f"  Height: {stable_steps}/{total_steps} stable, range=[{min_height:.2f}, {max_height:.2f}]m {'PASS' if height_pass else 'FAIL'}")
        if launched:
            print(f"         WARNING: LAUNCHED to {max_height:.2f}m")
        if crashed:
            print(f"         WARNING: CRASHED to {min_height:.2f}m")
        print(f"  XY:     {xy_displacement:.2f}m final, {max_xy:.2f}m max {'PASS' if xy_pass else 'FAIL'}")
        print(f"  Speed:  {mean_speed:.3f} m/s {'PASS' if speed_pass else 'FAIL'}")

    return result


def validate_all():
    results_dir = Path("/workspace/isaac-lab-locomotion/results/trajectories")
    terrains = ["flat", "slopes", "stairs", "contact_aware"]

    print("=" * 70)
    print("GROUND TRUTH VALIDATION")
    print("=" * 70)

    all_results = {}
    all_pass = True

    for terrain in terrains:
        traj_path = results_dir / f"trajectory_{terrain}.npz"
        if not traj_path.exists():
            print(f"\n{terrain.upper()}: NO TRAJECTORY FILE")
            all_pass = False
            continue
        result = validate_trajectory(str(traj_path), terrain)
        all_results[terrain] = result
        if not result["OVERALL_PASS"]:
            all_pass = False

    print("\n" + "=" * 70)
    print("ALL PASS -- PROCEED TO ABLATION" if all_pass else "SOME FAIL -- FIX BEFORE ABLATION")
    print("=" * 70)

    with open("/workspace/results/ground_truth_validation.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_pass, all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_trajectory(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown")
    else:
        validate_all()
