#!/usr/bin/env python3
"""
Analyze gait quality from trajectory data.
Proper trotting gait should have:
- Duty cycle per foot: 50-70% (not 100% = shuffling)
- Diagonal pair correlation: negative (alternating)
- Clear swing phases visible in contact pattern
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import json

# AnymalC foot body indices in the 17-body contact sensor
FOOT_INDICES = [4, 8, 12, 16]  # LF, RF, LH, RH
FOOT_NAMES = ["LF", "RF", "LH", "RH"]
CONTACT_FORCE_THRESHOLD = 5.0  # N — foot is in contact if force > this


def extract_foot_contact(traj_path):
    """Extract binary foot contact from contact force data."""
    d = np.load(traj_path)
    cf = d["foot_contact_forces"]  # (N, 17, 3) or (N, 4, 3)

    if cf.shape[1] == 17:
        # Full body contact — extract foot bodies
        foot_forces = cf[:, FOOT_INDICES, :]  # (N, 4, 3)
    elif cf.shape[1] == 4:
        foot_forces = cf
    else:
        # Try to use all bodies, pick top 4 by mean magnitude
        mags = np.linalg.norm(cf, axis=-1).mean(axis=0)
        top4 = np.argsort(mags)[-4:]
        foot_forces = cf[:, sorted(top4), :]

    force_mag = np.linalg.norm(foot_forces, axis=-1)  # (N, 4)
    contact = force_mag > CONTACT_FORCE_THRESHOLD  # (N, 4) bool
    return contact, d


def analyze_gait(traj_path, terrain_name):
    """Analyze gait quality from trajectory data."""
    contact, d = extract_foot_contact(traj_path)
    pos = d["base_position"]
    vel = d["base_linear_velocity"]
    N = len(contact)

    # Duty cycle per foot (fraction of time in contact)
    duty_cycles = contact.mean(axis=0)

    # Diagonal pair analysis
    # AnymalC: LF=0, RF=1, LH=2, RH=3
    # Trot: LF+RH swing together, RF+LH swing together
    diag1 = contact[:, 0].astype(float) + contact[:, 3].astype(float)  # LF + RH
    diag2 = contact[:, 1].astype(float) + contact[:, 2].astype(float)  # RF + LH

    if len(diag1) > 10:
        correlation = float(np.corrcoef(diag1, diag2)[0, 1])
    else:
        correlation = 0.0

    # Swing phase detection
    swing_phases = []
    for leg in range(4):
        c = contact[:, leg]
        in_swing = False
        swings = []
        swing_start = 0
        for t in range(len(c)):
            if not c[t] and not in_swing:
                swing_start = t
                in_swing = True
            elif c[t] and in_swing:
                swings.append(t - swing_start)
                in_swing = False
        swing_phases.append(swings)

    avg_swing_duration = [float(np.mean(s)) if s else 0.0 for s in swing_phases]
    num_swings = [len(s) for s in swing_phases]

    # Step frequency: count stance-to-swing transitions
    transitions = 0
    for leg in range(4):
        c = contact[:, leg]
        transitions += np.sum(np.diff(c.astype(int)) != 0)
    step_freq = transitions / (N * 4) * 2  # transitions per step per leg

    # XY displacement and speed
    xy_disp = float(np.sqrt((pos[-1, 0] - pos[0, 0]) ** 2 + (pos[-1, 1] - pos[0, 1]) ** 2))
    mean_speed = float(np.linalg.norm(vel[50:, :2], axis=-1).mean()) if N > 50 else 0.0

    mean_duty = float(duty_cycles.mean())
    result = {
        "terrain": terrain_name,
        "total_steps": N,
        "duty_cycles": duty_cycles.tolist(),
        "mean_duty_cycle": mean_duty,
        "diagonal_correlation": correlation,
        "avg_swing_duration": avg_swing_duration,
        "num_swing_phases": num_swings,
        "step_frequency": float(step_freq),
        "xy_displacement": xy_disp,
        "mean_speed": mean_speed,
        "is_shuffling": mean_duty > 0.95,
        "is_trotting": correlation < -0.3 and mean_duty < 0.8,
    }

    return result, contact


def generate_gait_diagram(contact, output_path, terrain_name, steps=200):
    """Generate publication-quality gait diagram."""
    contacts = contact[:steps]
    leg_names = FOOT_NAMES
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    fig, ax = plt.subplots(figsize=(14, 4))

    for leg_idx, (leg_name, color) in enumerate(zip(leg_names, colors)):
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
                    linewidth=0.5,
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
                linewidth=0.5,
            )
            ax.add_patch(rect)

    ax.set_xlim(0, steps)
    ax.set_ylim(-0.5, len(leg_names) + 0.5)
    ax.set_yticks([i + 0.5 for i in range(len(leg_names))])
    ax.set_yticklabels(leg_names[::-1], fontsize=12)
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Leg", fontsize=12)
    ax.set_title(
        f"Gait Diagram - {terrain_name}\n(Colored bars = stance phase, white = swing phase)",
        fontsize=14,
    )

    for i in range(len(leg_names) + 1):
        ax.axhline(i, color="gray", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gait diagram: {output_path}")


def main():
    traj_dir = Path("results/trajectories")
    terrains = ["flat", "slopes", "stairs", "contact_aware"]

    print("=" * 70)
    print("GAIT QUALITY ANALYSIS")
    print("=" * 70)

    all_results = {}

    for terrain in terrains:
        traj_path = traj_dir / f"trajectory_{terrain}.npz"
        if not traj_path.exists():
            print(f"\n{terrain.upper()}: No trajectory file")
            continue

        result, contact = analyze_gait(str(traj_path), terrain)
        all_results[terrain] = result

        if result["is_shuffling"]:
            gait_type = "SHUFFLING"
        elif result["is_trotting"]:
            gait_type = "TROTTING"
        else:
            gait_type = "MIXED"

        print(f"\n{terrain.upper()}: {gait_type}")
        print(
            f"  Duty cycles: LF={result['duty_cycles'][0]:.2f} RF={result['duty_cycles'][1]:.2f} "
            f"LH={result['duty_cycles'][2]:.2f} RH={result['duty_cycles'][3]:.2f}"
        )
        print(f"  Mean duty cycle: {result['mean_duty_cycle']:.2f} (target: 0.5-0.7)")
        print(f"  Diagonal correlation: {result['diagonal_correlation']:.2f} (target: < -0.3)")
        print(f"  Swing phases per leg: {result['num_swing_phases']}")
        print(f"  Avg swing duration: {[f'{d:.1f}' for d in result['avg_swing_duration']]} steps")
        print(f"  Step frequency: {result['step_frequency']:.3f}")
        print(f"  XY displacement: {result['xy_displacement']:.2f}m  Speed: {result['mean_speed']:.2f}m/s")

        output_path = f"results/gait_diagram_{terrain}.png"
        generate_gait_diagram(contact, output_path, terrain)

    print("\n" + "=" * 70)
    shuffling_count = sum(1 for r in all_results.values() if r.get("is_shuffling", True))
    trotting_count = sum(1 for r in all_results.values() if r.get("is_trotting", False))
    print(f"Trotting: {trotting_count}/{len(all_results)}  Shuffling: {shuffling_count}/{len(all_results)}")
    if shuffling_count > 0:
        print(f"  {shuffling_count}/{len(all_results)} POLICIES ARE SHUFFLING -- NEED GAIT FIX")
    else:
        print("  ALL POLICIES HAVE PROPER GAIT")
    print("=" * 70)

    with open("results/gait_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved analysis to results/gait_analysis.json")


if __name__ == "__main__":
    main()
