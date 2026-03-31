"""
Trajectory recording script — runs policy for N steps and saves state data.

Usage
-----
    cd /workspace/IsaacLab
    /workspace/run_training.sh \\
        /workspace/isaac-lab-locomotion/scripts/record_trajectory.py \\
        --checkpoint /workspace/checkpoints/flat/best_model.pt \\
        --terrain flat \\
        --steps 500

Outputs
-------
    /workspace/isaac-lab-locomotion/results/trajectories/trajectory_{terrain}.npz

Saved arrays (all indexed by step):
    base_position       [steps, 3]   - world-frame XYZ
    base_orientation    [steps, 4]   - quaternion (w, x, y, z)
    joint_positions     [steps, N_dof]
    foot_contact_forces [steps, 4, 3] - zeros if sensor unavailable
    base_linear_velocity [steps, 3]
    base_angular_velocity [steps, 3]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record AnymalC policy trajectory.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--terrain",
    type=str,
    choices=["flat", "slopes", "stairs", "contact_aware"],
    default="flat",
)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.runner_utils import create_env, load_policy  # noqa: E402


def record(checkpoint: str, terrain: str, num_steps: int, num_envs: int = 1, device: str = "cuda"):
    env = create_env(terrain, num_envs)
    policy = load_policy(checkpoint, env, device=device)

    base_positions = []
    base_orientations = []
    joint_positions = []
    foot_contact_forces = []
    base_linear_velocities = []
    base_angular_velocities = []

    obs_td, _ = env.reset()

    with torch.no_grad():
        for _ in range(num_steps):
            actions = policy(obs_td)
            obs_td, _, _, _ = env.step(actions)

            try:
                isaac_env = env._env.unwrapped
                rd = isaac_env.scene["robot"].data

                base_positions.append(rd.root_pos_w[0].cpu().numpy())
                base_orientations.append(rd.root_quat_w[0].cpu().numpy())
                joint_positions.append(rd.joint_pos[0].cpu().numpy())
                base_linear_velocities.append(rd.root_lin_vel_w[0].cpu().numpy())
                base_angular_velocities.append(rd.root_ang_vel_w[0].cpu().numpy())

                # Contact forces — try multiple sensor locations
                cf_found = False
                for sensor_key in ["contact_forces", "feet_contact", "foot_contact"]:
                    try:
                        sensor = isaac_env.scene.sensors[sensor_key]
                        cf = sensor.data.net_forces_w[0].cpu().numpy()  # [4, 3]
                        foot_contact_forces.append(cf)
                        cf_found = True
                        break
                    except (KeyError, AttributeError):
                        pass
                if not cf_found:
                    # Fallback: try scene dict directly
                    for sensor_key in ["contact_forces", "feet_contact", "foot_contact"]:
                        try:
                            sensor = isaac_env.scene[sensor_key]
                            cf = sensor.data.net_forces_w[0].cpu().numpy()
                            foot_contact_forces.append(cf)
                            cf_found = True
                            break
                        except (KeyError, AttributeError):
                            pass
                if not cf_found:
                    foot_contact_forces.append(np.zeros((4, 3), dtype=np.float32))

            except Exception:
                # If state extraction fails, fill with zeros
                base_positions.append(np.zeros(3, dtype=np.float32))
                base_orientations.append(np.array([1, 0, 0, 0], dtype=np.float32))
                joint_positions.append(np.zeros(12, dtype=np.float32))
                foot_contact_forces.append(np.zeros((4, 3), dtype=np.float32))
                base_linear_velocities.append(np.zeros(3, dtype=np.float32))
                base_angular_velocities.append(np.zeros(3, dtype=np.float32))

    env.close()

    # Physical verification
    pos = np.array(base_positions)
    vel = np.array(base_linear_velocities)

    if len(pos) > 50:
        xy_disp = np.sqrt((pos[-1, 0] - pos[0, 0]) ** 2 + (pos[-1, 1] - pos[0, 1]) ** 2)
        mean_speed = np.linalg.norm(vel[50:], axis=-1).mean()
        print(f"PHYSICAL CHECK: XY={xy_disp:.3f}m  speed={mean_speed:.3f}m/s", flush=True)
        if terrain == "flat":
            assert xy_disp > 0.5, f"FLAT POLICY NOT WALKING: only {xy_disp:.3f}m displacement"

    out_dir = pathlib.Path(__file__).resolve().parents[1] / "results" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trajectory_{terrain}.npz"

    np.savez(
        out_path,
        base_position=pos,
        base_orientation=np.array(base_orientations),
        joint_positions=np.array(joint_positions),
        foot_contact_forces=np.array(foot_contact_forces),
        base_linear_velocity=vel,
        base_angular_velocity=np.array(base_angular_velocities),
    )
    print(f"Saved trajectory → {out_path}", flush=True)
    return str(out_path)


if __name__ == "__main__":
    record(
        checkpoint=args_cli.checkpoint,
        terrain=args_cli.terrain,
        num_steps=args_cli.steps,
        num_envs=args_cli.num_envs,
    )
    simulation_app.close()
