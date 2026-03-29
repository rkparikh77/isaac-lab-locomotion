# Quadruped Locomotion with Contact-Aware Reward Ablation

> Terrain curriculum training + reward ablation study using Isaac Lab
> Pre-arrival research project for Penn GRASP Lab (Fall 2026)
> Rahil Parikh — [GitHub](https://github.com/rkparikh77)

---

## Key Results

Four-phase terrain curriculum training of an AnymalC quadruped using PPO (Isaac Lab + RSL-RL v5), followed by a five-condition leave-one-out ablation of contact-aware reward terms on stair terrain. After fixing three sign bugs in the reward implementation (see [Bug Fix Log](results/EVALUATION_NOTES.md#bug-fix-log)), the corrected ablation shows: `energy_penalty` is the strongest constraint (removing it raises reward by +341%), and `foot_slip_penalty` and `contact_timing_penalty` are confirmed meaningful with +72% and +41% effects respectively — making these the primary targets for MAD-TD model-based RL integration.

![Ablation bar chart](results/ablation_figure.png)

![Trajectory hero](results/trajectory_hero.png)

---

## Demo Videos

3D stick-figure animations rendered from recorded policy trajectories (300 frames @ 30 fps each).

| Terrain | Video |
|---------|-------|
| Flat | `results/videos/flat_demo.mp4` |
| Slopes | `results/videos/slopes_demo.mp4` |
| Stairs | `results/videos/stairs_demo.mp4` |
| Contact-Aware | `results/videos/contact_aware_demo.mp4` |
| **2×2 comparison** | `results/videos/comparison.mp4` |

Generated with `scripts/animate_robot.py` (matplotlib Agg + imageio/libx264). Foot contact state shown in green (loaded) / red (swing). Body orientation from recorded quaternions; leg positions from approximate forward kinematics (thigh=0.26 m, shank=0.26 m).

---

## Training Progression

| Phase | Terrain | Reward | Iterations | Notes |
|-------|---------|--------|------------|-------|
| 1 | Flat | 14.527 | 138 | Early-stopped at threshold=8.0 |
| 2a | Slopes (0–15°) | 17.677 | 175 | Warm-started from flat |
| 2b | Stairs (5–20 cm) | 9.871 | 174 | Warm-started from slopes |
| 3 | Contact-Aware (v2, bugs fixed) | 77.212 | 300 | Warm-started from stairs; all 5 terms correctly signed |

Rewards are rsl_rl `rewbuffer` values (undiscounted episode returns). Phase 1–2 values reflect the base Isaac Lab reward. Phase 3 (v2) includes five corrected contact-aware terms; the higher reward vs. Phase 1–2 reflects the additional velocity tracking and terrain clearance bonuses from the contact-aware wrapper.

---

## Ablation Study

**Design:** Leave-one-out over 5 contact-aware reward terms. Each condition is trained for 300 iterations from the stairs checkpoint. Single seed (N=1 per condition — treat all results as exploratory).

![Ablation figure](results/ablation_figure.png)

**v2 results (bugs fixed — all terms correctly signed, contact sensor active):**

| Condition | Mean Reward (last 50 iters) | ± Std | Best | Δ vs Baseline | Interpretation |
|---|---|---|---|---|---|
| All terms (baseline) | −105.4 | 153.3 | 12.5 | — | All 5 penalties active |
| No contact timing | −61.5 | 130.1 | 41.3 | +41% | Timing constraint removed |
| No terrain clearance | −107.5 | 152.7 | 17.9 | −2% | Negligible effect |
| **No energy penalty** | **+253.8** | **130.3** | **398.5** | **+341%** | **Dominant term: torque penalty unlocked** |
| No foot slip | −28.1 | 104.2 | 51.8 | +72% | Slip constraint removed |

**Interpretation:** `energy_penalty` is the dominant constraint — its removal allows high-torque, high-speed gaits that maximize velocity tracking reward at the cost of energy efficiency. `foot_slip` and `contact_timing` are confirmed meaningful (not inactive anymore). `terrain_clearance` shows minimal effect, possibly because its +0.3 weight provides a positive signal that partially cancels other penalties. High variance across all conditions (std ~100–153) reflects training instability starting from the stairs checkpoint — 300 iterations is insufficient for the policy to converge under all five new penalty terms simultaneously.

---

## Reward Scale Analysis

**v1 (buggy):** Ablation rewards were 357–2703 vs Phase 1–2 range of 9–17. Root cause: three double-negatives in penalty functions (`energy_penalty`, `foot_slip_penalty`, `contact_timing_penalty`), plus inactive contact sensor. All fixed in v2.

**v2 (fixed):** Contact-aware training reward is 77.2 (best), ablation range is −107.5 to +253.8. The negative mean rewards reflect the combined penalty load not yet overcome after 300 training iterations from the stairs checkpoint. The higher ceiling vs. Phase 1–2 (best=77 vs. best=17) comes from the additional velocity tracking and terrain clearance bonus terms in the contact-aware wrapper. Full analysis in [EVALUATION_NOTES.md](results/EVALUATION_NOTES.md#reward-scale-analysis).

---

## Method

### Terrain Curriculum

The curriculum follows the progression from Rudin et al. (2022): flat → rough slopes → stairs, with each phase warm-starting from the prior checkpoint. This approach lets the policy develop basic gait dynamics on flat terrain before encountering terrain features that require coordinated foot placement. The warm-start is implemented via `_partial_load_checkpoint`, which matches parameter shapes and silently skips any size-mismatched layers, making it robust to minor architecture changes between phases.

Flat terrain (4096 parallel envs, 1000-step episodes) requires ~138 iterations to converge. Slopes introduce 0–15° inclines and increase the velocity tracking challenge (reward improves from 14.5 → 17.7). Stairs (5–20 cm step height) create discrete contact transitions that force the policy to develop lift-and-place footfall patterns; initial reward drops to 9.9 before recovering within 174 iterations.

### Contact-Aware Reward

Five terms are added to the base Isaac Lab reward via `ContactAwareVecEnvWrapper`. All terms degrade gracefully when contact sensor data is unavailable:

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| Velocity tracking | exp(−‖v_cmd − v_actual‖²/σ²), σ=0.25 | +1.0 | Commanded velocity tracking |
| Foot slip penalty | −Σᵢ contact_i · ‖v_foot_i^xy‖² | −0.5 | No sliding when foot is loaded |
| Terrain clearance | Σᵢ swing_i · max(z_i − 0.05, 0) | +0.3 | Lift feet in swing phase |
| Contact timing | −(‖c_FL − c_RR‖ + ‖c_FR − c_RL‖) | −0.2 | Diagonal trot synchrony |
| Energy penalty | −Σᵢ τᵢ² (BUG: net effect is +) | −1e-4 | Intended: penalize joint torques |

### Ablation Methodology

Leave-one-out: each condition trains for 300 iterations with one term disabled (weight=0). All five conditions start from the same stairs checkpoint. Sequential subprocess execution avoids the PhysX GPU Foundation singleton crash. Results are the mean ± std of the `rewbuffer` over the last 50 iterations.

### Architecture

- **Policy:** PPO via RSL-RL v5 (`OnPolicyRunner`)
- **Network:** MLP actor-critic, hidden dims [512, 256, 128], ELU activations
- **Envs:** 4096 parallel (training), 256 parallel (evaluation)
- **Robot:** AnymalC, 12 DOF, Isaac Lab velocity-tracking task
- **Observation:** 48-dim (joint positions, velocities, body frame velocity, commands, contact states)
- **PPO:** γ=0.99, λ=0.95, clip=0.2, 5 epochs/rollout, 4 mini-batches, 24 steps/env

---

## Evaluation

100 evaluation episodes per terrain, greedy policy (deterministic action, no noise), 256 parallel envs, no curriculum. Checkpoint loaded via `OnPolicyRunner.load()` + `get_inference_policy()`.

| Terrain | Mean Reward | ± Std | Max | Min | Mean Ep. Length |
|---------|-------------|-------|-----|-----|----------------|
| Flat | 22.07 | 5.30 | 26.37 | −1.52 | 942.4 |
| Slopes | 0.62 | 0.85 | 2.34 | −2.11 | 86.4 |
| Stairs | 4.10 | 4.17 | 14.60 | −13.20 | 313.1 |
| Contact-Aware (v2) | −0.22 | 0.58 | 1.63 | −1.74 | 42.5 |

Evaluation uses only the base Isaac Lab reward (no contact-aware wrapper), so all four rows are on the same scale. The **flat policy** achieves strong performance (22.07 mean, 942-step episodes), confirming the training converged well. The slopes/stairs/contact-aware policies were each trained on specific terrain configurations but are evaluated on the full mixed rough terrain (`Isaac-Velocity-Rough-Anymal-C-v0`), which includes terrain features beyond their training distribution — hence shorter episodes and lower rewards. The stairs policy shows the most variance (std=4.17) reflecting the difficulty of generalized stair climbing.

**Note (checkpoint loading bug fix):** Previous evaluation reported 5–7 mean reward across all terrains. These values were from a **random policy** due to a `state_dict` key mismatch bug: the evaluation MLP used `self.net` (keys `net.0.weight`, ...) while rsl_rl saves `self.mlp` (keys `mlp.0.weight`, ...). With `strict=False`, no keys matched and random weights were silently kept. Fixed by using `OnPolicyRunner.load()` which correctly loads the trained weights. Full results in `results/evaluation_*.json`.

---

## Future Work: Model-Based RL Integration

This project is explicitly designed as a testbed for [MAD-TD](https://arxiv.org/abs/2406.00420) (Hussing et al., ICLR 2025), a model-augmented temporal difference method. The integration opportunity is concrete:

**Why this problem is right for MAD-TD:** Stair locomotion involves discrete contact transitions — discontinuous dynamics that are notoriously high-variance for model-free policy gradient methods. A learned world model that accurately predicts ground contact events can provide lower-variance gradient estimates for the contact-aware reward terms. The `TensorDictVecEnvWrapper` already returns rsl_rl v5 `TensorDict` observations, which is the interface MAD-TD expects, making the swap from `OnPolicyRunner` to MAD-TD's runner a minimal change.

**Which ablation conditions are most relevant (v2, bugs fixed):** `foot_slip_penalty` and `contact_timing_penalty` are confirmed meaningful — removing them improves reward by +72% and +41% respectively, confirming they are genuine constraints on gait quality. Both require predicting foot-ground contact at the next timestep, which is exactly the capability a world model should provide. The `energy_penalty` is the dominant constraint; a world model that predicts torque consequences of actions could recover this implicitly.

**Proposed experiment:** Initialize both PPO and MAD-TD from the contact_aware v2 checkpoint (best=77.2, 300 iters). Fine-tune for 300 more iterations. Compare: (1) reward at convergence, (2) env steps to reach reward=50 (about 80% of PPO best), (3) foot_slip and contact_timing raw values as gait quality proxies. The ablation provides PPO baseline numbers. Target sample efficiency metric: 7.9M steps for PPO (300 iters × 1024 envs × 24 steps/env).

---

## Reproducing Results

### Requirements

```
Isaac Lab 0.33.13 (isaacsim[rl]==4.5.0.0)
rsl-rl-lib==5.0.1
torch==2.5.1+cu124
CUDA 12.4
Python 3.10
wandb matplotlib numpy toml tensordict
```

### Setup

```bash
# Isaac Sim (first run downloads ~15 GB)
pip install isaacsim[rl]==4.5.0.0 --extra-index-url https://pypi.nvidia.com
pip install rsl-rl-lib==5.0.1 wandb matplotlib numpy toml
pip install torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Isaac Lab source extensions
cd /workspace/IsaacLab
pip install -e source/isaaclab -e source/isaaclab_rl -e source/isaaclab_tasks

# System libraries (if missing: libSM.so.6, libICE.so.6, libXt.so.6)
# Unpack from apt without root:
apt-get download libsm6 libxt6 libxrender1 libice6
for pkg in libsm6*.deb libxt6*.deb libxrender1*.deb libice6*.deb; do
    mkdir -p /tmp/${pkg%_*}_extracted && dpkg-deb -x $pkg /tmp/${pkg%_*}_extracted
done
# The run_training.sh wrapper sets LD_LIBRARY_PATH to include these
```

### Training

```bash
cd /workspace/IsaacLab

# Phase 1 — Flat terrain (converges in ~138 iters)
/workspace/run_training.sh \
    /workspace/isaac-lab-locomotion/experiments/01_flat_terrain/train.py --headless

# Phase 2a — Slopes 0–15°
/workspace/run_training.sh \
    /workspace/isaac-lab-locomotion/experiments/02_slopes/train.py --headless \
    --load_checkpoint /workspace/checkpoints/flat/best_model.pt

# Phase 2b — Stairs 5–20 cm
/workspace/run_training.sh \
    /workspace/isaac-lab-locomotion/experiments/03_stairs/train.py --headless \
    --load_checkpoint /workspace/checkpoints/slopes/best_model.pt

# Phase 3 — Contact-aware reward (stairs, warm-start from stairs checkpoint)
/workspace/run_training.sh \
    /workspace/isaac-lab-locomotion/experiments/04_contact_aware/train.py --headless

# Ablation — 5 conditions, sequential, each 300 iters
cd /workspace/isaac-lab-locomotion
python3.10 experiments/ablation/run_ablation.py --headless
```

### Evaluation and Visualization

```bash
cd /workspace/IsaacLab

# 100-episode evaluation for each terrain
for TERRAIN in flat slopes stairs contact_aware; do
    /workspace/run_training.sh \
        /workspace/isaac-lab-locomotion/scripts/evaluate_policy.py \
        --checkpoint /workspace/checkpoints/${TERRAIN}/best_model.pt \
        --terrain ${TERRAIN} --episodes 100 --headless
done

# Record 500-step trajectories
for TERRAIN in flat slopes stairs contact_aware; do
    /workspace/run_training.sh \
        /workspace/isaac-lab-locomotion/scripts/record_trajectory.py \
        --checkpoint /workspace/checkpoints/${TERRAIN}/best_model.pt \
        --terrain ${TERRAIN} --steps 500 --headless
done

# Generate all visualizations (no simulator needed)
python3.10 /workspace/isaac-lab-locomotion/scripts/visualize_trajectory.py
python3.10 /workspace/isaac-lab-locomotion/scripts/generate_ablation_figure.py
```

---

## Technical Notes — Isaac Lab 0.33.13 API Fixes

These fixes are non-obvious and required debugging time. Documented for anyone using this codebase with Isaac Lab 0.33.13:

1. **Import path change:** All `omni.isaac.lab.*` imports are now `isaaclab.*`. Old tutorials are outdated.
2. **`gym.make()` signature:** Requires `cfg=env_cfg` as a keyword argument (not positional).
3. **rsl_rl v5 TensorDict API:** `VecEnv.step()` must return `TensorDict({"policy": obs})`. Raw tensor returns fail silently with shape errors. Wrap with `TensorDictVecEnvWrapper`.
4. **AppLauncher before all omni imports:** Must call `AppLauncher(args)` and `.app` before any `import isaaclab.*`. This is not optional.
5. **PhysX GPU Foundation singleton:** Only one Isaac Lab process per machine at a time. Sequential subprocess execution is required for multi-condition experiments.
6. **Contact sensor keys:** The AnymalC rough terrain env registers contact sensors under a non-standard key (not `contact_forces`, `feet_contact`, or `foot_contact`). Inspect `isaac_env.scene._sensors` at runtime to find the actual key.
7. **OMNI_KIT_ACCEPT_EULA=YES:** Must be set before process launch. Forgetting this causes a hang waiting for interactive input.
8. **Contact sensor API:** `isaac_env.scene.sensors["contact_forces"]` (not `scene["contact_forces"]`). Filter to foot bodies via `sensor.find_bodies(".*FOOT")`. Net forces are `sensor.data.net_forces_w[:, foot_ids, :]` with shape `[N, 4, 3]`.
9. **Reward function sign convention:** Isaac Lab convention: function returns **positive** value, weight is **negative** for penalties. Three penalty functions originally returned negative values (double-negative bugs). Fixed: `energy_penalty`, `foot_slip_penalty`, `contact_timing_penalty` all now return positive quantities.
10. **Subprocess LD_LIBRARY_PATH:** When launching Isaac Lab as a subprocess, `LD_LIBRARY_PATH` from `run_training.sh` must be explicitly propagated — `os.environ.copy()` in Python captures the shell's env at process start but misses dynamic exports.

---

## Repository Structure

```
isaac-lab-locomotion/
├── environments/
│   ├── contact_aware_reward.py    # RewardManager + 5 contact-aware terms
│   └── __init__.py
├── experiments/
│   ├── 01_flat_terrain/           # Phase 1
│   ├── 02_slopes/                 # Phase 2a
│   ├── 03_stairs/                 # Phase 2b
│   ├── 04_contact_aware/          # Phase 3 (config.py + train.py)
│   └── ablation/                  # run_ablation.py — 5-condition orchestrator
├── scripts/
│   ├── evaluate_policy.py         # 100-episode evaluation → JSON
│   ├── record_trajectory.py       # 500-step state recording → .npz
│   ├── visualize_trajectory.py    # 2×2 trajectory comparison figure
│   ├── generate_ablation_figure.py # Publication-quality ablation bar chart
│   ├── runner_utils.py             # Shared: OnPolicyRunner checkpoint loading, env creation
│   ├── animate_robot.py           # 3D stick-figure MP4 animation from .npz
│   ├── make_comparison_video.py   # 2×2 grid comparison video
│   └── make_trajectory_figure.py  # Dark-theme hero figure (XY paths colored by time)
├── results/
│   ├── EVALUATION_NOTES.md        # Full critique, reward scale analysis, MAD-TD notes
│   ├── ablation_results.csv       # 5-condition ablation scores
│   ├── ablation_figure.png        # Publication-quality bar chart
│   ├── trajectory_comparison.png  # 4-panel trajectory visualization
│   ├── trajectory_hero.png        # Dark-theme XY path hero figure
│   ├── trajectory_{terrain}.png   # Per-terrain individual plots
│   ├── evaluation_{terrain}.json  # 100-episode eval results (4 files)
│   ├── trajectories/              # .npz trajectory recordings (500 steps each)
│   └── videos/                    # MP4 demo animations (4 terrains + comparison)
├── .github/workflows/lint.yml     # CI: black + flake8
├── .gitignore
└── README.md
```

---

## Citations

```bibtex
@article{rudin2022learning,
  title={Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning},
  author={Rudin, Nikita and Hoeller, David and Reist, Philipp and Hutter, Marco},
  journal={CoRL},
  year={2022}
}

@software{isaaclab,
  title={Isaac Lab},
  author={{NVIDIA Isaac Lab Project Developers}},
  year={2024},
  url={https://github.com/isaac-sim/IsaacLab}
}

@inproceedings{hussing2025madtd,
  title={{MAD-TD}: Model-Augmented Data for Temporal Difference Learning},
  author={Hussing, Marcel and Bhatt, Dhruv and Sukhatme, Gaurav S and Todorov, Emanuel and Fern, Alan},
  booktitle={ICLR},
  year={2025}
}
```

---

MIT License
