# Evaluation Notes — Isaac Lab Quadruped Locomotion

**Author:** Rahil Parikh
**Date:** March–April 2026
**Reviewer-grade self-critique for Penn GRASP Lab (Marcel Hussing / Pratik Kunapuli)**

---

## Training History

### v1 (Initial): Early-stopped, low thresholds
- Flat: 138 iters, reward 14.5 (threshold=8.0)
- Slopes: 175 iters, reward 17.7 (threshold=7.0)
- Stairs: 174 iters, reward 9.9 (threshold=6.5)
- Contact-aware: 300 iters, reward 77.2 (energy penalty -1e-4 killed locomotion)
- **Problem:** Stairs policy balanced but didn't climb. Contact-aware froze.

### v2 (Retrained to convergence):
- Flat: 1500 iters, reward 25.85, XY=10.57m, speed=1.26m/s
- Slopes: 318 iters (early stop at 21.09), XY=1.95m, speed=0.95m/s
- Stairs: 1006 iters (early stop at 15.76), XY=5.64m, speed=0.70m/s
- Contact-aware: 500 iters, reward 762.9, XY=6.71m, speed=0.83m/s
- **Fix:** Raised early-stop thresholds (25.0 flat, 20.0 slopes, 15.0 stairs), trained more iters
- **Fix:** Reduced energy_penalty weight from -1e-4 to -1e-6

### Energy Penalty Fix
**Root cause:** `energy_penalty` returns `sum(torque^2)` which is ~7900 for AnymalC. At weight -1e-4, this is -0.79/step, completely dominating all other rewards. The base Isaac Lab already has `dof_torques_l2` at -1e-5 weight. The contact-aware energy penalty at -1e-4 added 10x MORE torque penalty on top, forcing the policy to minimize all joint movement (freezing).

**Fix:** Weight reduced from -1e-4 to -1e-6, making per-step contribution -0.008 instead of -0.79. This is 10x LESS than the base torque penalty — a gentle refinement, not a dominator.

---

## Bug Fix Log

### Bug 1 — Energy Penalty Weight Too Aggressive (FIXED v2)

**File:** `experiments/04_contact_aware/config.py` and `environments/contact_aware_reward.py`

**Root cause:** Weight -1e-4 × raw torque^2 ~7900 = -0.79/step dominated all other rewards. Policy learned to freeze (minimize torque) instead of walk.

**Fix:** Weight changed to -1e-6. Per-step contribution now -0.008, allowing locomotion while still penalizing energy.

### Bug 2 — Contact Sensor Key Not Found (FIXED v1)

**File:** `experiments/04_contact_aware/train.py` — `_extract_contact_data()`

**Root cause:** Code searched `isaac_env.scene["contact_forces"]`, but Isaac Lab registers contact sensors under `isaac_env.scene.sensors["contact_forces"]`.

**Fix:** Changed to `sensor = isaac_env.scene.sensors["contact_forces"]`, then filter to foot bodies via `sensor.find_bodies(".*FOOT")`.

### Bug 3 — Checkpoint Loading Key Mismatch in Evaluation Scripts (FIXED v1)

**Files:** `scripts/evaluate_policy.py`, `scripts/record_trajectory.py`, `scripts/record_policy_video.py`

**Root cause:** Evaluation scripts used `self.net` (keys `net.0.weight`) but rsl_rl v5 saves `self.mlp` (keys `mlp.0.weight`). With `strict=False`, zero keys matched — random policy silently used.

**Fix:** All scripts now use `OnPolicyRunner.load()` + `get_inference_policy()` via `scripts/runner_utils.py`. `strict=False` is never used.

---

## Ablation Results Interpretation (v3)

### Design
- Leave-one-out over 5 contact-aware reward terms
- 500 iterations per condition (up from 300 in v2)
- Started from walking stairs checkpoint (verified XY > 5m)
- Energy penalty corrected to -1e-6
- All conditions physically verified walking

### Results

| Condition | Mean Reward (last 50) | ± Std | Best | Δ vs Baseline | XY Disp. |
|---|---|---|---|---|---|
| All terms (baseline) | 469.2 | 132.1 | 650.3 | — | 2.75m |
| No contact timing | 510.9 | 135.7 | 683.3 | +9% | 3.12m |
| **No terrain clearance** | **371.7** | **111.9** | **543.5** | **-21%** | **3.99m** |
| No energy penalty | 475.6 | 146.4 | 687.8 | +1% | 3.81m |
| **No foot slip** | **607.5** | **166.1** | **790.7** | **+29%** | **1.93m** |

### Interpretation

**terrain_clearance is the most valuable term.** Removing it causes the largest reward drop (-21%). This confirms that explicit foot-lifting incentives are critical for stair traversal — without them, the policy reverts to shuffling gaits that earn less base reward. The z-range for no_terrain_clearance (0.58m) is lower than the baseline (0.94m), consistent with less foot lifting.

**foot_slip_penalty is the strongest constraint.** Removing it boosts reward by +29%, but the resulting gait has lower XY displacement (1.93m vs 2.75m in 200 steps) and lower z-range (0.25m). The higher reward comes from the absence of slip penalties, not from better locomotion.

**contact_timing has minimal individual effect (+9%).** At weight -0.2, the diagonal gait synchrony penalty is relatively soft. May need higher weight to show meaningful effect.

**energy_penalty is negligible at -1e-6 (+1%).** At the corrected weight, the per-step energy contribution is -0.008, which doesn't meaningfully constrain the policy. This is expected — the base Isaac Lab `dof_torques_l2` at -1e-5 already provides 10x more torque penalty.

### Key improvement over v2 ablation
- v2: All conditions had negative mean rewards (-107 to +254) because training started from a non-walking stairs checkpoint with an over-aggressive energy penalty
- v3: All conditions have positive mean rewards (372 to 608) and all produce walking policies, making the reward differences meaningful for comparing term importance

---

## Limitations

1. **Single seed per condition.** No statistical significance. Minimum viable: 3 seeds, Mann-Whitney U, p < 0.05.
2. **Slopes evaluation terrain mismatch.** Evaluation runs on generic rough terrain (Isaac-Velocity-Rough-Anymal-C-v0), not the 0-15 degree slopes used in training. This causes short episode lengths (87 steps) despite the policy walking when tested on appropriate terrain.
3. **No sim-to-real validation.** Requires domain randomization of motor strength, friction, sensor noise, actuator delay.
4. **No comparison to published benchmarks.** Rudin et al. (2022) report AnymalC at ~1.0 m/s on stairs. Our stairs policy achieves 0.70 m/s — competitive but not matching.
5. **Contact-aware eval reward is negative (-2.78)** because evaluation only uses base Isaac Lab reward, not the contact-aware terms. This makes cross-stage reward comparison misleading.

---

## MAD-TD Integration Talking Points (v3)

1. **terrain_clearance isolates the foot-lifting dimension.** The -21% drop when removed confirms this term drives stair traversal behavior. A learned world model that predicts foot-terrain clearance at the next timestep could replace this explicit reward term. Test: run MAD-TD without terrain_clearance and measure whether the world model learns foot-lifting as an emergent property.

2. **foot_slip_penalty constrains gait quality.** The +29% boost when removed shows it's a genuine constraint. Predicting foot-ground contact and slip at the next timestep is exactly the capability a world model should provide. MAD-TD's model rollouts should reduce variance for this term.

3. **Concrete experiment.** Initialize both PPO and MAD-TD from the contact_aware checkpoint (best=762.9). Fine-tune for 500 iterations. Compare: (a) episode return at convergence, (b) terrain_clearance raw value (higher = better foot lifting), (c) foot_slip raw value (lower = less slipping), (d) env steps to reach reward=500. The v3 ablation provides PPO baselines.

---

## Citation

```
@misc{parikh2026isaaclab,
  title={Isaac Lab Quadruped Locomotion with Terrain Curriculum and Contact-Aware Reward Ablation},
  author={Parikh, Rahil},
  year={2026},
  note={Pre-arrival research project for Penn GRASP Lab},
  url={https://github.com/rkparikh77/isaac-lab-locomotion}
}
```
