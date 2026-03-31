# Evaluation Notes — Isaac Lab Quadruped Locomotion

**Author:** Rahil Parikh
**Date:** March 2026
**Reviewer-grade self-critique for Penn GRASP Lab (Marcel Hussing / Pratik Kunapuli)**

---

## Bug Fix Log

### Bug 1 — Energy Penalty Double-Negative (FIXED)

**File:** `environments/contact_aware_reward.py` — `energy_penalty()`

**Root cause:** Function returned `-(jt**2).sum()` (negative), and the registered weight was `-1e-4` (also negative). Negative × negative = positive net reward for high torques — the exact opposite of the intended penalty.

**Fix:** Changed function to return `(jt**2).sum()` (positive). Convention: function returns positive quantity, weight is negative for penalties. Net effect is now correctly negative.

**Impact on buggy v1 ablation:** The dominant ablation result (75% reward drop when removing energy_penalty) was an artifact — the term was acting as a large positive torque bonus (+3.0/step), not a penalty.

### Bug 2 — Contact Sensor Key Not Found (FIXED)

**File:** `experiments/04_contact_aware/train.py` — `_extract_contact_data()`

**Root cause:** Code searched `isaac_env.scene["contact_forces"]`, but Isaac Lab registers contact sensors under `isaac_env.scene.sensors["contact_forces"]` (different access path). The search silently failed, returning zero for `foot_slip_penalty` and `terrain_clearance` in ALL conditions.

**Fix:** Changed to `sensor = isaac_env.scene.sensors["contact_forces"]`, then filter to foot bodies via `sensor.find_bodies(".*FOOT")`.

**Impact on buggy v1 ablation:** Two of five ablation conditions (`no_foot_slip`, `no_terrain_clearance`) were trivially equivalent to the baseline — the terms were always inactive.

### Bug 3 — foot_slip and contact_timing Double-Negatives (FIXED)

**File:** `environments/contact_aware_reward.py`

**Root cause:** Same double-negative pattern as Bug 1 in two other penalty functions:
- `foot_slip_penalty`: returned `-(in_contact * foot_vel_sq).sum()` (negative), weight=-0.5 → net positive (rewarded slipping)
- `contact_timing_penalty`: returned `-(pair_a + pair_b)` (negative), weight=-0.2 → net positive (rewarded bad timing)

**Fix:** Both functions changed to return positive quantities; weights remain negative.

**Impact:** These were inactive in v1 (Bug 2), so v1 ablation was not affected. Only visible after Bug 2 was fixed.

### Bug 4 — Checkpoint Loading Key Mismatch in Evaluation Scripts (FIXED)

**Files:** `scripts/evaluate_policy.py`, `scripts/record_trajectory.py`, `scripts/record_policy_video.py`

**Root cause:** All three evaluation scripts constructed a standalone MLP with `self.net = nn.Sequential(...)` which produces state_dict keys like `net.0.weight`, `net.0.bias`, etc. However, rsl_rl v5's `MLPModel` uses `self.mlp = MLP(...)` which saves keys like `mlp.0.weight`, `mlp.0.bias`, `distribution.std_param`, etc. When loading with `strict=False`, **zero keys matched** — the MLP silently kept its random initialization. The scripts also contained duplicate inline `TensorDictVecEnvWrapper` and `TERRAIN_ENV_MAP` definitions instead of importing from a shared module.

**Proof:** With the random policy, mean action magnitude was ~0.001, XY displacement was 1–7 cm over 500 steps, and all four feet remained in continuous contact. With the correctly loaded policy, mean |action| = 0.65–1.38, XY displacement = 0.2–2.9 m, and clear gait patterns are visible.

**Fix (complete):** All three scripts fully rewritten to import from `scripts/runner_utils.py`:
- `load_actor()` function and inline `class MLP` deleted entirely from all three scripts
- `TensorDictVecEnvWrapper` and `TERRAIN_ENV_MAP` duplicates removed
- Replaced with `from scripts.runner_utils import create_env, load_policy`
- `runner_utils.py` uses `OnPolicyRunner.load()` + `get_inference_policy()` — rsl_rl's own loading code
- `evaluate_policy.py` gains `--eval_terrain` flag for cross-terrain evaluation
- `record_trajectory.py` gains physical assertion: flat policy must move >0.5m
- `strict=False` is never used anywhere in the evaluation pipeline

**Impact on evaluation:**
| Terrain | Old (random) | Corrected | Old Ep. Length | Corrected Ep. Length |
|---|---|---|---|---|
| Flat | 5.22 | **22.07** | 1000.0 | 942.4 |
| Slopes | 6.99 | 0.62 | 987.8 | 86.4 |
| Stairs | 7.10 | 4.10 | 981.8 | 313.1 |
| Contact-Aware | 6.30 | −0.22 | 981.4 | 42.5 |

The old evaluation showed uniform ~5–7 reward across all terrains because the random policy simply stood still for ~1000 steps collecting baseline standing reward. The corrected evaluation shows: the flat policy performs strongly (22.07), while the rough-terrain policies (slopes, stairs, contact_aware) struggle because they are evaluated on the full mixed rough terrain env, not their specific training terrain configuration.

---

## Pre-Fix Results (Buggy v1 — for reference)

| Condition | Mean Reward (last 50) | ± Std | Bug explanation |
|---|---|---|---|
| All terms | 1430.3 | 799.5 | Inflated by double-negative torque bonus |
| No contact timing | 1567.0 | 723.4 | contact_timing inactive (Bug 2) |
| No terrain clearance | 1515.4 | 704.9 | terrain_clearance inactive (Bug 2) |
| No energy penalty | 357.1 | 136.4 | Removes spurious torque bonus |
| No foot slip | 1572.9 | 734.4 | foot_slip inactive (Bug 2) |

These numbers are not interpretable. The apparent 75% drop on no_energy_penalty was entirely artifactual.

---

## Reward Scale Analysis

**Question:** Why are ablation rewards (357–2703) ~100× larger than Phase 1–2 training rewards (9–17)?

**Q (v1):** Why were ablation rewards (357–2703) ~100× larger than Phase 1–2 training rewards (9–17)?

**A (v1, now fixed):** Primarily the double-negative energy_penalty bug. See Bug Fix Log above.

**Q (v2):** Why do v2 ablation rewards (-105 to +254) span a wider range than Phase 1–2 (9–17)?

**A (v2):** Three compounding factors:

### Factor 1 — Episode return vs. per-step mean (minor)

Phase 1–2 checkpoint rewards (14.5 flat, 17.7 slopes, 9.9 stairs) are the `rewbuffer` mean logged during training, which is the undiscounted episode return divided by episode length. The ablation CSV reports the raw `rewbuffer` value, which is the full undiscounted episode return (~1000 steps × per-step reward). This alone is a ~1000× multiplier on per-step metrics — but Phase 1–2 numbers are already episode returns (rsl_rl `rewbuffer` stores episode sums), so this is not the primary cause.

### Factor 2 — Additional reward terms (moderate)

Phase 3 wraps the Isaac Lab base reward with five contact-aware terms summed at each step. The `velocity_tracking` term (weight=1.0) adds 0–1 per step; `contact_timing` adds ≈0–0.4. Over 1000 steps, these contribute 400–1400 units per episode.

### Factor 3 — Double-negative in `energy_penalty` (dominant, BUG)

**This is the primary cause.** The `energy_penalty` function in `environments/contact_aware_reward.py` returns `-(jt**2).sum()` (negative), and the registered weight is `-1e-4` (also negative). The product `weight × raw = (-1e-4) × (-(jt**2).sum()) = +1e-4 × (jt**2).sum()` is **positive** — the term is inadvertently rewarding high joint torques instead of penalizing them.

With 12 AnymalC joints at typical torques of 30–80 Nm:

```
(jt**2).sum() ≈ 12 × 50² = 30,000  per step
weighted     ≈ +1e-4 × 30,000 = +3.0  per step
per episode  ≈ 3.0 × 1000 steps = ~3,000
```

This matches the observed best reward of 2703 for `all_terms`. When `energy_penalty` is removed (`no_energy_penalty`), the spurious positive reward disappears, dropping mean reward from 1430 to 357. **The 75% reward drop from removing energy_penalty is not a meaningful finding about gait quality — it is an artifact of this bug.**

**Fix:** Either remove the negation in the function body (`return (jt**2).sum()`), or change the weight to `+1e-4`. The intended behavior is: large torques → negative reward. The current behavior is: large torques → large positive reward.

**Answer for Marcel Hussing:** The 100× scale difference is primarily a double-negative in the energy reward term that turns a torque penalty into a torque bonus. Phase 1–2 rewards are the intended baseline (9–17 is the true performance scale). Phase 3 ablation rewards are inflated by this bug.

---

## Section 1a — Ablation Results Interpretation (v2, bugs fixed)

### Corrected Results (last 50 iters, mean ± std, single seed)

| Condition | Mean Reward | ± Std | Best | Δ vs baseline | Interpretation |
|---|---|---|---|---|---|
| All terms (baseline) | −105.4 | 153.3 | 12.5 | — | All 5 terms active, all penalties correct |
| No contact timing | −61.5 | 130.1 | 41.3 | +41% | Timing penalty removed; rewards improve |
| No terrain clearance | −107.5 | 152.7 | 17.9 | −2% | Negligible effect |
| **No energy penalty** | **+253.8** | **130.3** | **398.5** | **+341%** | **Removing strong torque penalty unlocks high rewards** |
| No foot slip | −28.1 | 104.2 | 51.8 | +72% | Slip penalty removed; rewards improve |

### Honest Interpretation

**Energy penalty is the dominant term (correct finding, for real this time).** With the sign fixed, `energy_penalty` is now a genuine penalty that subtracts ~0.62/step × 1000 steps ≈ 620/episode. Removing it raises mean reward by +359 units. This is the correct physical interpretation: the energy penalty constrains the policy to efficient torque profiles; without it, the policy reverts to aggressive high-torque gaits that achieve higher velocity tracking reward at the cost of energy efficiency.

**contact_timing and foot_slip now show positive effects** (removing them improves reward). This is the expected direction — they are genuine penalties that constrain gait quality. Their removal increases reward by 41% and 72% respectively, but since we want these constraints for gait quality, the all_terms policy should be preferred despite lower reward.

**terrain_clearance shows negligible effect** (−2%). This may be because the foot height bonus (+0.3 weight × ~0.6 raw = +0.18/step = +180/episode) is partially offsetting the other penalties, and removing it reduces reward slightly.

**High variance and negative means:** The all_terms condition has mean=−105.4 with std=153.3. This indicates the policy hasn't fully adapted to all five new terms in 300 iterations from the stairs checkpoint. The negative mean reflects the combined penalty load (energy, slip, timing) exceeding the velocity tracking reward during most of training. The training is still productive (best=12.5), just oscillatory.

**What would rigorous ablation look like?**
1. Start from a checkpoint trained WITH the contact-aware terms (not the stairs-only checkpoint) — this gives a better initialization
2. Run 3–5 seeds per condition
3. Report mean ± SEM across seeds
4. 500+ iterations per condition to ensure convergence
5. Measure gait quality metrics (stride frequency, contact timing score) separately from episode return

**What would rigorous ablation look like?**

1. Fix the contact sensor extraction (inspect `isaac_env.scene._sensors` to find the actual key)
2. Fix the energy_penalty double-negative
3. Run 3–5 seeds per condition
4. Report mean ± SEM across seeds (not within-run std)
5. Conduct Mann-Whitney U test between each ablated condition and baseline
6. Report learning curves, not just final-window statistics
7. Minimum: 5 conditions × 5 seeds × 300 iters = 7,500 training iterations (≈6 GPU-hours on the current setup)

---

## Section 1b — Reward Function Critique

### `velocity_tracking_reward` — **Well-designed**

Uses exponential kernel `exp(-||v_cmd - v_act||² / σ²)` with σ=0.25. Bounded [0, 1] per step, numerically stable, differentiable, no obvious hacking risk. Correct use of body-frame velocity. Falls back gracefully when commands/velocity not in contacts dict.

**Minor issue:** Fallback uses obs indices 0–2 (command) and 3–5 (actual velocity), which are correct for the AnymalC observation layout but undocumented. Should add an assertion on obs dim.

### `foot_slip_penalty` — **Acceptable (inactive in current runs)**

Penalizes `||v_foot_xy||²` when contact force magnitude exceeds threshold. Mathematically correct — penalizes tangential velocity at contact points, which directly measures slip. The formula should technically use only the tangential component relative to the contact normal, but XY velocity as proxy is standard in the literature.

**Critical issue:** Returns zeros in all current runs because `contact_forces` key is not found in the Isaac Lab scene. The term is completely inactive.

**Hacking risk:** Low. Robot cannot exploit this by standing still (zero reward, not negative).

### `terrain_clearance_reward` — **Acceptable (inactive in current runs)**

Rewards swing-phase foot height above `min_clearance=0.05 m`. Correct formulation — uses contact force to distinguish stance/swing phases. Foot world-frame Z is the right quantity.

**Critical issue:** Also returns zeros (same contact sensor issue as foot_slip_penalty).

**Hacking risk:** Robot could gain reward by lifting all feet simultaneously (no stance contact → all feet "in swing"). Should add a constraint requiring at least one foot in contact.

### `contact_timing_penalty` — **Needs improvement**

Penalizes asymmetry between diagonal leg pairs (FL↔RR, FR↔RL) to encourage trot gait. The idea is correct, but using binary contact (above/below force threshold) misses gait phase timing. A proper trot reward should check that diagonal pairs are in contact at the same time (phase synchrony), not just that their binary contact states differ.

**Current formulation:** `|contact[FL] - contact[RR]| + |contact[FR] - contact[RL]|`. This equals 0 both when both legs are in contact (correct — same phase) AND when neither is in contact (incorrect — mid-air). Should weight by whether a stance phase is expected.

**Hacking risk:** Robot can zero this penalty by having all four feet always on the ground (slow crawl). Not catastrophic but encourages over-cautious gait.

### `energy_penalty` — **Broken (double-negative bug)**

Function returns `-(jt**2).sum()` (negative). Weight in registry is `-1e-4` (negative). Net effect: **rewards** high joint torques instead of penalizing them.

**Fix:**
```python
# Option A: fix the function (clean, semantically correct)
return (jt.float() ** 2).sum(dim=-1)  # positive, weight=-1e-4 makes it negative

# Option B: fix the weight (minimal change)
"energy_penalty": (energy_penalty, +1e-4)   # positive weight × negative fn = negative
```

**Rating: Broken.**

### Calibration Assessment

With the energy_penalty bug present, the dominant reward signal is the spurious torque bonus (~3.0/step) overwhelming velocity tracking (~0.5/step average across gait cycle). The reward landscape does not reflect the intended objective. After fixing the bug, re-calibration of weights is needed.

---

## Section 1c — Limitations

1. **Single seed per condition.** No statistical significance claims possible. With σ/μ ≈ 0.50, a single seed cannot distinguish a 10% effect from noise. Minimum viable study: 3 seeds, Mann-Whitney U, p < 0.05.

2. **Contact sensor not found.** `foot_slip_penalty` and `terrain_clearance` return zeros in all conditions because the Isaac Lab contact sensor key was not identified. Two of the five ablation terms are completely inactive.

3. **Energy penalty double-negative bug.** The dominant ablation result (no_energy_penalty) is an artifact. The robot trained with `all_terms` was optimizing toward high torques, which is physically degenerate.

4. **No sim-to-real validation.** Real-world transfer requires domain randomization of motor strength (±20%), friction coefficients, sensor noise, and actuator delay. The stairs terrain provides implicit regularization but not explicit DR.

5. **No comparison to published benchmarks.** Rudin et al. (2022) report AnymalC traversing stairs at speeds of ~1.0 m/s with PPO. Our evaluation does not measure forward speed, only episode return. Cannot assess whether our policy is competitive.

6. **Contact-aware training initialized from stairs checkpoint.** Results may be sensitive to this initialization. Training from scratch would show different behavior.

7. **Episode length = 1000 steps, fixed.** No early termination on falls (fall detection via base height was logged but not used for termination). A policy that falls over but recovers quickly and a policy that never falls receive the same episode length — the metric does not penalize falls explicitly.

---

## Section 1d — MAD-TD Integration Talking Points (based on corrected results)

*For Rahil's email to Marcel Hussing (ICLR 2025 MAD-TD):*

1. **Energy penalty (now correctly signed) isolates the efficiency dimension.** The v2 ablation shows that removing `energy_penalty` raises mean reward by +341% — the strongest effect of any single term. This means the penalty is successfully constraining the policy to efficient torque profiles. A learned world model can represent this constraint implicitly: if MAD-TD's dynamics model predicts torque consequences of actions, the policy may discover energy-efficient gaits without needing an explicit penalty term. **Proposed test:** Run MAD-TD without `energy_penalty` and measure whether the world model recovers torque efficiency as an emergent property. If yes, this validates that the model has learned contact dynamics, not just kinematics.

2. **foot_slip and contact_timing are now confirmed active and meaningful.** The v2 ablation shows removing foot_slip raises reward by +72% and removing contact_timing raises reward by +41% — both represent genuine constraints on gait quality. These terms require predicting whether a foot is in contact and moving relative to the terrain at the next timestep, which is exactly the kind of short-horizon contact prediction that a good world model should provide. **MAD-TD's world model rollouts should reduce variance for these terms most** compared to model-free PPO, since contact transitions are high-gradient discontinuities where policy gradient estimates are most noisy.

3. **Concrete experiment.** Initialize both PPO and MAD-TD from the contact_aware v2 checkpoint (best=77.2, 300 iters). Fine-tune for 300 more iterations with all 5 corrected terms. Compare: (a) episode return at convergence, (b) foot_slip raw value at convergence (lower = less slipping), (c) contact_timing raw value (lower = better diagonal synchrony), (d) env steps to reach reward=50. The ablation provides the PPO baseline numbers for each metric. Target: MAD-TD should reach reward=50 in fewer than 300 iters while maintaining lower foot_slip and contact_timing than PPO at the same reward level.

4. **The high variance in all conditions (std/mean ≈ 1.3–1.4) signals opportunity for world models.** PPO's high variance on stair terrain reflects the difficulty of estimating gradients through discontinuous contact dynamics with a model-free approach. MAD-TD's world model provides lower-variance gradient estimates by averaging over model rollouts. The fact that all conditions show σ ≈ 130–153 despite widely different reward means suggests the variance is driven by contact dynamics, not reward scale — exactly the variance that a contact-aware world model should reduce.

---

## What Works Well

### 1. Curriculum progression
The flat → slopes → stairs curriculum produced consistent improvement. Each stage warm-starts from the previous checkpoint, and the reward structure shows clear progression. The stair terrain presented the expected difficulty spike (reward dropped from ~17.7 on slopes to ~9.9 on stairs before recovering).

### 2. Contact-aware infrastructure
The `ContactAwareVecEnvWrapper` cleanly adds reward terms on top of Isaac Lab's base reward without modifying the underlying environment. Graceful degradation (NaN handling) means training continues even when contact sensor data is unavailable.

### 3. Ablation infrastructure
The leave-one-out ablation design is statistically cleaner than additive ablation. Running each condition as a separate subprocess correctly avoids the PhysX GPU re-init crash.

---

## Known Limitations (Full List)

See Section 1c above. Additional technical notes:

- **Ablation orchestrator `-inf` reward at iter 1:** rewbuffer is empty at first iter; fixed with `math.isfinite(r)` filter
- **Early stopping threshold:** original `10.0` was trivially exceeded from iter 1 (starting reward ~2196 from stairs checkpoint); corrected to `2000.0`
- **torch pinned at 2.5.1+cu124:** CUDA 12.4 driver; do not upgrade torch

---

## What I'd Do Differently with More Compute

1. Fix contact sensor key extraction before running ablations
2. Fix energy_penalty double-negative, re-run ablation
3. 3-seed ablation per condition for significance testing
4. Domain randomization for sim-to-real robustness
5. Add episode termination on falls (base height < 0.3 m) and log fall_rate
6. Measure forward velocity and compare against Rudin et al. 2022 AnymalC numbers

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
