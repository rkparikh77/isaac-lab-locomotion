# FINAL REPORT: AnymalC Quadruped Locomotion with Verified Trotting Gait

**Date:** 2026-04-01
**Context:** Pre-arrival research project for Penn GRASP Lab (Marcel Hussing / Pratik Kunapuli)

---

## 1. Summary

All 4 AnymalC locomotion policies exhibit **proper trotting gait** with clear swing phases, diagonal pair alternation, and sub-0.7 duty cycles. No policy is shuffling. Ground truth validation from trajectory .npz data confirms all policies walk stably on their trained terrain.

---

## 2. Ground Truth Validation -- ALL PASS

Criteria: terrain-relative height stability, XY displacement, speed > 0.1 m/s.

| Terrain | Stable Steps | Height Range (m) | XY Disp (m) | Speed (m/s) | Status |
|---|---|---|---|---|---|
| Flat | 500/500 | [0.57, 0.60] | 7.57 | 0.82 | PASS |
| Slopes | 500/500 | [0.51, 0.66] | 5.01 | 0.67 | PASS |
| Stairs | 500/500 | [0.49, 1.30] | 11.94 | 1.28 | PASS |
| Contact-Aware | 500/500 | [0.44, 0.98] | 5.86 | 0.69 | PASS |

---

## 3. Gait Quality Analysis -- ALL TROTTING

| Terrain | Mean Duty Cycle | Diagonal Corr | Swing Phases/Leg | Gait Type |
|---|---|---|---|---|
| Flat | 0.53 | -0.90 | 17-19 | TROTTING |
| Slopes | 0.62 | -0.72 | 22-28 | TROTTING |
| Stairs | 0.50 | -0.84 | 21-30 | TROTTING |
| Contact-Aware | 0.33 | -0.56 | 14-15 | TROTTING |

**Interpretation:**
- All duty cycles well below 0.70 threshold (range: 0.33 to 0.62)
- All diagonal correlations strongly negative (range: -0.56 to -0.90)
- Clear swing phases visible in gait diagrams (see results/gait_diagram_*.png)
- Flat policy has nearly textbook trot (duty=0.53, corr=-0.90)
- Contact-aware has longest swing phases (25 steps avg) -- aggressive foot lifting for stair clearance

---

## 4. Evaluation Results (100 episodes, 256 envs)

| Policy | Mean Reward | Std | Episode Length | Notes |
|---|---|---|---|---|
| Flat | 26.12 | 1.39 | 995.5 | Full episodes, stable |
| Slopes | 25.94 | 3.05 | 980.9 | Slight variance from terrain |
| Stairs | 25.36 | 1.80 | 995.4 | Full episodes on stairs |
| Contact-Aware | -1.86* | 2.73 | 991.0 | Base reward only* |

*Contact-aware eval reward is negative because evaluation uses only the base Isaac Lab reward function, not the additive contact-aware terms. The policy walks well (XY=5.86m, speed=0.69m/s, proper trot gait).

---

## 5. Ablation Results (500 iters x 5 conditions)

### Reward-Based (from training logs)

| Condition | Mean Reward (last 50) | +/- Std | Best | Delta |
|---|---|---|---|---|
| All terms (baseline) | 467.3 | 131.9 | 642.6 | -- |
| No contact timing | 511.4 | 139.9 | 723.2 | +9.4% |
| **No terrain clearance** | **372.8** | **113.6** | **543.5** | **-20.2%** |
| No energy penalty | 498.2 | 148.6 | 700.7 | +6.6% |
| **No foot slip** | **619.7** | **164.7** | **835.5** | **+32.6%** |

### Physical Metrics (from trajectory data)

| Condition | XY (m) | Speed (m/s) | Duty Cycle | Diag Corr | Gait |
|---|---|---|---|---|---|
| All terms | 6.27 | 0.64 | 0.40 | -0.76 | TROT |
| No contact timing | 6.48 | 0.66 | 0.36 | -0.64 | TROT |
| No terrain clearance | 9.43 | 1.11 | 0.41 | -0.78 | TROT |
| No energy penalty | 8.72 | 0.93 | 0.36 | -0.72 | TROT |
| No foot slip | 2.72 | 0.37 | 0.42 | -0.81 | TROT |

### Key Findings

1. **terrain_clearance is the most valuable term (-20.2% reward).** Removing it causes the largest reward drop. However, the no_terrain_clearance policy actually travels FARTHER (9.43m vs 6.27m) -- it optimizes for speed over careful foot placement. This suggests terrain_clearance trades off speed for gait safety.

2. **foot_slip_penalty is the strongest locomotion constraint.** Removing it boosts reward +32.6% but HALVES XY displacement (2.72m vs 6.27m). The policy achieves high reward by minimizing slip penalties rather than moving efficiently. This is the most important constraint for actual locomotion quality.

3. **contact_timing has minimal effect (+9.4%).** Can be removed without meaningful impact.

4. **energy_penalty is negligible at -1e-6 (+6.6%).** Dominated by base dof_torques_l2.

5. **ALL 5 ablation conditions produce trotting gait** (duty < 0.7, negative diagonal correlation). The base Isaac Lab `feet_air_time` reward at weight 0.5 is already sufficient to prevent shuffling -- no additional gait enforcement needed.

---

## 6. Bug Fixes Applied

| # | Bug | File | Fix |
|---|---|---|---|
| 1 | Terrain config mismatch in eval/recording | runner_utils.py | Added `_patch_terrain()` matching training scripts |
| 2 | Energy penalty too aggressive (-1e-4) | config.py, contact_aware_reward.py | Reduced to -1e-6 |
| 3 | Contact sensor key not found | 04_contact_aware/train.py | `scene.sensors["contact_forces"]` |
| 4 | Checkpoint key mismatch (strict=False) | All eval scripts | `OnPolicyRunner.load()` via runner_utils.py |
| 5 | Height validation on non-flat terrain | ground_truth_validate.py | Terrain-relative height checking |

---

## 7. Limitations

1. **Single seed per condition.** No statistical significance. Need 3+ seeds with Mann-Whitney U test.
2. **No sim-to-real validation.** Requires domain randomization.
3. **Contact-aware eval reward misleading.** Base reward doesn't include contact-aware terms, making cross-stage comparison invalid.
4. **Ablation physical metrics are single-trajectory.** Stochastic initial conditions affect XY displacement.
5. **No comparison to Rudin et al. (2022) on identical terrain.** Our stairs speed (1.28 m/s) exceeds their reported ~1.0 m/s, but terrain configurations may differ.

---

## 8. Output Files

### Checkpoints
```
/workspace/checkpoints/{flat,slopes,stairs,contact_aware}/best_model.pt
/workspace/checkpoints/ablation/{all_terms,no_contact_timing,no_terrain_clearance,no_energy_penalty,no_foot_slip}/best_model.pt
```

### Trajectories
```
/workspace/isaac-lab-locomotion/results/trajectories/trajectory_{flat,slopes,stairs,contact_aware}.npz
/workspace/isaac-lab-locomotion/results/trajectories/trajectory_ablation_*.npz
```

### Figures (publication-quality, 150 DPI)
```
results/trajectory_comparison_publication.png  -- 4-panel: XY path, height, speed, duty cycle
results/gait_comparison_publication.png        -- 4-terrain gait diagrams (stance/swing)
results/gait_diagram_{terrain}.png             -- Individual gait diagrams
results/ablation_publication.png               -- 3-panel: XY, speed, duty cycle per ablation
results/ablation_reward_publication.png        -- Reward bar chart with error bars
results/ablation_bar_chart.png                 -- Original ablation chart
```

### Data
```
results/gait_analysis.json                     -- Per-terrain gait metrics
results/ground_truth_validation.json           -- Trajectory validation results
results/evaluation_{terrain}.json              -- 100-episode eval stats
/workspace/results/ablation_results.csv        -- Ablation reward summary
```
