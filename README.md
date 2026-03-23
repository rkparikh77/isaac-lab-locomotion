# isaac-lab-locomotion

Quadruped locomotion training experiments using [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and RSL-RL PPO.

---

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA RTX 3090 (24 GB) | A100 SXM4 80 GB |
| CUDA | 12.1 | 12.4 |
| RAM | 32 GB | 64 GB |
| Storage | 20 GB free | 100 GB |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

> Tested on RunPod A100 SXM4 80 GB · Ubuntu 22.04 · CUDA 12.4 · PyTorch 2.4.0.

---

## Quick start — zero to training

```bash
# 1. Install Isaac Lab (≈ 10 min on first run — downloads Isaac Sim)
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 2. Verify installation
python scripts/tutorials/00_sim/create_empty.py --headless

# 3. Install extra dependencies for this repo
pip install wandb

# 4. Clone this repo
cd /workspace
git clone <YOUR_REPO_URL> isaac-lab-locomotion

# 5. Create required directories
mkdir -p /workspace/checkpoints/flat /workspace/results

# 6. Log in to wandb (skip if offline)
wandb login

# 7. Start training in a persistent tmux session
tmux new-session -d -s training \
  "cd /workspace/IsaacLab && \
   python /workspace/isaac-lab-locomotion/experiments/_01_flat_terrain/train.py \
   --headless 2>&1 | tee /workspace/results/flat_terrain_training.log"

# 8. Attach to monitor (Ctrl+B then D to detach without killing)
tmux attach -t training
```

---

## Experiments

### 01 — Flat terrain (AnymalC)

| Setting | Value |
|---------|-------|
| Robot | ANYmal C |
| Terrain | Flat |
| Task | Velocity tracking |
| Algorithm | RSL-RL PPO |
| Environments | 4 096 |
| Max iterations | 1 500 |
| Learning rate | 1e-3 (adaptive) |

Config: [`experiments/_01_flat_terrain/config.py`](experiments/_01_flat_terrain/config.py)

---

## Results

> Replace placeholders after training completes.

| Experiment | Iterations | Mean reward | Velocity error | Notes |
|------------|-----------|-------------|----------------|-------|
| 01 flat terrain | — | — | — | — |

wandb dashboard: _add link after first run_

---

## Project structure

```
isaac-lab-locomotion/
  environments/          Custom env overrides (future)
  experiments/
    _01_flat_terrain/
      config.py          All hyperparameters (dataclass)
      train.py           Training entry point
  scripts/               Utility scripts (eval, export, etc.)
  results/               Log files (git-ignored)
  checkpoints/           Saved models (git-ignored)
  SETUP.md               Full reproducibility guide
```
