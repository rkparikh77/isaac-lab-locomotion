# SETUP.md — Full Reproducibility Guide

Tested on: **RunPod A100 SXM4 80 GB · Ubuntu 22.04 · CUDA 12.4 · PyTorch 2.4.0**

Follow every step in order on a fresh RunPod instance.

---

## Prerequisites (RunPod instance)

The following are pre-installed on a standard RunPod PyTorch 2.4.0 / CUDA 12.4 template:
- Python 3.10
- PyTorch 2.4.0
- CUDA 12.4 + cuDNN
- git, curl, wget, tmux

---

## Step 1 — Install Isaac Lab

```bash
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

The installer:
1. Detects a headless / container environment automatically.
2. Downloads NVIDIA Isaac Sim via pip (≈ 15–20 min on a RunPod instance).
3. Installs all Isaac Lab Python extensions in development mode.
4. Downloads the default assets.

### Verify installation

```bash
cd /workspace/IsaacLab
python scripts/tutorials/00_sim/create_empty.py --headless
```

Expected output ends with something like:
```
[INFO]: Simulation Loop is Terminated.
```

If the command fails, common causes and fixes:
- **`ModuleNotFoundError: No module named 'isaacsim'`** — the pip install of Isaac Sim
  did not complete. Re-run `./isaaclab.sh --install`.
- **CUDA out-of-memory during verify** — safe to ignore if the tutorial exits cleanly;
  the tutorial uses very few resources.
- **`AppLauncher` import error** — make sure you are running from inside the
  `/workspace/IsaacLab/` directory where Isaac Lab's Python path is configured.

---

## Step 2 — Create the project repository

```bash
cd /workspace
mkdir -p isaac-lab-locomotion/{environments,experiments/_01_flat_terrain,scripts,results,checkpoints/flat}
cd isaac-lab-locomotion
git init
git remote add origin <YOUR_REPO_URL>   # skip if not using remote git
```

### Copy project files

If you have the files on your local Mac, transfer them:

```bash
# From your Mac:
scp -r ~/projects/isaac-lab-locomotion/* root@<RUNPOD_IP>:/workspace/isaac-lab-locomotion/
```

Or create `config.py` and `train.py` directly on the RunPod by cloning your repo:

```bash
git clone <YOUR_REPO_URL> /workspace/isaac-lab-locomotion
```

---

## Step 3 — Install extra Python dependencies

```bash
pip install wandb
```

Optional (for linting / development):
```bash
pip install black flake8
```

---

## Step 4 — Configure wandb

```bash
wandb login   # paste your API key from https://wandb.ai/authorize
```

To run without wandb (offline):
```bash
export WANDB_MODE=offline
```

---

## Step 5 — Create required runtime directories

```bash
mkdir -p /workspace/checkpoints/flat
mkdir -p /workspace/results
```

---

## Step 5b — Verify the training script parses cleanly

```bash
cd /workspace/IsaacLab
python /workspace/isaac-lab-locomotion/experiments/_01_flat_terrain/train.py --help
```

---

## Step 6 — Start training in tmux

```bash
tmux new-session -d -s training \
  "cd /workspace/IsaacLab && \
   python /workspace/isaac-lab-locomotion/experiments/_01_flat_terrain/train.py \
   --headless 2>&1 | tee /workspace/results/flat_terrain_training.log"
```

### Verify the session is running

```bash
tmux list-sessions            # should show: training: 1 windows
tmux attach -t training       # attach to watch live output
# Press Ctrl+B then D to detach without killing the session
```

### Check the log file

```bash
tail -f /workspace/results/flat_terrain_training.log
```

---

## Monitoring

### wandb

Training metrics (episode_reward, episode_length, velocity_tracking_error) are
logged every 10 iterations. Open your wandb project to see live curves:

```
https://wandb.ai/<YOUR_ENTITY>/motionmd-locomotion
```

### GPU utilisation

```bash
watch -n 2 nvidia-smi
```

### Checkpoints

```bash
ls -lh /workspace/checkpoints/flat/
```

Saved every 100 iterations as `model_00100.pt`, `model_00200.pt`, etc.
`best_model.pt` is written if early stopping triggers (mean reward > 8.0 for 50 consecutive iters).

---

## Stopping and resuming

### Stop training

```bash
tmux kill-session -t training
```

### Resume from a checkpoint

Edit `config.py` or pass overrides, then re-launch. The RSL-RL runner supports
`--resume` via its own config (`resume=True`, `load_run`, `load_checkpoint` fields
on `RslRlOnPolicyRunnerCfg`). Uncomment those fields in `_build_agent_cfg()` in
`train.py` and point `load_checkpoint` to the desired `.pt` file.

---

## Clean up (free disk space)

```bash
# Remove all checkpoints except the best
find /workspace/checkpoints/flat -name "model_*.pt" -delete

# Remove Isaac Sim download cache (large — only if disk is tight)
# pip cache purge
```

---

## Dependency versions (pinned at test time)

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.4.0 |
| CUDA | 12.4 |
| Isaac Lab | main (cloned 2026-03-22) |
| rsl-rl | ≥ 3.0.1 |
| isaaclab-rl | installed via ./isaaclab.sh |
| wandb | ≥ 0.16 |
