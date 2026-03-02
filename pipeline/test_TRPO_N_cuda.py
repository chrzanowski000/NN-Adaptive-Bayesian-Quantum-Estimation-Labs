import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sb3_contrib import TRPO as SB3TRPO
from tqdm import tqdm

from modules.algorithms.seq_montecarlo import (
    resample_liu_west_batched,
)
from modules.rollout_sb3_cuda import rollout_batch

# ============================================================
# CONFIG
# ============================================================

RUN_ID = "043c5ba57da047c8904425713319ca5e"
MODEL_NAME = "trpo_sb3_policy.zip"

N_PARTICLES = 2000
EPISODE_LEN = 125
HISTORY_SIZE = 30
N_OMEGAS = 10000

BATCH_SIZE = 512
ACTION_LOW = 0.1
ACTION_HIGH = 3000.0

DEVICE = "cuda"
SEED = 42
np.random.seed(SEED)
TRUE_OMEGAS_LIST = np.random.uniform(0.0, 1.0, size=N_OMEGAS)


# ============================================================
# Utility
# ============================================================


def get_next_run_dir(base_dir="validation"):
    base = Path(base_dir)
    base.mkdir(exist_ok=True)

    run_ids = []
    for p in base.iterdir():
        if p.is_dir():
            m = re.match(r"run(\d+)", p.name)
            if m:
                run_ids.append(int(m.group(1)))

    next_id = max(run_ids) + 1 if run_ids else 0
    run_dir = base / f"run{next_id}"
    run_dir.mkdir()

    return run_dir


# ============================================================
# Load SB3 model on CUDA
# ============================================================

mlflow.set_tracking_uri("file:///home/chrzanowski/mlflow_tracking")
model_uri = f"runs:/{RUN_ID}/{MODEL_NAME}"
local_path = mlflow.artifacts.download_artifacts(model_uri)

sb3_model = SB3TRPO.load(local_path, device=DEVICE)
sb3_model.policy.eval()

print("Loaded model on:", DEVICE)

run_dir = get_next_run_dir("validation")
print("Saving results to:", run_dir)


# ============================================================
# Batched evaluation
# ============================================================

all_var = []
all_mean = []
all_ess = []
all_t = []

for i in tqdm(range(0, N_OMEGAS, BATCH_SIZE)):
    batch = TRUE_OMEGAS_LIST[i : i + BATCH_SIZE]

    var_traj, mean_traj, ess_traj, t_traj = rollout_batch(
        model=sb3_model,
        omegas=batch,
        resample_fn=resample_liu_west_batched,
        N_PARTICLES=N_PARTICLES,
        EPISODE_LEN=EPISODE_LEN,
        HISTORY_SIZE=HISTORY_SIZE,
        action_low=ACTION_LOW,
        action_high=ACTION_HIGH,
        device=DEVICE,
    )

    all_var.append(var_traj.cpu())
    all_mean.append(mean_traj.cpu())
    all_ess.append(ess_traj.cpu())
    all_t.append(t_traj.cpu())

# Stack: (N_OMEGAS, T)
var_all = torch.cat(all_var, dim=0)
mean_all = torch.cat(all_mean, dim=0)
ess_all = torch.cat(all_ess, dim=0)
t_all = torch.cat(all_t, dim=0)

# Mean across omegas
var_mean = var_all.mean(dim=0).numpy()
mean_mean = mean_all.mean(dim=0).numpy()
ess_mean = ess_all.mean(dim=0).numpy()
t_mean = t_all.mean(dim=0).numpy()

# Single representative trajectory (first omega)
var_single = var_all[0].numpy()
mean_single = mean_all[0].numpy()
ess_single = ess_all[0].numpy()
t_single = t_all[0].numpy()

steps = np.arange(EPISODE_LEN)

# Reward per step (variance reduction)
reward = np.zeros(EPISODE_LEN)
reward[1:] = var_single[:-1] - var_single[1:]


# ============================================================
# Save summary
# ============================================================

summary = {
    "run_id": RUN_ID,
    "n_particles": N_PARTICLES,
    "episode_len": EPISODE_LEN,
    "n_omegas": N_OMEGAS,
    "final_variance_over_omegas": float(var_mean[-1]),
}

with open(run_dir / "episode_summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# PLOTS
# ============================================================

# ---- Predicted t
plt.figure(figsize=(6, 4))
plt.plot(steps, t_single)
plt.xlabel("Step")
plt.ylabel("Predicted measurement time t")
plt.title("Adaptive policy: predicted t")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "predicted_t.png")
plt.close()

# ---- Posterior variance over N omegas (log)
plt.figure(figsize=(6, 4))
plt.plot(steps, var_mean)
plt.ylim(1e-4, 1e-1)
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Posterior variance over N omegas")
plt.title("Posterior collapse (log)")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega_log.png")
plt.close()

# ---- Posterior variance over N omegas
plt.figure(figsize=(6, 4))
plt.plot(steps, var_mean)
plt.xlabel("Step")
plt.ylabel("Posterior variance over N omegas")
plt.title("Posterior collapse")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega.png")
plt.close()

# ---- Posterior variance (single)
plt.figure(figsize=(6, 4))
plt.plot(steps, var_single)
plt.xlabel("Step")
plt.ylabel("Posterior variance")
plt.title("Posterior variance (single omega)")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance.png")
plt.close()

# ---- ESS
plt.figure(figsize=(6, 4))
plt.plot(steps, ess_single)
plt.xlabel("Step")
plt.ylabel("ESS")
plt.title("Effective Sample Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "ess.png")
plt.close()

# ---- Posterior mean
plt.figure(figsize=(6, 4))
plt.plot(steps, mean_single, label="posterior mean")
plt.axhline(TRUE_OMEGAS_LIST[0], linestyle="--", label="true ω")
plt.xlabel("Step")
plt.ylabel("ω")
plt.title("Posterior mean convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_mean.png")
plt.close()

# ---- Reward
plt.figure(figsize=(6, 4))
plt.plot(steps, reward)
plt.xlabel("Step")
plt.ylabel("Reward (variance reduction)")
plt.title("Reward vs step")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "reward_vs_step.png")
plt.close()

print("\nAll plots saved to:", run_dir)
