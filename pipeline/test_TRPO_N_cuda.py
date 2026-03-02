import json
import re
import time
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
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

rng = np.random.default_rng(SEED)
TRUE_OMEGAS_LIST = rng.uniform(0.0, 1.0, size=N_OMEGAS)

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
# Load SB3 model
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
# Batched evaluation over N omegas
# ============================================================

start_time = time.time()

var_list_N = torch.zeros(EPISODE_LEN, 1)

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

    var_list_N = torch.cat(
        [var_list_N, var_traj.cpu().T],
        dim=1,
    )

# remove dummy column
var_list_N = var_list_N[:, 1:]
var_list_N_mean = var_list_N.mean(dim=1).numpy()

# ============================================================
# Single omega rollout (for identical summary + plots)
# ============================================================

TRUE_OMEGA = TRUE_OMEGAS_LIST[0]

var_traj, mean_traj, ess_traj, t_traj = rollout_batch(
    model=sb3_model,
    omegas=[TRUE_OMEGA],
    resample_fn=resample_liu_west_batched,
    N_PARTICLES=N_PARTICLES,
    EPISODE_LEN=EPISODE_LEN,
    HISTORY_SIZE=HISTORY_SIZE,
    action_low=ACTION_LOW,
    action_high=ACTION_HIGH,
    device=DEVICE,
)

var_list = var_traj[0].cpu().numpy()
mean_list = mean_traj[0].cpu().numpy()
ess_list = ess_traj[0].cpu().numpy()
t_list = t_traj[0].cpu().numpy()

steps = np.arange(EPISODE_LEN)

reward_per_step = np.zeros(EPISODE_LEN)
reward_per_step[1:] = var_list[:-1] - var_list[1:]

elapsed_time = time.time() - start_time

# ============================================================
# Episode summary (IDENTICAL STRUCTURE + device + batch size)
# ============================================================

episode_summary = {
    # --- identification
    "run_id": RUN_ID,
    "model_name": MODEL_NAME,
    "resampling_model": resample_liu_west_batched.__name__,
    # --- hardware
    "device": DEVICE,
    "batch_size": BATCH_SIZE,
    "gpu_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    "runtime_seconds": elapsed_time,
    # --- experiment configuration
    "true_omega": float(TRUE_OMEGA),
    "n_particles": N_PARTICLES,
    "episode_len": EPISODE_LEN,
    "history_size": HISTORY_SIZE,
    "number_of_omegas": N_OMEGAS,
    # --- single trajectory results
    "initial_variance": float(var_list[0]),
    "final_variance": float(var_list[-1]),
    "total_reward": float(var_list[0] - var_list[-1]),
    "final_ess": float(ess_list[-1]),
    "final_posterior_mean": float(mean_list[-1]),
    "final_posterior_variance": float(var_list[-1]),
    # --- multi omega statistic
    "final_variance_over_N_omegas": float(var_list_N_mean[-1]),
    # --- trajectory summaries
    "mean_t": float(np.mean(t_list)),
    "min_t": float(np.min(t_list)),
    "max_t": float(np.max(t_list)),
}

print("\n=== Episode summary ===")
for k, v in episode_summary.items():
    print(f"{k:>30s} : {v}")

summary_path = run_dir / "episode_summary.json"
with open(summary_path, "w") as f:
    json.dump(episode_summary, f, indent=2)

print("\nEpisode summary saved to", summary_path)

# ============================================================
# PLOTS (IDENTICAL TO ORIGINAL CPU VERSION)
# ============================================================

# Predicted t
plt.figure(figsize=(6, 4))
plt.plot(steps, t_list)
plt.xlabel("Step")
plt.ylabel("Predicted measurement time t")
plt.title("Adaptive policy: predicted t during episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "predicted_t.png")
plt.close()

# Posterior variance over N omegas (log)
plt.figure(figsize=(6, 4))
plt.plot(steps, var_list_N_mean)
plt.yscale("log")
plt.ylim(1e-4, 1e-1)
plt.xlabel("Step")
plt.ylabel(f"Posterior variance over {N_OMEGAS} omegas")
plt.title("Posterior collapse during experiment")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega_log.png")
plt.close()

# Posterior variance over N omegas
plt.figure(figsize=(6, 4))
plt.plot(steps, var_list_N_mean)
plt.xlabel("Step")
plt.ylabel(f"Posterior variance over {N_OMEGAS} omegas")
plt.title("Posterior collapse during experiment")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega.png")
plt.close()

# Posterior variance (single)
plt.figure(figsize=(6, 4))
plt.plot(steps, var_list)
plt.xlabel("Step")
plt.ylabel("Posterior variance")
plt.title("Posterior collapse during experiment")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance.png")
plt.close()

# ESS
plt.figure(figsize=(6, 4))
plt.plot(steps, ess_list)
plt.xlabel("Step")
plt.ylabel("ESS")
plt.title("Effective Sample Size during episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "ess.png")
plt.close()

# Posterior mean
plt.figure(figsize=(6, 4))
plt.plot(steps, mean_list, label="posterior mean")
plt.axhline(TRUE_OMEGA, linestyle="--", label="true ω")
plt.xlabel("Step")
plt.ylabel("ω")
plt.title("Posterior mean convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_mean.png")
plt.close()

# Reward
plt.figure(figsize=(6, 4))
plt.plot(steps, reward_per_step)
plt.xlabel("Episode step")
plt.ylabel("Reward (variance reduction)")
plt.title("Reward vs episode step")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "reward_vs_step.png")
plt.close()

print("\nAll figures saved to:", run_dir)
