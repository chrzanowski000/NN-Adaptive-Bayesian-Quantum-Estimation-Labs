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
# Batched evaluation (MEAN over omegas only)
# ============================================================

start_time = time.time()

var_sum = torch.zeros(EPISODE_LEN)
mean_sum = torch.zeros(EPISODE_LEN)
ess_sum = torch.zeros(EPISODE_LEN)
t_sum = torch.zeros(EPISODE_LEN)

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

    var_sum += var_traj.sum(dim=0).cpu()
    mean_sum += mean_traj.sum(dim=0).cpu()
    ess_sum += ess_traj.sum(dim=0).cpu()
    t_sum += t_traj.sum(dim=0).cpu()

# Compute ensemble means
var_list = (var_sum / N_OMEGAS).numpy()
mean_list = (mean_sum / N_OMEGAS).numpy()
ess_list = (ess_sum / N_OMEGAS).numpy()
t_list = (t_sum / N_OMEGAS).numpy()

steps = np.arange(EPISODE_LEN)

reward_per_step = np.zeros(EPISODE_LEN)
reward_per_step[1:] = var_list[:-1] - var_list[1:]

elapsed_time = time.time() - start_time

# ============================================================
# Episode summary
# ============================================================

episode_summary = {
    "run_id": RUN_ID,
    "model_name": MODEL_NAME,
    "resampling_model": resample_liu_west_batched.__name__,
    "device": DEVICE,
    "batch_size": BATCH_SIZE,
    "gpu_name": torch.cuda.get_device_name(0),
    "runtime_seconds": elapsed_time,
    "n_particles": N_PARTICLES,
    "episode_len": EPISODE_LEN,
    "history_size": HISTORY_SIZE,
    "number_of_omegas": N_OMEGAS,
    "initial_variance": float(var_list[0]),
    "final_variance": float(var_list[-1]),
    "total_reward": float(var_list[0] - var_list[-1]),
    "final_ess": float(ess_list[-1]),
    "final_posterior_mean": float(mean_list[-1]),
    "final_posterior_variance": float(var_list[-1]),
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
# PLOTS (Line only)
# ============================================================

plt.figure(figsize=(6, 4))
plt.plot(steps, t_list)
plt.xlabel("Step")
plt.ylabel("Predicted measurement time t")
plt.title("Adaptive policy: predicted t during episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "predicted_t.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, var_list)
plt.yscale("log")
plt.ylim(1e-4, 1e-1)
plt.xlabel("Step")
plt.ylabel(f"Posterior variance over {N_OMEGAS} omegas")
plt.title("Posterior collapse (log)")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega_log.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, var_list)
plt.xlabel("Step")
plt.ylabel(f"Posterior variance over {N_OMEGAS} omegas")
plt.title("Posterior collapse")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance_over_omega.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, ess_list)
plt.xlabel("Step")
plt.ylabel("ESS")
plt.title("Effective Sample Size")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "ess.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, mean_list)
plt.xlabel("Step")
plt.ylabel("ω")
plt.title("Posterior mean convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_mean.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, reward_per_step)
plt.xlabel("Episode step")
plt.ylabel("Reward (variance reduction)")
plt.title("Reward vs step")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "reward_vs_step.png")
plt.close()

print("\nAll figures saved to:", run_dir)
