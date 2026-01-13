import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json

from modules.rollout import rollout   # your UPDATED rollout

# ============================================================
# CONFIG
# ============================================================

RUN_ID = "ca446d98c593478e8d58c12b8e0d51c3"
MODEL_NAME = "policy"

TRUE_OMEGA = 0.80
N_PARTICLES = 3000
EPISODE_LEN = 120
HISTORY_SIZE = 30

# ============================================================
# Utility: create validation/run{idx}/
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

    return run_dir, next_id

# ============================================================
# Load trained policy from MLflow
# ============================================================

model_uri = f"runs:/{RUN_ID}/{MODEL_NAME}"
policy = mlflow.pytorch.load_model(model_uri)
policy.eval()

print("Loaded policy:")
print(policy)

# ============================================================
# Extract theta ONCE
# ============================================================

theta = torch.cat([p.detach().flatten() for p in policy.parameters()])

# ============================================================
# Create validation directory
# ============================================================

run_dir, run_idx = get_next_run_dir("validation")
print(f"Saving inference results to: {run_dir}")

# ============================================================
# Run ONE adaptive experiment (ONE rollout)
# ============================================================

info = rollout(
    policy=policy,
    theta=theta,
    TRUE_OMEGA=TRUE_OMEGA,
    N_PARTICLES=N_PARTICLES,
    EPISODE_LEN=EPISODE_LEN,
    HISTORY_SIZE=HISTORY_SIZE,
)



# ============================================================
# Extract trajectories
# ============================================================

t_list    = np.array(info["t_list"])
var_list  = np.array(info["var_list"])
ess_list  = np.array(info["ess_list"])
mean_list = np.array(info["mean_list"])

steps = np.arange(len(t_list))

reward_per_step = np.zeros(len(var_list))
reward_per_step[1:] = var_list[:-1] - var_list[1:]



# ============================================================
# Episode summary
# ============================================================


episode_summary = {
    # --- identification
    "run_id": RUN_ID,
    "model_name": MODEL_NAME,

    # --- experiment configuration
    "true_omega": TRUE_OMEGA,
    "n_particles": N_PARTICLES,
    "episode_len": EPISODE_LEN,
    "history_size": HISTORY_SIZE,

    # --- results
    "initial_variance": float(info["initial_var"]),
    "final_variance": float(info["final_var"]),
    "total_reward": float(info["reward"]),
    "final_ess": float(info["final_ess"]),

    # --- prediction
    "final_posterior_mean": float(mean_list[-1]),
    "final_posterior_variance": float(var_list[-1]),

    # --- trajectory summaries
    "mean_t": float(np.mean(t_list)),
    "min_t": float(np.min(t_list)),
    "max_t": float(np.max(t_list)),
}


print("\n=== Episode summary ===")
for k, v in episode_summary.items():
    print(f"{k:>20s} : {v}")

summary_path = run_dir / "episode_summary.json"
with open(summary_path, "w") as f:
    json.dump(episode_summary, f, indent=2)

print(f"\nEpisode summary saved to {summary_path}")

# ============================================================
# Save raw data
# ============================================================

# np.save(run_dir / "t_list.npy", t_list)
# np.save(run_dir / "var_list.npy", var_list)
# np.save(run_dir / "ess_list.npy", ess_list)
# np.save(run_dir / "mean_list.npy", mean_list)
# np.save(run_dir / "reward_per_step.npy", reward_per_step)

# ============================================================
# PLOTS (saved to disk, not shown)
# ============================================================

# ---- Predicted t
plt.figure(figsize=(6, 4))
plt.plot(steps, t_list, marker="o")
plt.xlabel("Step")
plt.ylabel("Predicted measurement time t")
plt.title("Adaptive policy: predicted t during episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "predicted_t.png")
plt.close()

# ---- Posterior variance
plt.figure(figsize=(6, 4))
plt.plot(steps, var_list, marker="o")
plt.xlabel("Step")
plt.ylabel("Posterior variance")
plt.title("Posterior collapse during adaptive experiment")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_variance.png")
plt.close()

# ---- ESS
plt.figure(figsize=(6, 4))
plt.plot(steps, ess_list, marker="o")
plt.xlabel("Step")
plt.ylabel("ESS")
plt.title("Effective Sample Size during episode")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "ess.png")
plt.close()

# ---- Posterior mean
plt.figure(figsize=(6, 4))
plt.plot(steps, mean_list, marker="o", label="posterior mean")
plt.axhline(TRUE_OMEGA, color="k", linestyle="--", label="true ω")
plt.xlabel("Step")
plt.ylabel("ω")
plt.title("Posterior mean convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "posterior_mean.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(steps, reward_per_step, marker="o")
plt.xlabel("Episode step")
plt.ylabel("Reward (variance reduction)")
plt.title("Reward vs episode step")
plt.grid(True)
plt.tight_layout()
plt.savefig(run_dir / "reward_vs_step.png")
plt.close()

print(f"\nAll figures and data saved to {run_dir}")