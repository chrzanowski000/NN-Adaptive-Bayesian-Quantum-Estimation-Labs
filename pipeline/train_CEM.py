import os
import platform

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from tqdm import tqdm

import models.nn
from modules.algorithms.CEM import CEM
from modules.algorithms.seq_montecarlo import normalize, resample_liu_west
from modules.rollout import rollout
from modules.simulation import FIXED_T2
from utils.git_utils.git import get_git_branch, get_git_commit, git_is_dirty

def _env(name, default, cast=int):
    """Allow every headline hyperparameter to be overridden from the shell.

    Lets the same script serve a 30-second smoke test and a multi-hour run
    without editing constants:  N_PARTICLES=100 EPISODE_LEN=5 python -m ...
    """
    raw = os.environ.get(name)
    return default if raw is None else cast(raw)

# ================= CONFIG =================
POLICY = models.nn.TimePolicy_Fiderer_16  # choose network
RESAMPLE_FN = resample_liu_west
# RESAMPLE_FN = resample
# POLICY = models.nn.TimePolicy_1

N_PARTICLES = _env("N_PARTICLES", 2000)
EPISODE_LEN = _env("EPISODE_LEN", 100)
CEM_POP = _env("CEM_POP", 1000)
CEM_ELITE_FRAC = _env("CEM_ELITE_FRAC", 0.1, float)
CEM_INIT_STD = _env("CEM_INIT_STD", 1.0, float)
CEM_GENERATIONS = _env("CEM_GENERATIONS", 100)
HISTORY_SIZE = _env("HISTORY_SIZE", 30)  # input_dim = HISTORY_SIZE + 2
RANDOM_SEED = _env("RANDOM_SEED", 50)

CHECKPOINT_EVERY_GENS = _env("CHECKPOINT_EVERY_GENS", 5)
CHECKPOINT_DIR = os.path.join("artifacts", "checkpoints")

np.random.seed(RANDOM_SEED)  # seed for omegas generation
torch.manual_seed(RANDOM_SEED)  # CEM population sampling
EPISODE_RNG = np.random.default_rng(RANDOM_SEED)  # measurement outcomes
TRUE_OMEGAS_LIST = np.random.uniform(
    0.0, 1.0, size=CEM_GENERATIONS
)  # generate list of random omegas


# ==========================================

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + os.path.abspath("mlflow.db"),
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# CEM used to log into the experiment literally named ".../ trpo", which made
# the two methods indistinguishable in the UI.
mlflow.set_experiment("fiderer / omega_estimation / cem")

project_tags = {"project": "fiderer", "algo": "cem", "env": "sequential_montecarlo"}

os.makedirs("artifacts", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log_system_static():
    mlflow.set_tags(
        {
            "system.os": platform.system(),
            "system.os_version": platform.version(),
            "system.python_version": platform.python_version(),
            "system.cpu_model": platform.processor(),
            "system.cpu_cores": os.cpu_count(),
            "system.device": "cpu" if not torch.cuda.is_available() else "cuda",
            "system.torch_version": torch.__version__,
        }
    )

    if torch.cuda.is_available():
        mlflow.set_tags(
            {
                "system.gpu_name": torch.cuda.get_device_name(0),
                "system.cuda_version": torch.version.cuda,
            }
        )


def save_checkpoint(cem, tag):
    """Persist the current CEM mean as a loadable policy state_dict."""
    model = cem.policy_model
    idx = 0
    for prm in model.parameters():
        n = prm.numel()
        prm.data.copy_(cem.mu[idx : idx + n].view_as(prm))
        idx += n
    path = os.path.join(CHECKPOINT_DIR, f"cem_{tag}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mu": cem.mu,
            "sigma": cem.sigma,
            "policy_name": POLICY.__name__,
            "history_size": HISTORY_SIZE,
        },
        path,
    )
    latest = os.path.join("artifacts", "cem_policy_latest.pt")
    torch.save(torch.load(path, weights_only=True), latest)
    try:
        mlflow.log_artifact(path, artifact_path="checkpoints")
        mlflow.log_artifact(latest)
    except Exception as exc:
        print(f"[checkpoint] mlflow upload failed: {exc}")
    return path


with mlflow.start_run():
    mlflow.set_tags(project_tags)
    log_system_static()

    mlflow.log_params(
        {
            "N_PARTICLES": N_PARTICLES,
            "EPISODE_LEN": EPISODE_LEN,
            "FIXED_T2": FIXED_T2,
            "TRUE_OMEGAS_LIST": TRUE_OMEGAS_LIST,
            "CEM_POP": CEM_POP,
            "CEM_ELITE_FRAC": CEM_ELITE_FRAC,
            "CEM_INIT_STD": CEM_INIT_STD,
            "HISOTRY_SIZE": HISTORY_SIZE,
            "policy_name": POLICY.__name__,
            "resample_function": RESAMPLE_FN.__name__,
            "RANDOM_SEED": RANDOM_SEED,
            "CEM_GENERATIONS": CEM_GENERATIONS,
            "CHECKPOINT_EVERY_GENS": CHECKPOINT_EVERY_GENS,
            # --- git ---
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "git_dirty": git_is_dirty(),
        }
    )

    cem = CEM(POLICY, CEM_POP, CEM_ELITE_FRAC, CEM_INIT_STD, HISTORY_SIZE)

    for gen in tqdm(range(CEM_GENERATIONS)):
        TRUE_OMEGA = TRUE_OMEGAS_LIST[gen]
        # print("TRUE OMEGA: ",TRUE_OMEGA)
        rewards, stats = cem.step(
            rollout_fn=lambda theta: rollout(
                cem.policy_model,
                RESAMPLE_FN,
                theta,
                TRUE_OMEGA,
                N_PARTICLES,
                EPISODE_LEN,
                HISTORY_SIZE,
                rng=EPISODE_RNG,
            ),
            debug=True,
        )

        # ---- stats ----
        mean_r = np.mean(rewards)
        max_r = np.max(rewards)

        mean_init_var = np.mean([s["initial_var"] for s in stats])
        mean_final_var = np.mean([s["final_var"] for s in stats])
        mean_ess = np.mean([s["final_ess"] for s in stats])
        mean_t = np.mean([s["mean_t"] for s in stats])

        # ---- metrics ----
        mlflow.log_metrics(
            {
                "mean_reward": float(mean_r),
                "max_reward": float(max_r),
                "sigma_mean": cem.sigma.mean().item(),
                "mu_norm": torch.norm(cem.mu).item(),
                "mean_init_var": float(mean_init_var),
                "mean_final_var": float(mean_final_var),
                "mean_ess": float(mean_ess),
                "mean_t": float(mean_t),
                "true_omega": float(TRUE_OMEGA),
            },
            step=gen,
        )

        if gen % 5 == 0:
            ###----
            # TEST RUN
            ###----

            # ---- posterior histogram (best policy) ---- #perhaps should be removed i dont know how it interfers exactly
            info = rollout(
                cem.policy_model,
                RESAMPLE_FN,
                cem.mu,
                TRUE_OMEGA,
                N_PARTICLES,
                EPISODE_LEN,
                HISTORY_SIZE,
                rng=EPISODE_RNG,
            )

            ### get metrics fror histograms
            logw = info["logw"]
            particles = info["particles"]
            ###
            # PLOTS
            ###
            w = normalize(logw)

            plt.figure(figsize=(6, 4))
            plt.hist(
                particles[:, 0],
                weights=w,
                bins=50,
                density=True,
            )
            plt.axvline(TRUE_OMEGA, color="red", linestyle="--", label="true ω")
            plt.xlabel("ω")
            plt.ylabel("posterior density")
            plt.title(f"Posterior (gen {gen})")
            plt.legend()

            fname = f"posterior_gen_{gen:03d}.png"
            plt.savefig(os.path.join("artifacts", fname))
            plt.close()

            mlflow.log_artifact(os.path.join("artifacts", fname))

        if gen % CHECKPOINT_EVERY_GENS == 0 or gen == CEM_GENERATIONS - 1:
            save_checkpoint(cem, f"gen_{gen:04d}")

            # print(f"gen {gen:02d} | mean R = {mean_r:.3e} | max R = {max_r:.3e}")

    # ---- save final policy ----
    # save_checkpoint copies cem.mu into the model before serialising, so the
    # saved weights are the CEM mean, not the last sampled population member.
    save_checkpoint(cem, "final")
    final_policy = cem.policy_model

    mlflow.pytorch.log_model(final_policy, name="policy")
    print("policy model saved")
