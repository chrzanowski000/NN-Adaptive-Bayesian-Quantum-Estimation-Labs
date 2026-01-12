import mlflow
import mlflow.pytorch
import torch
import matplotlib.pyplot as plt

from modules.algorithms.CEM import CEM
from models.nn import TimePolicy
from modules.algorithms.seq_montecarlo import build_model, normalize
from modules.rollout import rollout
from modules.simulation import FIXED_T2

# ================= CONFIG =================

N_PARTICLES = 3000
EPISODE_LEN = 100
TRUE_OMEGA = 0.7

CEM_POP = 40
CEM_ELITE_FRAC = 0.2
CEM_INIT_STD = 1.0
CEM_GENERATIONS = 20

# ==========================================

mlflow.set_experiment("cem_qubit_omega_only")

model, logp_fn = build_model()

with mlflow.start_run():

    mlflow.log_params({
        "N_PARTICLES": N_PARTICLES,
        "EPISODE_LEN": EPISODE_LEN,
        "FIXED_T2": FIXED_T2,
        "TRUE_OMEGA": TRUE_OMEGA,
        "CEM_POP": CEM_POP,
        "CEM_ELITE_FRAC": CEM_ELITE_FRAC,
        "CEM_INIT_STD": CEM_INIT_STD,
    })

    cem = CEM(TimePolicy, CEM_POP, CEM_ELITE_FRAC, CEM_INIT_STD)

    for gen in range(CEM_GENERATIONS):
        mean_r, max_r = cem.step(
            lambda theta: rollout(
                theta,
                model,
                logp_fn,
                TRUE_OMEGA,
                N_PARTICLES,
                EPISODE_LEN,
            )
        )

        # ---- metrics ----
        mlflow.log_metric("mean_reward", mean_r, step=gen)
        mlflow.log_metric("max_reward", max_r, step=gen)
        mlflow.log_metric("sigma_mean", cem.sigma.mean().item(), step=gen)
        mlflow.log_metric("mu_norm", torch.norm(cem.mu).item(), step=gen)

        # ---- posterior histogram (best policy) ----
        _, particles, logw = rollout(
            cem.mu,
            model,
            logp_fn,
            TRUE_OMEGA,
            N_PARTICLES,
            EPISODE_LEN,
            return_particles=True,
        )

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
        plt.savefig(fname)
        plt.close()

        mlflow.log_artifact(fname)

        print(f"gen {gen:02d} | mean R = {mean_r:.3e} | max R = {max_r:.3e}")

    # ---- save final policy ----
    final_policy = TimePolicy()
    idx = 0
    for p in final_policy.parameters():
        n = p.numel()
        p.data.copy_(cem.mu[idx:idx+n].view_as(p))
        idx += n

    mlflow.pytorch.log_model(final_policy, "final_policy")