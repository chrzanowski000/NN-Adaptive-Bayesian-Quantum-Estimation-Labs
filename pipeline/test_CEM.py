import mlflow
import mlflow.pytorch
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

from modules.algorithms.CEM import CEM
from models.nn import TimePolicy
from modules.algorithms.seq_montecarlo import build_model, normalize
from modules.rollout import rollout
from modules.simulation import FIXED_T2



# ================= CONFIG =================

N_PARTICLES = 3000
EPISODE_LEN = 100
TRUE_OMEGA = 0.7

CEM_POP = 100
CEM_ELITE_FRAC = 0.1
CEM_INIT_STD = 1.0
CEM_GENERATIONS = 100
WINDOW_SIZE = 32 #size od time array passed to networks + mu and sigma so 30+2

# ==========================================

mlflow.set_experiment("cem_qubit_omega_only")

model, logp_fn = build_model()

### temporary curvature check
# omega = np.linspace(0,1,500)
# t=1
# p0 = np.exp(-t/FIXED_T2)*np.cos(omega*t/2)**2 + (1-np.exp(-t/FIXED_T2))/2
# plt.plot(omega, p0)
# plt.savefig("artifacts/model.png")


with mlflow.start_run():

    mlflow.log_params({
        "N_PARTICLES": N_PARTICLES,
        "EPISODE_LEN": EPISODE_LEN,
        "FIXED_T2": FIXED_T2,
        "TRUE_OMEGA": TRUE_OMEGA,
        "CEM_POP": CEM_POP,
        "CEM_ELITE_FRAC": CEM_ELITE_FRAC,
        "CEM_INIT_STD": CEM_INIT_STD,
        "WINDOW_SIZE": WINDOW_SIZE
    })

    cem = CEM(TimePolicy, CEM_POP, CEM_ELITE_FRAC, CEM_INIT_STD)

    for gen in range(CEM_GENERATIONS):
        rewards, stats = cem.step(
            rollout_fn=lambda theta: rollout(
                                            theta,
                                            model,
                                            logp_fn,
                                            TRUE_OMEGA,
                                            N_PARTICLES,
                                            EPISODE_LEN,
                                            return_particles=False,#remove
            ),
            debug=True
        )

        # ---- stats ----
        mean_r = np.mean(rewards)
        max_r = np.max(rewards)

        mean_init_var = np.mean([s["initial_var"] for s in stats])
        mean_final_var = np.mean([s["final_var"] for s in stats])
        mean_ess = np.mean([s["final_ess"] for s in stats])
        mean_t = np.mean([s["mean_t"] for s in stats])

        # ---- metrics ----
        mlflow.log_metric("mean_reward", mean_r, step=gen)
        mlflow.log_metric("max_reward", max_r, step=gen)
        mlflow.log_metric("sigma_mean", cem.sigma.mean().item(), step=gen)
        mlflow.log_metric("mu_norm", torch.norm(cem.mu).item(), step=gen)
        mlflow.log_metric("mean_init_var", mean_init_var, step=gen)
        mlflow.log_metric("mean_final_var", mean_final_var, step=gen)
        mlflow.log_metric("mean_ess", mean_ess, step=gen)
        mlflow.log_metric("mean_t", mean_t, step=gen)

        # ---- posterior histogram (best policy) ---- #perhaps should be removed i dont know how it interfers exactly
        info = rollout(
            cem.mu,
            model,
            logp_fn,
            TRUE_OMEGA,
            N_PARTICLES,
            EPISODE_LEN,
            return_particles=True, #remove
     
        )
        ### get metrics fror histograms
        logw = info["logw"]
        particles = info["particles"]
        ###
        #PLOTS
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
        plt.savefig(os.path.join("artifacts",fname))
        plt.close()

        mlflow.log_artifact(os.path.join("artifacts",fname))

        print(f"gen {gen:02d} | mean R = {mean_r:.3e} | max R = {max_r:.3e}")

    # ---- save final policy ---- #i thing it does something wrong #to correct later
    final_policy = TimePolicy()
    idx = 0
    for p in final_policy.parameters():
        n = p.numel()
        p.data.copy_(cem.mu[idx:idx+n].view_as(p))
        idx += n

    mlflow.pytorch.log_model(final_policy, "final_policy")