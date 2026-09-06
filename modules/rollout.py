import numpy as np
import torch

from modules.simulation import measure
from modules.algorithms.seq_montecarlo import init_particles, smc_update_no_resample, ess
from modules.rewards import variance_reduction_reward, posterior_variance
from collections import deque



def rollout(
    policy,
    resample_fn,
    theta,
    TRUE_OMEGA,
    N_PARTICLES,
    EPISODE_LEN,
    HISTORY_SIZE,
    rng=None,
    ):
    # One generator per episode. Previously measure() was called with no rng, so
    # numpy built a fresh default_rng() from OS entropy on every single shot --
    # ~10M generator constructions per CEM run, and RANDOM_SEED had no effect.
    if rng is None:
        rng = np.random.default_rng()

    t_history_len=HISTORY_SIZE
    t_history = deque(maxlen=t_history_len)
    for _ in range(t_history_len):
        t_history.append(0.0)

    idx = 0
    for p in policy.parameters():
        n = p.numel()
        #print("dim in policy parameters: ", n, "\n")
        p.data.copy_(theta[idx:idx+n].view_as(p))
        idx += n

    particles, logw = init_particles(N_PARTICLES)
    initial_var = posterior_variance(particles, logw)



    # --- storage of episode parameters
    t_list = []
    var_list = []
    ess_list = []
    mean_list = []

    for _ in range(EPISODE_LEN):
        w = np.exp(logw - np.max(logw))
        w /= w.sum()

        mean = np.sum(w * particles[:, 0])
        var = np.sum(w * (particles[:, 0] - mean) ** 2)

        mean_list.append(mean)
        var_list.append(var)
        ess_list.append(ess(logw))


        state = torch.tensor(
                            [mean, var] + list(t_history),
                            dtype=torch.float32
                            )
        # No gradients are ever taken through the policy here -- CEM is
        # gradient-free -- so building the autograd graph is pure waste.
        with torch.no_grad():
            t = policy(state).item()
        t_list.append(t)
        t_history.append(t)
        d = measure(TRUE_OMEGA, t, rng=rng)

        particles, logw = smc_update_no_resample(
            particles, logw, d, t
        )
        if ess(logw) < 0.75 * len(logw):
            particles, logw = resample_fn(particles, logw)
            

    final_var = posterior_variance(particles, logw)


    info = {
        "initial_var": initial_var,
        "final_var": final_var,
        "reward": initial_var - final_var,
        "final_ess": ess(logw),
        "logw": logw,
        "mean_t": float(np.mean(t_list)),
        "particles": particles,

        # ---- trajectories per episode
        "t_list": t_list,
        "var_list": var_list,
        "ess_list": ess_list,
        "mean_list": mean_list,
    }

    return info