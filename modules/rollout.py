import numpy as np
import torch

from modules.simulation import measure
from modules.algorithms.seq_montecarlo import init_particles, smc_update_no_resample, ess, resample
from modules.rewards import variance_reduction_reward, posterior_variance
from models.nn import TimePolicy
from collections import deque



def rollout(
    theta,
    model,
    logp_fn,
    TRUE_OMEGA,
    N_PARTICLES,
    EPISODE_LEN,
    return_particles=False,
    ):

    t_history_len=5
    t_history = deque(maxlen=t_history_len)
    for _ in range(t_history_len):
        t_history.append(0.0) #change to uniform
    policy = TimePolicy()

    idx = 0
    for p in policy.parameters():
        n = p.numel()
        p.data.copy_(theta[idx:idx+n].view_as(p))
        idx += n

    particles, logw = init_particles(N_PARTICLES)
    initial_var = posterior_variance(particles, logw)

    for _ in range(EPISODE_LEN):
        w = np.exp(logw - np.max(logw))
        w /= w.sum()

        mean = np.sum(w * particles[:, 0])
        var = np.sum(w * (particles[:, 0] - mean) ** 2)


        state = torch.tensor(
                            [mean, var] + list(t_history),
                            dtype=torch.float32
                            )
        t = policy(state).item()
        t_history.append(t)
        d = measure(TRUE_OMEGA, t)
        #print(d)
        # print(particles, logw, d, t, model, logp_fn)

        particles, logw = smc_update_no_resample( #there is a problem SMC is not updating
            particles, logw, d, t, model, logp_fn
        )
        #print(particles)
        #print(logw)

        if ess(logw) < 0.8 * len(logw):
            particles, logw = resample(particles, logw)

    final_var = posterior_variance(particles, logw)


    info = {
        "initial_var": initial_var,
        "final_var": final_var,
        "reward": initial_var - final_var,
        "final_ess": ess(logw),
        "logw": logw,
        "mean_t": float(np.mean(t_history)),
        "particles": particles
    }

    return info