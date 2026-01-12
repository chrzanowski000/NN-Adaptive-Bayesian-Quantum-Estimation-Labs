import numpy as np
import torch

from modules.simulation import measure
from modules.algorithms.seq_montecarlo import init_particles, smc_update_no_resample, ess, resample
from modules.rewards import variance_reduction_reward, posterior_variance
from models.nn import TimePolicy

def rollout(
    theta,
    model,
    logp_fn,
    TRUE_OMEGA,
    N_PARTICLES,
    EPISODE_LEN,
    return_particles=False,
):
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

        state = torch.tensor([mean, var], dtype=torch.float32)

        t = policy(state).item()
        d = measure(TRUE_OMEGA, t)

        particles, logw = smc_update_no_resample(
            particles, logw, d, t, model, logp_fn
        )

        if ess(logw) < 0.8 * len(logw):
            particles, logw = resample(particles, logw)

    final_var = posterior_variance(particles, logw)
    reward = initial_var - final_var

    if return_particles:
        return reward, particles, logw

    return reward