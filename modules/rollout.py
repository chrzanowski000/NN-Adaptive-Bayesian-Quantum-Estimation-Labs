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

    t_history = deque(maxlen=30)
    for _ in range(30):
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

        """
        na wejściu do sieci wciskamy na pewno przeszłe czasy okno 30 trzeba je wygenrowac najpierw losowo dopiero wtedy wrzucać
        kiedy generujemy kolejny czas dodajemy do okna
        """
        # state = torch.tensor([mean, var], dtype=torch.float32) 
        state = torch.tensor(
                            [mean, var] + list(t_history),
                            dtype=torch.float32
                            )
        #tutaj chyba wciskamy rozkład a nie coś innego rozkład powinien być użyty do generacji parametrów sieci
        #trzeba dodać do parametrów okno czasów i będzie git 
        t = policy(state).item()
        t_history.append(t)

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

    return reward#, initial_var, final_var