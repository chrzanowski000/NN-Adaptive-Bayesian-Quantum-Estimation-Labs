import numpy as np
from modules.algorithms.seq_montecarlo import normalize

def posterior_variance(particles, logw):
    w = normalize(logw)
    mean = np.sum(w * particles[:, 0])
    var = np.sum(w * (particles[:, 0] - mean)**2)
    return var


def variance_reduction_reward(p_prev, w_prev, p_post, w_post):
    return posterior_variance(p_prev, w_prev) - posterior_variance(p_post, w_post)
