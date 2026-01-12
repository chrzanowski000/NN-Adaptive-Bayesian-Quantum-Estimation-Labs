import numpy as np

def normalize_log_weights(logw):
    maxlogw = np.max(logw)
    w = np.exp(logw - maxlogw)
    w /= w.sum()
    return np.log(w), w

def posterior_cov_trace(particles, logw):
    _, w = normalize_log_weights(logw)
    cov = np.cov(particles.T, aweights=w)
    return np.trace(cov)


def variance_reduction_reward(
    particles_prev, logw_prev,
    particles_post, logw_post
):
    return (
        posterior_cov_trace(particles_prev, logw_prev)
        - posterior_cov_trace(particles_post, logw_post)
    )
