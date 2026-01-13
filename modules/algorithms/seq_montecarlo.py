import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp
from modules.simulation import FIXED_T2



def init_particles(N):
    particles = np.random.uniform(0, 1, size=(N, 1))  # omega only
    logw = -np.log(N) * np.ones(N)
    return particles, logw


def normalize(logw):
    w = np.exp(logw - logsumexp(logw))
    return w




def ess(logw):
    w = normalize(logw)
    return 1.0 / np.sum(w**2)


def resample(particles, logw):
    w = normalize(logw)
    idx = np.random.choice(len(w), size=len(w), p=w)
    particles = particles[idx]
    logw = -np.log(len(w)) * np.ones(len(w))
    return particles, logw


# --------------------------------------------------
# One SMC update
# --------------------------------------------------

def smc_step(particles, logw, d, t, model, logp_fn):
    with model:
        pm.set_data({
            "omega": particles[:, 0],
            "t": float(t)
        })
        y = np.full(len(particles), d)
        logp = logp_fn({"y": y})

    logp = np.maximum(logp, -1e6)      # likelihood floor
    #print("logp std =", np.std(logp))
    logw = logw + logp
    logw = logw - logsumexp(logw)      # normalize
    print(ess(logw), len(logw))
    if ess(logw) < 0.1 * len(logw):
        print("resampled")
        particles, logw = resample(particles, logw)

    return particles, logw



def smc_update_no_resample(particles, logw, d, t):
    omega = particles[:, 0]

    p0 = (
        np.exp(-t / FIXED_T2)
        * np.cos(omega * t / 2)**2
        + (1 - np.exp(-t / FIXED_T2)) / 2
    )

    # Bernoulli log-likelihood per particle
    logp = np.where(
        d == 0,
        np.log(p0 + 1e-12),
        np.log(1 - p0 + 1e-12),
    )

    # print("logp std =", np.std(logp))

    logw = logw + logp
    logw = logw - logsumexp(logw)

    return particles, logw
