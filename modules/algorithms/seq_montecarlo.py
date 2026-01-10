import numpy as np
import pymc as pm
import pytensor.tensor as pt

# ============================================================
# Physics: likelihood
# ============================================================

def qubit_p0(omega, T2, t):
    return pt.exp(-t / T2) * pt.cos(omega * t / 2) ** 2 \
           + (1 - pt.exp(-t / T2)) / 2


# ============================================================
# Build PyMC model ONCE
# ============================================================

def build_model():
    with pm.Model() as model:
        omega = pm.Data("omega", [])
        T2    = pm.Data("T2", [])
        t     = pm.Data("t", 0.0)

        p0 = qubit_p0(omega, T2, t)
        pm.Bernoulli("y", p=p0)

    logp_fn = model.compile_logp()
    return model, logp_fn


# ============================================================
# SMC utilities
# ============================================================

def init_particles(N):
    particles = np.empty((N, 2))
    particles[:, 0] = np.random.uniform(0.0, 1.0, size=N)   # omega
    particles[:, 1] = np.random.uniform(5.0, 15.0, size=N) # T2

    logw = -np.log(N) * np.ones(N)  # log-weights (uniform prior)
    return particles, logw


def normalize_log_weights(logw):
    maxlogw = np.max(logw)
    w = np.exp(logw - maxlogw)
    w /= w.sum()
    return np.log(w), w


def ess_from_logw(logw):
    _, w = normalize_log_weights(logw)
    return 1.0 / np.sum(w ** 2)


def resample(particles, logw):
    _, w = normalize_log_weights(logw)
    N = len(w)
    idx = np.random.choice(N, size=N, p=w)

    particles = particles[idx]
    logw = -np.log(N) * np.ones(N)
    return particles, logw


# ============================================================
# Bayesian update (LOG SPACE)
# ============================================================

def smc_update(particles, logw, d, t_val, model, logp_fn):
    N = len(particles)

    with model:
        pm.set_data({
            "omega": particles[:, 0],
            "T2": particles[:, 1],
            "t": t_val
        })

        y = np.full(N, d, dtype=np.int64)
        logp = logp_fn({"y": y})

    logw = logw + logp
    logw, _ = normalize_log_weights(logw)
    return particles, logw


def smc_step(
    particles,
    logw,
    d,
    t_val,
    model,
    logp_fn,
    ess_threshold=0.5
):
    particles, logw = smc_update(
        particles, logw, d, t_val, model, logp_fn
    )

    if ess_from_logw(logw) < ess_threshold * len(logw):
        particles, logw = resample(particles, logw)

    return particles, logw


# ============================================================
# Measurement simulator
# ============================================================

def measure(theta, t):
    omega, T2 = theta
    p0 = np.exp(-t / T2) * np.cos(omega * t / 2) ** 2 \
         + (1 - np.exp(-t / T2)) / 2
    return 0 if np.random.rand() < p0 else 1


# ============================================================
# Demo run
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    true_theta = np.array([0.7, 10.0])
    model, logp_fn = build_model()

    particles, logw = init_particles(5000)

    for k in range(10):
        t = np.random.uniform(0.1, 10.0)
        d = measure(true_theta, t)

        particles, logw = smc_step(
            particles, logw, d, t, model, logp_fn
        )

        _, w = normalize_log_weights(logw)

        mean = np.average(particles, weights=w, axis=0)
        cov  = np.cov(particles.T, aweights=w)

        print(f"step {k+1}")
        print("  mean:", mean)
        print("  cov diag:", np.diag(cov))
