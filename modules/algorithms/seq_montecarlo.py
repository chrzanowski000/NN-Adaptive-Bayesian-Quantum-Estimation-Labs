import numpy as np
import pymc as pm
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


def resample_liu_west(particles, logw, a=0.98):
    """
    Liu–West resampling.

    Args:
        particles : (N, D) array
        logw      : (N,) log weights
        a         : shrinkage parameter (0 < a < 1)

    Returns:
        new_particles, new_logw
    """
    N, D = particles.shape

    # normalize weights
    w = np.exp(logw - logsumexp(logw))

    # weighted mean
    mean = np.sum(w[:, None] * particles, axis=0)

    # weighted covariance
    diff = particles - mean
    cov = (
        (w[:, None] * diff).T @ diff
    )  # w[:, None] adds new axis (N,) becomes (N,1) but (N,1) will become (N,1,1)

    # shrinkage noise scale
    h2 = 1.0 - a**2
    chol = np.linalg.cholesky(h2 * cov + 1e-12 * np.eye(D))

    # resample indices
    idx = np.random.choice(N, size=N, p=w)

    # Liu–West update
    new_particles = a * particles[idx] + (1 - a) * mean + np.random.randn(N, D) @ chol.T

    # reset weights
    new_logw = -np.log(N) * np.ones(N)

    return new_particles, new_logw


# --------------------------------------------------
# One SMC update
# --------------------------------------------------


def smc_step(particles, logw, d, t, model, logp_fn):
    with model:
        pm.set_data({"omega": particles[:, 0], "t": float(t)})
        y = np.full(len(particles), d)
        logp = logp_fn({"y": y})

    logp = np.maximum(logp, -1e6)  # likelihood floor
    # print("logp std =", np.std(logp))
    logw = logw + logp
    logw = logw - logsumexp(logw)  # normalize
    print(ess(logw), len(logw))
    if ess(logw) < 0.1 * len(logw):
        print("resampled")
        particles, logw = resample(particles, logw)

    return particles, logw


def smc_update_no_resample(particles, logw, d, t):
    omega = particles[:, 0]

    p0 = (
        np.exp(-t / FIXED_T2) * np.cos(omega * t / 2) ** 2
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


### CUDA
import torch

# --------------------------------------------------
# Initialize particles (batched)
# --------------------------------------------------


def init_particles_batched(B, N, device="cuda"):
    particles = torch.rand(B, N, 1, device=device)

    logw_value = -torch.log(torch.tensor(N, device=device, dtype=torch.float32))

    logw = torch.full(
        (B, N),
        logw_value,
        device=device,  # ← THIS WAS MISSING
        dtype=torch.float32,
    )

    return particles, logw


# --------------------------------------------------
# Normalize log weights (batched)
# --------------------------------------------------


def normalize_batched(logw):
    """
    logw: (B, N)
    """
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)
    return torch.exp(logw)


# --------------------------------------------------
# ESS (batched)
# --------------------------------------------------


def ess_batched(logw):
    """
    Returns:
        ESS per batch element: (B,)
    """
    w = normalize_batched(logw)
    return 1.0 / torch.sum(w**2, dim=1)


# --------------------------------------------------
# Liu–West resampling (batched)
# --------------------------------------------------


def resample_liu_west_batched(particles, logw, a=0.98):
    """
    particles: (B, N, D)
    logw:      (B, N)
    """
    B, N, D = particles.shape
    device = particles.device

    w = normalize_batched(logw)  # (B, N)

    # weighted mean: (B, D)
    mean = torch.sum(w.unsqueeze(-1) * particles, dim=1)

    # weighted covariance (diagonal-safe version)
    diff = particles - mean.unsqueeze(1)
    cov = torch.matmul((w.unsqueeze(-1) * diff).transpose(1, 2), diff)

    h2 = 1.0 - a**2
    eye = torch.eye(D, device=device).unsqueeze(0).repeat(B, 1, 1)
    chol = torch.linalg.cholesky(h2 * cov + 1e-12 * eye)

    # multinomial resampling per batch
    idx = torch.multinomial(w, N, replacement=True)

    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, D)
    resampled = torch.gather(particles, 1, idx_expanded)

    noise = torch.randn(B, N, D, device=device)
    new_particles = (
        a * resampled
        + (1 - a) * mean.unsqueeze(1)
        + torch.matmul(noise, chol.transpose(1, 2))
    )

    new_logw = torch.full_like(
        logw, -torch.log(torch.tensor(N, device=device, dtype=torch.float32))
    )

    return new_particles, new_logw


# --------------------------------------------------
# SMC update (batched, no resample)
# --------------------------------------------------


def smc_update_no_resample_batched(particles, logw, d, t, FIXED_T2):
    """
    particles: (B, N, 1)
    logw:      (B, N)
    d:         (B,)
    t:         (B,)
    """

    B, N, _ = particles.shape
    omega = particles[:, :, 0]  # (B, N)

    t = t.unsqueeze(1)  # (B,1)
    exp_term = torch.exp(-t / FIXED_T2)

    p0 = exp_term * torch.cos(omega * t / 2) ** 2 + (1 - exp_term) / 2

    p0 = torch.clamp(p0, 1e-8, 1 - 1e-8)

    d = d.unsqueeze(1)  # (B,1)

    logp = torch.where(d == 0, torch.log(p0), torch.log(1 - p0))

    logw = logw + logp
    logw = logw - torch.logsumexp(logw, dim=1, keepdim=True)

    return particles, logw
