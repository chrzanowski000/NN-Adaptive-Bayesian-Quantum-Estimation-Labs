"""Batched SMC rollout that accepts any `modules.policies.Policy`.

This is `modules.rollout_sb3_cuda.rollout_batch` generalised so that learned
and heuristic policies run through *identical* inference and measurement code
-- which is the only way a method comparison means anything.

Two deliberate differences from `rollout_sb3_cuda`:

1. Trajectories have EPISODE_LEN + 1 variance/mean/ESS entries, not
   EPISODE_LEN. The original recorded the belief *before* each measurement and
   so never stored the final posterior; its "final_posterior_variance" was one
   measurement stale. Index k here is the belief after k measurements, so
   index 0 is the prior and index EPISODE_LEN is the true final posterior.
2. `device` defaults to CPU and the RNG is seedable, so runs are reproducible
   on a machine without CUDA.
"""

import numpy as np
import torch

from modules.algorithms.seq_montecarlo import (
    ess_batched,
    init_particles_batched,
    normalize_batched,
    resample_liu_west_batched,
    smc_update_no_resample_batched,
)
from modules.simulation import FIXED_T2

ESS_RESAMPLE_FRACTION = 0.75


def rollout_policy_batch(
    policy,
    omegas,
    n_particles,
    episode_len,
    history_size,
    resample_fn=resample_liu_west_batched,
    action_low=0.1,
    action_high=3000.0,
    device="cpu",
    seed=None,
    record_particles=False,
):
    """Run `policy` on a batch of independent true omegas.

    Returns a dict of numpy arrays:
        var   (B, episode_len + 1)   posterior variance after k measurements
        mean  (B, episode_len + 1)   posterior mean
        ess   (B, episode_len + 1)   effective sample size
        t     (B, episode_len)       interrogation time chosen at step k
        n_resample (B,)              how many times the filter resampled
    plus `particles` / `weights` (B, N) when `record_particles` is set.
    """
    device = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)

    omegas = torch.as_tensor(omegas, dtype=torch.float32, device=device)
    B = omegas.shape[0]

    particles, logw = init_particles_batched(B, n_particles, device=device)
    t_history = torch.zeros(B, history_size, device=device)

    var_traj = torch.zeros(B, episode_len + 1, device=device)
    mean_traj = torch.zeros(B, episode_len + 1, device=device)
    ess_traj = torch.zeros(B, episode_len + 1, device=device)
    t_traj = torch.zeros(B, episode_len, device=device)
    n_resample = torch.zeros(B, device=device)

    policy.reset(B, device)

    def belief():
        w = normalize_batched(logw)
        omega_p = particles[:, :, 0]
        m = torch.sum(w * omega_p, dim=1)
        v = torch.sum(w * (omega_p - m.unsqueeze(1)) ** 2, dim=1)
        return w, m, v

    w, mean, var = belief()
    var_traj[:, 0], mean_traj[:, 0], ess_traj[:, 0] = var, mean, ess_batched(logw)

    for step in range(episode_len):
        t = policy(mean, var, particles, w, t_history)
        t = torch.clamp(t.reshape(B), min=action_low, max=action_high)
        # A NaN here would silently poison the whole filter; fail loudly.
        if not torch.isfinite(t).all():
            raise RuntimeError(
                f"policy {policy.name!r} produced non-finite t at step {step}"
            )
        t_traj[:, step] = t
        t_history = torch.cat([t_history[:, 1:], t.unsqueeze(1)], dim=1)

        # Measurement, inlined so the batched generator is seedable.
        exp_term = torch.exp(-t / FIXED_T2)
        p0 = exp_term * torch.cos(omegas * t / 2) ** 2 + (1 - exp_term) / 2
        d = torch.bernoulli(torch.clamp(p0, 1e-8, 1 - 1e-8))

        particles, logw = smc_update_no_resample_batched(
            particles, logw, d, t, FIXED_T2
        )

        resample_mask = ess_batched(logw) < ESS_RESAMPLE_FRACTION * n_particles
        if resample_mask.any():
            p_res, w_res = resample_fn(particles[resample_mask], logw[resample_mask])
            particles[resample_mask] = p_res
            logw[resample_mask] = w_res
            n_resample += resample_mask.float()

        w, mean, var = belief()
        k = step + 1
        var_traj[:, k], mean_traj[:, k], ess_traj[:, k] = var, mean, ess_batched(logw)

    out = {
        "var": var_traj.cpu().numpy(),
        "mean": mean_traj.cpu().numpy(),
        "ess": ess_traj.cpu().numpy(),
        "t": t_traj.cpu().numpy(),
        "n_resample": n_resample.cpu().numpy(),
        "omegas": omegas.cpu().numpy(),
    }
    if record_particles:
        out["particles"] = particles[:, :, 0].cpu().numpy()
        out["weights"] = w.cpu().numpy()
    return out


def evaluate_policy(
    policy,
    omegas,
    n_particles=2000,
    episode_len=125,
    history_size=30,
    batch_size=512,
    device="cpu",
    seed=0,
    progress=None,
):
    """Run `policy` over `omegas` in chunks and concatenate the trajectories."""
    omegas = np.asarray(omegas, dtype=np.float32)
    chunks = []
    for start in range(0, len(omegas), batch_size):
        chunk = omegas[start : start + batch_size]
        chunks.append(
            rollout_policy_batch(
                policy,
                chunk,
                n_particles=n_particles,
                episode_len=episode_len,
                history_size=history_size,
                device=device,
                seed=None if seed is None else seed + start,
            )
        )
        if progress is not None:
            progress(min(start + batch_size, len(omegas)), len(omegas))

    return {
        key: np.concatenate([c[key] for c in chunks], axis=0)
        for key in ("var", "mean", "ess", "t", "n_resample", "omegas")
    }
