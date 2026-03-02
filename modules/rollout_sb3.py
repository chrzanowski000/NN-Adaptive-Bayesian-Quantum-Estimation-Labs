from collections import deque

import numpy as np

from modules.algorithms.seq_montecarlo import (
    ess,
    init_particles,
    smc_update_no_resample,
)
from modules.rewards import posterior_variance
from modules.simulation import measure


def rollout(
    model,  # <-- SB3 model (e.g. TRPO instance)
    resample_fn,
    TRUE_OMEGA,
    N_PARTICLES,
    EPISODE_LEN,
    HISTORY_SIZE,
    action_low=None,  # optional clipping
    action_high=None,  # optional clipping
):
    """
    Rollout using an SB3 model
    """

    # -----------------------------
    # History buffer
    # -----------------------------
    t_history = deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

    # -----------------------------
    # Initialize SMC
    # -----------------------------
    particles, logw = init_particles(N_PARTICLES)
    initial_var = posterior_variance(particles, logw)

    # -----------------------------
    # Storage
    # -----------------------------
    t_list = []
    var_list = []
    ess_list = []
    mean_list = []

    # -----------------------------
    # Episode loop
    # -----------------------------
    for _ in range(EPISODE_LEN):
        # Normalize weights safely
        w = np.exp(logw - np.max(logw))
        w_sum = w.sum()
        if w_sum == 0 or not np.isfinite(w_sum):
            raise RuntimeError("Invalid weight normalization")
        w /= w_sum

        mean = np.sum(w * particles[:, 0])
        var = np.sum(w * (particles[:, 0] - mean) ** 2)

        if not np.isfinite(mean) or not np.isfinite(var):
            raise RuntimeError(f"Invalid state: mean={mean}, var={var}")

        mean_list.append(mean)
        var_list.append(var)
        ess_list.append(ess(logw))

        # -----------------------------
        # Build observation (NumPy!)
        # -----------------------------
        obs = np.array([mean, var] + list(t_history), dtype=np.float32)

        # -----------------------------
        # SB3 action prediction
        # -----------------------------
        action, _ = model.predict(obs, deterministic=True)
        t = float(action.item())

        # Optional: enforce action bounds
        if action_low is not None:
            t = max(t, action_low)
        if action_high is not None:
            t = min(t, action_high)

        if not np.isfinite(t):
            raise RuntimeError(f"Invalid action t={t}")

        t_list.append(t)
        t_history.append(t)

        # -----------------------------
        # Physics + SMC update
        # -----------------------------
        d = measure(TRUE_OMEGA, t)

        particles, logw = smc_update_no_resample(particles, logw, d, t)

        if ess(logw) < 0.75 * len(logw):
            particles, logw = resample_fn(particles, logw)

    # -----------------------------
    # Final statistics
    # -----------------------------
    final_var = posterior_variance(particles, logw)

    return {
        "initial_var": initial_var,
        "final_var": final_var,
        "reward": initial_var - final_var,
        "final_ess": ess(logw),
        "logw": logw,
        "mean_t": float(np.mean(t_history)),
        "particles": particles,
        "t_list": t_list,
        "var_list": var_list,
        "ess_list": ess_list,
        "mean_list": mean_list,
    }
