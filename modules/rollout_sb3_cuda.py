import torch

from modules.algorithms.seq_montecarlo import (
    ess_batched,
    init_particles_batched,
    normalize_batched,
    smc_update_no_resample_batched,
)
from modules.simulation import FIXED_T2, measure_batched


def rollout_batch(
    model,
    omegas,
    resample_fn,  # <---- passed here
    N_PARTICLES,
    EPISODE_LEN,
    HISTORY_SIZE,
    action_low=None,
    action_high=None,
    device="cuda",
):
    """
    Fully CUDA batched rollout with pluggable resampling function.

    resample_fn must accept:
        (particles, logw)
    and return:
        (particles, logw)
    """

    # --------------------------------------------------
    # Ensure policy on correct device
    # --------------------------------------------------

    model.policy.eval()
    model.policy.to(device)

    # --------------------------------------------------
    # Setup
    # --------------------------------------------------

    omegas = torch.tensor(omegas, dtype=torch.float32, device=device)
    B = omegas.shape[0]

    particles, logw = init_particles_batched(B, N_PARTICLES, device=device)

    t_history = torch.zeros(B, HISTORY_SIZE, device=device)

    var_traj = torch.zeros(B, EPISODE_LEN, device=device)
    mean_traj = torch.zeros(B, EPISODE_LEN, device=device)
    ess_traj = torch.zeros(B, EPISODE_LEN, device=device)
    t_traj = torch.zeros(B, EPISODE_LEN, device=device)

    # --------------------------------------------------
    # Episode loop
    # --------------------------------------------------

    for step in range(EPISODE_LEN):
        # ----- Weight normalization -----
        w = normalize_batched(logw)
        omega_particles = particles[:, :, 0]

        mean = torch.sum(w * omega_particles, dim=1)
        var = torch.sum(w * (omega_particles - mean.unsqueeze(1)) ** 2, dim=1)
        ess = ess_batched(logw)

        mean_traj[:, step] = mean
        var_traj[:, step] = var
        ess_traj[:, step] = ess

        # ----- Observation -----
        obs = torch.cat(
            [mean.unsqueeze(1), var.unsqueeze(1), t_history],
            dim=1,
        )

        # ----- Actor forward -----
        with torch.no_grad():
            latent_pi, _ = model.policy.mlp_extractor(obs)
            mean_actions = model.policy.action_net(latent_pi)
            t = mean_actions.squeeze(-1)

        # ----- Action clipping -----
        if action_low is not None:
            t = torch.clamp(t, min=action_low)
        if action_high is not None:
            t = torch.clamp(t, max=action_high)

        t_traj[:, step] = t

        # ----- Update history -----
        t_history = torch.cat(
            [t_history[:, 1:], t.unsqueeze(1)],
            dim=1,
        )

        # ----- Measurement -----
        d = measure_batched(omegas, t)

        # ----- SMC weight update -----
        particles, logw = smc_update_no_resample_batched(
            particles,
            logw,
            d,
            t,
            FIXED_T2,
        )

        # ----- Resampling condition -----
        ess_after = ess_batched(logw)
        resample_mask = ess_after < 0.75 * N_PARTICLES

        if resample_mask.any():
            particles_res, logw_res = resample_fn(
                particles[resample_mask],
                logw[resample_mask],
            )

            particles[resample_mask] = particles_res
            logw[resample_mask] = logw_res

    return var_traj, mean_traj, ess_traj, t_traj
