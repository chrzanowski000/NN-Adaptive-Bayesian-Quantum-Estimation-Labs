import numpy as np

FIXED_T2 = 100.0


def measure(true_omega, t, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    t = float(t)
    omega = float(true_omega)

    p0 = (
        np.exp(-t / FIXED_T2) * np.cos(omega * t / 2) ** 2
        + (1 - np.exp(-t / FIXED_T2)) / 2
    )

    return 0 if rng.random() < p0 else 1


# # example usage
# omega = 0.7
# T2 = 10.0
# t = 3.2
# N = 100_000
# samples = [measure(omega, T2, t) for _ in range(N)]
# print("freq(0) =", samples.count(0) / N)

import torch

FIXED_T2 = 10.0


def measure_batched(omegas, t, rng=None):
    """
    Batched measurement model (torch version).

    Args:
        omegas : (B,) tensor
        t      : (B,) tensor
        rng    : optional torch.Generator

    Returns:
        d : (B,) tensor of {0,1}
    """

    # Ensure float tensors
    omegas = omegas.float()
    t = t.float()

    exp_term = torch.exp(-t / FIXED_T2)

    p0 = exp_term * torch.cos(omegas * t / 2) ** 2 + (1 - exp_term) / 2

    # Numerical safety
    p0 = torch.clamp(p0, 1e-8, 1 - 1e-8)

    # Bernoulli sampling
    if rng is None:
        d = torch.bernoulli(p0)
    else:
        d = torch.bernoulli(p0, generator=rng)

    return d  # (B,)
