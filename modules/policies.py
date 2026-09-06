"""Measurement-time policies, all sharing one batched interface.

Every policy answers the same question: given the current SMC belief about the
qubit frequency omega, how long should the next Ramsey interrogation run?

    t = policy(mean, var, particles, w, t_history)      # all shapes (B, ...)

`mean`, `var` are (B,), `particles` is (B, N, 1), `w` is (B, N) normalised
weights, `t_history` is (B, HISTORY_SIZE). The return is a (B,) tensor of
interrogation times, which the rollout clips to the action bounds.

The learned policies (TRPO, CEM) are here so they can be benchmarked against
the training-free heuristics under identical SMC and measurement code.
"""

import numpy as np
import torch

import models.nn
from modules.simulation import FIXED_T2


class Policy:
    """Base class. Subclasses implement `__call__` and set `name`/`label`."""

    name = "policy"
    label = "policy"
    # Matplotlib colour, kept stable across every figure in the report.
    color = "#666666"

    def reset(self, batch_size, device):
        """Called once at the start of each rollout batch."""

    def __call__(self, mean, var, particles, w, t_history):
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Learned policies
# ----------------------------------------------------------------------------


class TRPOPolicy(Policy):
    """sb3-contrib TRPO actor, evaluated deterministically (Gaussian mean)."""

    color = "#1f77b4"

    def __init__(self, model, name="trpo", label="TRPO (learned)"):
        self.model = model
        self.name = name
        self.label = label

    def __call__(self, mean, var, particles, w, t_history):
        obs = torch.cat([mean.unsqueeze(1), var.unsqueeze(1), t_history], dim=1)
        with torch.no_grad():
            latent_pi, _ = self.model.policy.mlp_extractor(obs)
            return self.model.policy.action_net(latent_pi).squeeze(-1)


class CEMPolicy(Policy):
    """The CEM-optimised `TimePolicy_Fiderer_16` network.

    The network applies its own sigmoid squashing to [t_min, t_max], so no
    external clipping is needed -- but the rollout clips anyway, harmlessly.
    """

    color = "#9467bd"

    def __init__(self, net, name="cem", label="CEM (learned)"):
        self.net = net
        self.name = name
        self.label = label

    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kw):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cls_name = ckpt.get("policy_name", "TimePolicy_Fiderer_16")
        history_size = ckpt.get("history_size", 30)
        net = getattr(models.nn, cls_name)(history_size)
        net.load_state_dict(ckpt["state_dict"])
        net.eval().to(device)
        return cls(net, **kw)

    def __call__(self, mean, var, particles, w, t_history):
        obs = torch.cat([mean.unsqueeze(1), var.unsqueeze(1), t_history], dim=1)
        with torch.no_grad():
            return self.net(obs).squeeze(-1)


# ----------------------------------------------------------------------------
# Training-free heuristics
# ----------------------------------------------------------------------------


class PGHPolicy(Policy):
    """Particle Guess Heuristic (PGH).

    One of the two adaptive baselines in Fiderer et al.: draw two particles
    from p(theta | D_{k-1}) and set

        t_k = || theta_1 - theta_2 ||^{-1}

    The interrogation time therefore tracks the inverse posterior width, which
    keeps the accumulated phase difference across the posterior at roughly one
    radian. The paper describes PGH as a faster proxy for the sigma^-1
    heuristic that "introduces additional randomness" via the particle draw.

    With finite T2 this is asymptotically self-defeating -- as the posterior
    narrows, t grows past the coherence time and the fringe visibility
    exp(-t/T2) kills the information content. It can also settle into an
    aliasing trap: at t ~ 1/sigma ~ 11 the fringe wraps over a Uniform(0,1)
    prior, the posterior stays multimodal, sigma stays large, and t stays put.
    `t_cap` breaks the loop and is the standard fix.
    """

    color = "#d62728"

    def __init__(self, t_cap=None, name=None, label=None):
        self.t_cap = t_cap
        if name is None:
            name = "pgh" if t_cap is None else "pgh_capped"
        if label is None:
            label = "PGH" if t_cap is None else f"PGH (capped at {t_cap:g})"
        self.name = name
        self.label = label

    def __call__(self, mean, var, particles, w, t_history):
        omega = particles[:, :, 0]                       # (B, N)
        idx = torch.multinomial(w, 2, replacement=True)  # (B, 2)
        pair = torch.gather(omega, 1, idx)
        delta = (pair[:, 0] - pair[:, 1]).abs()
        # Two identical draws would give t = inf; floor the separation.
        delta = torch.clamp(delta, min=1e-9)
        t = 1.0 / delta
        if self.t_cap is not None:
            t = torch.clamp(t, max=self.t_cap)
        return t


class SigmaInversePolicy(Policy):
    """The sigma^-1 heuristic: t = k / sigma.

    The other adaptive baseline in Fiderer et al., there defined for the
    general multiparameter case as

        t_k = tr[ Cov(theta | D_{k-1}) ]^{-1/2}

    which for a single parameter is the inverse posterior standard deviation
    (k = 1 here). Originally derived by Ferrie et al. for omega estimation
    without decoherence (T2 -> infinity), where the paper notes it is optimal
    "in the greedy sense and only in the asymptotic limit N -> infinity".

    Same 1/width scaling as PGH but computed from the weighted moments rather
    than a two-particle sample, so it carries no sampling jitter.
    """

    color = "#ff7f0e"

    def __init__(self, k=1.0, t_cap=None, name=None, label=None):
        self.k = k
        self.t_cap = t_cap
        self.name = name or ("sigma_inv" if t_cap is None else "sigma_inv_capped")
        if label is None:
            label = f"{k:g}/sigma"
            if t_cap is not None:
                label += f" (capped at {t_cap:g})"
        self.label = label

    def __call__(self, mean, var, particles, w, t_history):
        sigma = torch.sqrt(torch.clamp(var, min=1e-18))
        t = self.k / sigma
        if self.t_cap is not None:
            t = torch.clamp(t, max=self.t_cap)
        return t


class FixedTPolicy(Policy):
    """Non-adaptive control: the same interrogation time at every step."""

    color = "#2ca02c"

    def __init__(self, t, name=None, label=None):
        self.t = float(t)
        self.name = name or f"fixed_t{self.t:g}".replace(".", "p")
        self.label = label or f"fixed t = {self.t:g}"

    def __call__(self, mean, var, particles, w, t_history):
        return torch.full_like(mean, self.t)


class RandomTPolicy(Policy):
    """Uniform random t over the action space -- the do-nothing control."""

    color = "#8c564b"

    def __init__(self, low=0.1, high=3000.0, seed=0, name="random_t", label=None):
        self.low, self.high, self.seed = float(low), float(high), seed
        self.name = name
        self.label = label or f"random t ~ U({low:g}, {high:g})"
        self._gen = None

    def reset(self, batch_size, device):
        self._gen = torch.Generator(device="cpu").manual_seed(self.seed)

    def __call__(self, mean, var, particles, w, t_history):
        u = torch.rand(mean.shape, generator=self._gen).to(mean.device)
        return self.low + (self.high - self.low) * u


class ExponentialSweepPolicy(Policy):
    """t_k = t0 * r^k -- the exp-sparse heuristic.

    Fiderer et al.'s non-adaptive baseline, there defined as t_k = (9/8)^k.
    Adaptive only in the trivial sense that it depends on the step index, not
    on the data. Included to separate "adaptivity helps" from "growing t helps".

    NOTE: the defaults here (t0=0.1, r=1.05) are NOT the paper's (9/8)^k, so as
    configured this is a slower ladder than the published exp-sparse heuristic.
    The defaults are kept because the committed baseline results table was
    measured with them; pass t0=1.0, r=1.125 to reproduce the paper's version.
    """

    color = "#e377c2"

    def __init__(self, t0=0.1, r=1.05, t_cap=None, name="exp_sweep", label=None):
        self.t0, self.r, self.t_cap = float(t0), float(r), t_cap
        self.name = name
        self.label = label or f"exponential sweep ({t0:g}*{r:g}^k)"
        self._k = 0

    def reset(self, batch_size, device):
        self._k = 0

    def __call__(self, mean, var, particles, w, t_history):
        t = self.t0 * (self.r**self._k)
        self._k += 1
        if self.t_cap is not None:
            t = min(t, self.t_cap)
        return torch.full_like(mean, t)


# ----------------------------------------------------------------------------
# Analytic reference: the single-shot Fisher-optimal time
# ----------------------------------------------------------------------------


def fisher_information(omega, t, T2=FIXED_T2):
    """Single-shot Fisher information of the decohering Ramsey likelihood.

        p0 = exp(-t/T2) cos^2(omega t / 2) + (1 - exp(-t/T2)) / 2
           = [1 + v cos(omega t)] / 2,      v = exp(-t/T2)

        I(omega; t) = (dp0/domega)^2 / [p0 (1 - p0)]
                    = t^2 v^2 sin^2(omega t) / [1 - v^2 cos^2(omega t)]

    Without decoherence (v -> 1) this is I = t^2: the quadratic growth that
    underlies Heisenberg scaling. With finite T2 the envelope t^2 exp(-2t/T2)
    peaks at t = T2, so information per shot is bounded and the asymptotic
    scaling in the shot count reverts to the standard quantum limit.
    """
    omega = np.asarray(omega, dtype=float)
    t = np.asarray(t, dtype=float)
    v = np.exp(-t / T2)
    num = (t**2) * (v**2) * np.sin(omega * t) ** 2
    den = 1.0 - (v**2) * np.cos(omega * t) ** 2
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


def optimal_t_grid(omega, t_grid=None, T2=FIXED_T2):
    """Numerically maximise the single-shot Fisher information over t."""
    if t_grid is None:
        t_grid = np.linspace(1e-3, 10 * T2, 200_000)
    info = fisher_information(omega, t_grid, T2=T2)
    i = int(np.argmax(info))
    return float(t_grid[i]), float(info[i])
