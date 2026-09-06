"""Evaluate one policy over many true omegas and write plots + a summary.

Works for the learned policies (TRPO, CEM) and for every training-free
baseline, through the same SMC and measurement code, so the numbers are
comparable across methods.

    python -m pipeline.evaluate --policy trpo --checkpoint artifacts/trpo_sb3_policy.zip
    python -m pipeline.evaluate --policy cem  --checkpoint artifacts/cem_policy_latest.pt
    python -m pipeline.evaluate --policy pgh-capped
    python -m pipeline.evaluate --policy fixed --fixed-t 1.0

Results land in `validation/run<N>/` (auto-incremented) unless --out is given.

This replaces the older `pipeline/test_*.py` scripts, which hardcode MLflow run
ids from a machine that no longer exists. See the README.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.policies import (
    CEMPolicy,
    ExponentialSweepPolicy,
    FixedTPolicy,
    PGHPolicy,
    RandomTPolicy,
    SigmaInversePolicy,
    TRPOPolicy,
)
from modules.rollout_generic import evaluate_policy
from modules.simulation import FIXED_T2
from utils.plotstyle import LINE, apply_style, log_y, refline

CHOICES = ["trpo", "cem", "pgh", "pgh-capped", "sigma-inv", "fixed", "exp-sweep", "random"]


def build_policy(args):
    needs_ckpt = {"trpo": "an SB3 .zip", "cem": "a CEM .pt"}
    if args.policy in needs_ckpt and not args.checkpoint:
        raise SystemExit(
            f"--policy {args.policy} needs --checkpoint ({needs_ckpt[args.policy]})"
        )

    if args.policy == "trpo":
        from sb3_contrib import TRPO as SB3TRPO

        return TRPOPolicy(SB3TRPO.load(args.checkpoint, device=args.device))
    if args.policy == "cem":
        return CEMPolicy.from_checkpoint(args.checkpoint, device=args.device)
    if args.policy == "pgh":
        return PGHPolicy()
    if args.policy == "pgh-capped":
        return PGHPolicy(t_cap=args.t_cap if args.t_cap is not None else FIXED_T2)
    if args.policy == "sigma-inv":
        return SigmaInversePolicy(k=args.k, t_cap=args.t_cap)
    if args.policy == "fixed":
        return FixedTPolicy(args.fixed_t)
    if args.policy == "exp-sweep":
        return ExponentialSweepPolicy(t0=args.t0, r=args.r, t_cap=args.t_cap)
    if args.policy == "random":
        return RandomTPolicy(args.action_low, args.action_high, seed=args.seed)
    raise SystemExit(f"unknown policy {args.policy!r}")


def next_run_dir(root="validation"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name[3:]) for p in root.glob("run*") if p.name[3:].isdigit()
    ]
    return root / f"run{max(existing, default=0) + 1}"


def make_plots(out, res, policy, args):
    steps = np.arange(res["var"].shape[1])
    var_mean = res["var"].mean(axis=0)
    ess_mean = res["ess"].mean(axis=0)
    t_mean = res["t"].mean(axis=0)
    t_steps = np.arange(res["t"].shape[1])
    # Per-step reward is the drop in ensemble-mean variance.
    reward = np.zeros_like(var_mean)
    reward[1:] = var_mean[:-1] - var_mean[1:]

    apply_style()

    def fig(name, title, ylabel, xlabel="Measurement number"):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}\n{policy.label}, {args.n_omegas} true omegas")
        plt.tight_layout()
        plt.savefig(out / name)
        plt.close()

    plt.figure()
    plt.plot(steps, var_mean, **LINE)
    log_y()
    fig("posterior_variance_log.png", "Posterior collapse", "Posterior variance")

    plt.figure()
    plt.plot(steps, var_mean, **LINE)
    fig("posterior_variance.png", "Posterior collapse", "Posterior variance")

    plt.figure()
    plt.plot(t_steps, t_mean, **LINE)
    refline(plt.gca(), y=FIXED_T2, label=f"T2 = {FIXED_T2:g}")
    log_y()
    plt.legend()
    fig("predicted_t.png", "Chosen interrogation time", "t")

    plt.figure()
    plt.plot(steps, ess_mean, **LINE)
    refline(plt.gca(), y=0.75 * args.n_particles, label="resample threshold")
    plt.legend()
    fig("ess.png", "Effective sample size", "ESS")

    plt.figure()
    plt.plot(steps, reward, **LINE)
    fig("reward_vs_step.png", "Reward per measurement", "Variance reduction")

    # Final-variance spread across true omegas: the mean alone hides whether a
    # method fails on a subset of frequencies (phase aliasing does exactly that).
    plt.figure()
    plt.plot(res["omegas"], res["var"][:, -1], linestyle="none", marker="o",
             markersize=3.0, markeredgewidth=0, alpha=0.6, color="#2a78d6")
    log_y()
    fig("final_variance_vs_omega.png", "Final variance by true omega",
        "Final posterior variance", xlabel="True omega")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=CHOICES)
    p.add_argument("--checkpoint", help="path for --policy trpo / cem")
    p.add_argument("--n-omegas", type=int, default=2000)
    p.add_argument("--episode-len", type=int, default=125)
    p.add_argument("--n-particles", type=int, default=2000)
    p.add_argument("--history-size", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", help="output dir (default: validation/run<N>)")
    p.add_argument("--action-low", type=float, default=0.1)
    p.add_argument("--action-high", type=float, default=3000.0)
    p.add_argument("--fixed-t", type=float, default=1.0, help="for --policy fixed")
    p.add_argument("--t-cap", type=float, default=None, help="cap on t, where supported")
    p.add_argument("--k", type=float, default=1.0, help="for --policy sigma-inv")
    p.add_argument("--t0", type=float, default=0.1, help="for --policy exp-sweep")
    p.add_argument("--r", type=float, default=1.05, help="for --policy exp-sweep")
    args = p.parse_args()

    policy = build_policy(args)
    omegas = np.random.default_rng(args.seed).uniform(0, 1, args.n_omegas)

    out = Path(args.out) if args.out else next_run_dir()
    out.mkdir(parents=True, exist_ok=True)

    print(f"evaluating {policy.label} over {args.n_omegas} omegas -> {out}")
    t0 = time.time()
    res = evaluate_policy(
        policy,
        omegas,
        n_particles=args.n_particles,
        episode_len=args.episode_len,
        history_size=args.history_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        progress=lambda done, tot: print(f"  {done}/{tot}", end="\r", flush=True),
    )
    runtime = time.time() - t0
    print()

    final = res["var"][:, -1]
    summary = {
        "policy": policy.name,
        "label": policy.label,
        "checkpoint": args.checkpoint,
        "n_omegas": args.n_omegas,
        "episode_len": args.episode_len,
        "n_particles": args.n_particles,
        "seed": args.seed,
        "FIXED_T2": FIXED_T2,
        "initial_variance": float(res["var"][:, 0].mean()),
        "final_variance_mean": float(final.mean()),
        "final_variance_median": float(np.median(final)),
        "final_variance_p10": float(np.percentile(final, 10)),
        "final_variance_p90": float(np.percentile(final, 90)),
        "total_variance_reduction": float(res["var"][:, 0].mean() - final.mean()),
        "mean_t": float(res["t"].mean()),
        "median_t": float(np.median(res["t"])),
        "max_t": float(res["t"].max()),
        "final_ess_mean": float(res["ess"][:, -1].mean()),
        "mean_resamples_per_episode": float(res["n_resample"].mean()),
        "runtime_seconds": runtime,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "trajectories.npz", **res)
    make_plots(out, res, policy, args)

    width = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"  {k:<{width}}  {v}")
    print(f"\nwrote {out}/  (summary.json, trajectories.npz, 6 plots)")


if __name__ == "__main__":
    main()
