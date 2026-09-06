# NN-Adaptive Bayesian Quantum Estimation

Learning **when to measure** a qubit, so that you learn **what its frequency is** as
fast as physically possible.

This repository implements adaptive Bayesian experimental design for single-qubit
Ramsey frequency estimation. A sequential Monte Carlo filter maintains a posterior
over the unknown Larmor frequency $\omega$; a policy reads that posterior and picks
the interrogation time $t$ for the next shot; the measurement outcome sharpens the
posterior; repeat. The policy is trained by reinforcement learning (TRPO) or by a
gradient-free evolutionary search (CEM), following the approach of Fiderer, Schuff
& Braun, *Neural-network heuristics for adaptive Bayesian quantum estimation*
(PRX Quantum 2, 020303, 2021) — the MLflow runs are tagged `project: fiderer`.

---

## Table of contents

- [The physics](#the-physics)
- [The inference layer](#the-inference-layer-smc-particle-filter)
- [The control problem](#the-control-problem)
- [Methods implemented](#methods-implemented)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running things](#running-things)
  - [TRPO, end to end](#trpo-end-to-end)
  - [CEM, end to end](#cem-end-to-end)
  - [Baselines (no training)](#baselines-no-training)
- [Experiment tracking](#experiment-tracking)
- [Known issues and gotchas](#known-issues-and-gotchas)

---

## The physics

### The experiment

A single qubit undergoing **Ramsey interferometry** in the presence of pure
dephasing:

1. Prepare $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$.
2. Let it precess freely under $H = \tfrac{\omega}{2}\sigma_z$ for a time $t$.
   This is the **control variable** — the one thing the experimenter chooses.
3. Apply a $\pi/2$ read-out pulse and measure in the computational basis.
4. Record the binary outcome $d \in \{0, 1\}$.

The qubit is not isolated: transverse noise destroys the coherence
$\rho_{01}(t) = \rho_{01}(0)\,e^{-t/T_2}e^{-i\omega t}$ on a timescale $T_2$.

### The likelihood

Solving the Lindblad pure-dephasing master equation and applying the Born rule
gives the outcome probability implemented in `modules/simulation.py:13-16`
(NumPy) and `modules/simulation.py:51-53` (Torch):

$$
P(d=0 \mid \omega, t) \;=\; e^{-t/T_2}\cos^2\!\left(\frac{\omega t}{2}\right) \;+\; \frac{1 - e^{-t/T_2}}{2}
\;\equiv\; \frac{1}{2}\Bigl[\,1 + e^{-t/T_2}\cos(\omega t)\,\Bigr]
$$

and $P(d=1) = 1 - P(d=0)$. The two forms are algebraically identical.

Read it as a **decaying interference fringe**:

| term | meaning |
|---|---|
| $\cos^2(\omega t/2)$ | the Ramsey fringe — this is what carries information about $\omega$ |
| $v(t) = e^{-t/T_2}$ | the **visibility**, or coherent fraction |
| $(1-v)/2$ | the fully dephased fraction, which returns a coin flip and tells you nothing |

As $t \to \infty$ the visibility vanishes, $P(d=0) \to 1/2$, and the measurement
becomes pure noise. **Decoherence therefore imposes a hard ceiling on how long
it is worth interrogating.**

Both the simulator and the filter's likelihood use this same expression, and
$T_2$ is treated as exactly known. The model is well-specified: there is no
model mismatch and no nuisance parameter.

### What is being estimated

| quantity | value | where |
|---|---|---|
| parameter | $\omega$, a single scalar (the Larmor frequency / detuning) | `seq_montecarlo.py:9` |
| prior | $\omega \sim \mathrm{Uniform}(0, 1)$, so $\mathrm{Var}[\omega] = 1/12 \approx 0.0833$ | `seq_montecarlo.py:9`, `:127` |
| true values | drawn from the same $\mathrm{Uniform}(0,1)$ | `train_TRPO_baseline.py:67` |
| coherence time | $T_2 = 10$ | `modules/simulation.py:31` (see [gotchas](#1-fixed_t2-is-defined-twice)) |
| control | $t \in [0.1,\ 3000]$ | `train_TRPO_baseline.py:54-55` |

Units are arbitrary — $\omega$ and $t$ only ever appear as the product $\omega t$,
and $T_2$ shares the units of $t$. With $\omega \lesssim 1$ and $T_2 = 10$, roughly
$\omega T_2 \lesssim 10$ radians of phase accumulate before coherence is lost, so a
few fringes are resolvable within the coherence window.

### Fisher information: why this problem is interesting

Differentiating the likelihood gives the single-shot Fisher information
(implemented for reference in `modules/policies.py:fisher_information`):

$$
I(\omega; t) \;=\; \frac{(\partial_\omega p_0)^2}{p_0(1-p_0)} \;=\; \frac{t^2\,v^2\sin^2(\omega t)}{1 - v^2\cos^2(\omega t)},
\qquad v = e^{-t/T_2}
$$

Two regimes fall out of this, and the tension between them is the whole design
problem:

**Without decoherence** ($T_2 \to \infty$, $v \to 1$) this collapses to $I = t^2$.
Information grows *quadratically* in interrogation time. Splitting a total resource
time $T$ optimally then yields $\mathrm{Var}(\omega) \propto 1/T^2$ — **Heisenberg
scaling** — instead of the $1/T$ **standard quantum limit** of non-adaptive
strategies. This quadratic gain is the entire motivation for adaptive protocols.

**With finite $T_2$** the envelope $t^2 e^{-2t/T_2}$ is maximised at $t = T_2$, so
information per shot is *bounded*:

$$
I_{\max} \approx 0.135\,T_2^2 \quad\text{at}\quad t = T_2
$$

No choice of $t$ escapes that bound. Over $K$ adaptive shots the Cramér–Rao bound
therefore gives $\mathrm{Var}(\omega) \gtrsim 1/(K\,I_{\max})$: asymptotically
**SQL-like in the shot count**, with the quantum advantage surviving only as the
$1/T_2^2$ prefactor. Decoherence converts the $t^2$ enhancement into a constant
factor.

Numerically, for $T_2 = 10$ the Fisher-optimal time sits at $t^\star \approx 8.4$–$9.5$
depending on $\omega$ (computed by `optimal_t_grid`), consistent with the $t = T_2$
envelope argument.

### The competing constraint: phase ambiguity

Maximising single-shot Fisher information is *not* the whole story, and this is
what makes a learned policy worth having. The likelihood depends on $\omega$ only
through $\cos(\omega t)$, which is periodic. If $t$ is large while the posterior is
still broad, several distinct $\omega$ values produce the same fringe phase, the
posterior goes **multimodal**, and the filter cannot tell the aliases apart.

So a good policy must trade off:

- **small $t$** — unambiguous, but low information per shot ($I \sim t^2$);
- **$t \approx T_2$** — maximum information per shot, but aliases a broad prior;
- **$t \gg T_2$** — decohered, zero information, strictly wasted.

The textbook resolution is to grow $t$ as the posterior narrows, keeping the phase
spread across the posterior at roughly one radian — which is exactly what the
Particle Guess Heuristic does (`modules/policies.py:PGHPolicy`), and what an RL
policy is expected to discover on its own.

### The reward

The reward is the **reduction in posterior variance** at each step
(`modules/rewards.py:11-12`, inlined at `train_TRPO_baseline.py:300,317`):

$$
r_k \;=\; V_{k-1} - V_k, \qquad V_k \;=\; \sum_i w_i\bigl(\omega_i - \bar{\omega}\bigr)^2
$$

The per-step rewards telescope, so the undiscounted episode return is exactly
$V_0 - V_T$. Note this is a **linear** variance reduction, not log-variance or
information gain — see [gotcha 4](#4-the-reward-signal-vanishes-after-the-first-few-steps).

---

## The inference layer: SMC particle filter

`modules/algorithms/seq_montecarlo.py`

$\omega$ is a **static** parameter — it does not evolve — so the filter only ever
reweights particles; the particle locations move solely during resampling.

**Representation.** $N$ particles $\{\omega_i\}$ with log-weights $\{\log w_i\}$.
Shapes are `(N, 1)` / `(N,)` in the NumPy path, `(B, N, 1)` / `(B, N)` in the
batched Torch path, where `B` indexes independent experiments.

**Bayes update** (`seq_montecarlo.py:97-117`). Log-domain throughout:

$$
\log w_i \;\leftarrow\; \log w_i + \log P(d \mid \omega_i, t) \;-\; \mathrm{logsumexp}(\cdot)
$$

Renormalised every step, so weights always sum to one.

**Effective sample size** (`seq_montecarlo.py:19-21`). The Kish ESS,
$\mathrm{ESS} = 1/\sum_i w_i^2 \in [1, N]$.

**Resampling** fires when $\mathrm{ESS} < 0.75N$ — an aggressive threshold (0.5N is
the usual default), so it triggers on most steps. Two schemes exist; the default
everywhere is **Liu–West kernel resampling** (`seq_montecarlo.py:32-71`, $a = 0.98$):

$$
\omega_i' \;=\; a\,\omega_{\mathrm{idx}(i)} + (1-a)\,\bar{\omega} + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal{N}(0,\, h^2\Sigma),\quad h^2 = 1 - a^2
$$

The shrinkage toward the weighted mean exactly cancels the variance added by the
jitter ($a^2\Sigma + h^2\Sigma = \Sigma$), so the posterior's first two moments are
preserved. The jitter is what prevents particle impoverishment — plain multinomial
resampling (`resample`, `seq_montecarlo.py:24-29`) would collapse all particles onto
a handful of duplicated values, since a static parameter never gets diffused by
process noise.

---

## The control problem

Framed as a Gymnasium environment, `AdaptiveSMCEnv` (`train_TRPO_baseline.py:225-343`):

| | |
|---|---|
| **observation** | `[posterior_mean, posterior_var, *last_30_interrogation_times]` → `Box(32,)` |
| **action** | interrogation time $t$ → `Box(0.1, 3000.0, (1,))` |
| **reward** | $V_{k-1} - V_k$ |
| **episode** | 100 measurements (training) / 125 (evaluation), one true $\omega$ per episode |

---

## Methods implemented

### Learned policies

| method | optimiser | network | entry point |
|---|---|---|---|
| **TRPO** | trust-region policy optimisation (`sb3_contrib`) | `MlpPolicy`, $\pi$ and $V$ both `[256, 256]` | `pipeline/train_TRPO_baseline.py` |
| **CEM** | cross-entropy method over the flat 545-dim weight vector | `TimePolicy_Fiderer_16` — one hidden layer of 16, `Tanh`, sigmoid-squashed output | `pipeline/train_CEM.py` |

The CEM network squashes its output structurally,
$t = t_{\min} + (t_{\max}-t_{\min})\,\sigma(z)$ (`models/nn.py:97`), so it can
never emit an out-of-range time. The TRPO actor is an unbounded Gaussian whose
sample is clipped in `step()` (`train_TRPO_baseline.py:298`).

### Training-free baselines

`modules/policies.py` — these exist so the learned policies can be measured
against something. All share one batched interface and run through **identical**
SMC and measurement code via `modules/rollout_generic.py`, which is the only way a
comparison means anything.

| policy | rule | why it's here |
|---|---|---|
| `PGHPolicy` | $t = 1/\lvert\omega_1 - \omega_2\rvert$, two particles drawn from the posterior | the standard baseline in this literature (Wiebe & Granade); keeps phase spread $\approx 1$ rad |
| `PGHPolicy(t_cap=T2)` | as above, capped at the coherence time | the standard fix for the decoherence-limited case |
| `SigmaInversePolicy` | $t = k/\sigma$ | deterministic cousin of PGH, no sampling jitter |
| `FixedTPolicy` | constant $t$ | non-adaptive control — separates "adaptivity helps" from "a good $t$ helps" |
| `ExponentialSweepPolicy` | $t_k = t_0 r^k$ | open-loop ladder; depends on step index, not on data |
| `RandomTPolicy` | $t \sim U(0.1, 3000)$ | the do-nothing floor |

---

## Repository layout

```
modules/
  simulation.py            measurement model (Born rule + T2 decay), NumPy and Torch
  rewards.py               posterior variance and variance-reduction reward
  algorithms/
    seq_montecarlo.py      SMC filter: init, Bayes update, ESS, Liu-West resampling
    CEM.py                 cross-entropy method optimiser
  rollout.py               episode rollout for a raw torch policy      -> used by CEM
  rollout_sb3.py           episode rollout for an SB3 model, CPU       -> used by test_TRPO_N
  rollout_sb3_cuda.py      batched rollout for an SB3 model, GPU       -> used by test_TRPO_N_cuda
  rollout_generic.py       batched rollout for ANY policy              -> used by evaluate
  policies.py              baseline policy zoo + Fisher-information reference

models/nn.py               policy networks (TimePolicy_Fiderer_16 is the live one)

pipeline/
  evaluate.py              evaluate ANY policy, write plots + summary   <- start here
  train_TRPO_baseline.py   TRPO training (defines AdaptiveSMCEnv)
  train_CEM.py             CEM training
  test_TRPO_N.py           evaluate a TRPO policy over N true omegas, CPU
  test_TRPO_N_cuda.py      same, batched on GPU
  test_CEM.py              evaluate a CEM policy, single omega
  test_CEM_N.py            evaluate a CEM policy over N true omegas
                           (the four test_*.py are legacy; see below)

utils/
  plotstyle.py             shared matplotlib style: palette, marks, log axes
  git_utils/git.py         commit / branch / dirty-flag provenance for MLflow
  cov.py, network_fill.py  unused

artifacts/     scratch output dir, shared across ALL runs (see gotcha 6)
mlruns/        MLflow artifact root
mlflow.db      MLflow sqlite backend (~320 MB, gitignored)
```

Note there are **four** independent copies of the SMC episode loop: the three
`rollout*.py` files plus a fourth inlined in `AdaptiveSMCEnv.step`. They differ in
small ways (see [gotcha 5](#5-the-four-rollout-implementations-are-not-equivalent)).

---

## Installation

```bash
conda create -n NNBQE python=3.11 -y
conda activate NNBQE
pip install -r requirements.txt

# CUDA build of torch, if you have a GPU (skip on Apple Silicon)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130
```

Verified working set: Python 3.11.14, torch 2.9.1, stable-baselines3 2.7.1,
sb3-contrib 2.7.1, gymnasium 1.2.3, numpy 2.3.5, mlflow 3.8.1. The saved TRPO
checkpoint was produced with exactly these versions.

A GPU is **not** required. The batched rollout does ~4.4 ms per (125-step,
2000-particle) episode on CPU, so a full 10 000-$\omega$ evaluation takes about a
minute.

---

## Running things

All commands are run as modules **from the repository root**, with the `NNBQE`
environment active.

Two entry points matter:

| | |
|---|---|
| `python -m pipeline.train_TRPO_baseline` / `train_CEM` | train a policy |
| `python -m pipeline.evaluate --policy <name>` | evaluate any policy and write plots |

`pipeline/evaluate.py` handles **every** method — both learned policies and all
six training-free baselines — through identical inference and measurement code,
which is what makes the numbers comparable. It supersedes the older
`pipeline/test_*.py` scripts ([see below](#the-legacy-pipelinetest_py-scripts)).

### Quick check that everything works

No training required — the repo ships a trained TRPO policy, and the baselines
need none. This takes a few seconds:

```bash
python -m pipeline.evaluate --policy trpo \
    --checkpoint artifacts/trpo_sb3_policy.zip --n-omegas 256
python -m pipeline.evaluate --policy pgh-capped --n-omegas 256
```

---

### TRPO, end to end

**1. Train.**

```bash
python -m pipeline.train_TRPO_baseline
```

Defaults: 10<sup>5</sup> episodes × 100 steps = 10<sup>7</sup> timesteps, 10 000
particles, `net_arch=[256,256]`, `target_kl=1e-2`. Expect **4–25 hours** on CPU
(the spread is real: the 100k-timestep run took 154 s, the 100M-timestep run
averaged 6× slower per step). Set `DEVICE=cuda` if you have a GPU.

Checkpoints are written every 50 000 timesteps, so you can evaluate a partially
trained policy at any point and a crash costs you at most one interval:

```
artifacts/checkpoints/trpo_step_<N>.zip    numbered history
artifacts/trpo_sb3_policy_latest.zip       rolling latest
artifacts/trpo_sb3_policy.zip              final, written when training completes
artifacts/checkpoints/trpo_interrupted.zip written on crash or Ctrl-C
```

Shorten a run with the environment overrides:

```bash
# ~10 minutes instead of hours
N_TRAIN_EPISODES=5000 N_PARTICLES=2000 CHECKPOINT_EVERY_STEPS=25000 \
python -m pipeline.train_TRPO_baseline
```

**2. Evaluate.**

```bash
python -m pipeline.evaluate --policy trpo \
    --checkpoint artifacts/trpo_sb3_policy_latest.zip \
    --n-omegas 10000
```

Any of these checkpoints work as `--checkpoint`:

```bash
artifacts/trpo_sb3_policy.zip                                    # best available (run caaabcb8)
artifacts/checkpoints/trpo_step_000250000.zip                    # any mid-training checkpoint
mlruns/3/809c8bf6c4cb45569dc7719cc82e24a9/artifacts/trpo_sb3_policy.zip
```

---

### CEM, end to end

**1. Train.**

```bash
python -m pipeline.train_CEM
```

Defaults: 100 generations × population 1000, elite fraction 0.1, 2000 particles,
100-step episodes, optimising the flat 545-dim weight vector of
`TimePolicy_Fiderer_16`. **About 45 minutes** on CPU (~27 s per generation).

Checkpoints every 5 generations:

```
artifacts/checkpoints/cem_gen_<N>.pt   numbered history
artifacts/cem_policy_latest.pt         rolling latest
artifacts/checkpoints/cem_final.pt     written when training completes
```

Each `.pt` holds the policy `state_dict` alongside the CEM distribution
(`mu`, `sigma`), so a run can be inspected or resumed.

```bash
# ~2 minutes instead of 45
CEM_GENERATIONS=10 CEM_POP=200 python -m pipeline.train_CEM
```

**2. Evaluate.**

```bash
python -m pipeline.evaluate --policy cem \
    --checkpoint artifacts/cem_policy_latest.pt \
    --n-omegas 10000
```

> The CEM checkpoint that ships in `mlruns/1/` is from a 2-generation run and is
> effectively untrained. Train first before drawing any conclusion about CEM.

---

### Baselines (no training)

```bash
python -m pipeline.evaluate --policy pgh                        # t = 1/|w1 - w2|
python -m pipeline.evaluate --policy pgh-capped                 # capped at T2
python -m pipeline.evaluate --policy pgh-capped --t-cap 2.0     # capped elsewhere
python -m pipeline.evaluate --policy sigma-inv --k 1.0          # t = k/sigma
python -m pipeline.evaluate --policy fixed --fixed-t 1.0        # constant t
python -m pipeline.evaluate --policy exp-sweep --t0 0.1 --r 1.05
python -m pipeline.evaluate --policy random                     # t ~ U(0.1, 3000)
```

### What evaluation produces

Output goes to `validation/run<N>/` (auto-incremented), or `--out DIR`:

| file | contents |
|---|---|
| `summary.json` | final variance mean/median/p10/p90, mean & max `t`, ESS, resample count, runtime |
| `trajectories.npz` | raw `var`, `mean`, `ess`, `t`, `omegas`, `n_resample` arrays |
| `posterior_variance_log.png` | posterior collapse, log axis — the headline plot |
| `posterior_variance.png` | same, linear axis |
| `predicted_t.png` | chosen interrogation time vs step, against the $T_2$ line |
| `ess.png` | effective sample size, against the resample threshold |
| `reward_vs_step.png` | per-measurement variance reduction |
| `final_variance_vs_omega.png` | final variance against true $\omega$ — exposes aliasing |

Useful flags: `--n-omegas` (default 2000), `--episode-len` (125),
`--n-particles` (2000), `--batch-size` (512), `--seed` (42), `--device`,
`--out`. Run `python -m pipeline.evaluate --help` for the full list.

Cost is roughly 4.4 ms per episode, so 10 000 $\omega$ takes about a minute on CPU.

### Comparing methods

Give each run its own directory and read the summaries side by side:

```bash
for m in trpo pgh pgh-capped sigma-inv random; do
    extra=""
    [ "$m" = trpo ] && extra="--checkpoint artifacts/trpo_sb3_policy.zip"
    python -m pipeline.evaluate --policy "$m" $extra \
        --n-omegas 4096 --out "validation/cmp_$m"
done

python - <<'EOF'
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob("validation/cmp_*/summary.json"))]
print(f"{'method':<32}{'final var':>12}{'median t':>10}")
for r in sorted(rows, key=lambda r: r["final_variance_mean"]):
    print(f"{r['label']:<32}{r['final_variance_mean']:>12.3e}{r['median_t']:>10.2f}")
EOF
```

### Plot styling

All figures use `utils/plotstyle.py` — one validated categorical palette, thin
lines, small point marks, and a recessive grid. Import `apply_style()` before
creating figures and use `LINE` for trajectories, `log_y()` for log axes (it
labels the minor ticks, which matplotlib does not by default), and `refline()`
for annotation lines. Change the look in one place rather than per script.

### Loading a policy in your own code

```python
from sb3_contrib import TRPO
model = TRPO.load("artifacts/trpo_sb3_policy.zip", device="cpu")

from modules.policies import CEMPolicy, TRPOPolicy, PGHPolicy
cem = CEMPolicy.from_checkpoint("artifacts/cem_policy_latest.pt")
```

Then roll any of them out directly:

```python
import numpy as np
from modules.rollout_generic import evaluate_policy
from modules.simulation import FIXED_T2

omegas = np.random.default_rng(42).uniform(0, 1, 4096)
for policy in [TRPOPolicy(model), cem, PGHPolicy(t_cap=FIXED_T2)]:
    r = evaluate_policy(policy, omegas, n_particles=2000, episode_len=125)
    print(f"{policy.label:28s} final var = {r['var'][:, -1].mean():.3e}")
```

`evaluate_policy` returns numpy arrays: `var`, `mean`, `ess` of shape
`(B, episode_len + 1)` and `t` of shape `(B, episode_len)`. Index $k$ of `var` is
the posterior **after** $k$ measurements, so `var[:, 0]` is the prior and
`var[:, -1]` is the true final posterior.

### Environment overrides for training

TRPO: `N_PARTICLES`, `EPISODE_LEN`, `HISTORY_SIZE`, `RANDOM_SEED`,
`N_TRAIN_EPISODES`, `PLOT_COUNT`, `CHECKPOINT_EVERY_STEPS`, `DEVICE`,
`MLFLOW_TRACKING_URI`.

CEM: `N_PARTICLES`, `EPISODE_LEN`, `CEM_POP`, `CEM_ELITE_FRAC`, `CEM_INIT_STD`,
`CEM_GENERATIONS`, `HISTORY_SIZE`, `RANDOM_SEED`, `CHECKPOINT_EVERY_GENS`,
`MLFLOW_TRACKING_URI`.

```bash
# 30-second smoke tests
N_PARTICLES=200 EPISODE_LEN=10 N_TRAIN_EPISODES=300 PLOT_COUNT=3 \
CHECKPOINT_EVERY_STEPS=1000 python -m pipeline.train_TRPO_baseline

N_PARTICLES=200 EPISODE_LEN=10 CEM_POP=20 CEM_GENERATIONS=3 \
python -m pipeline.train_CEM
```

### The legacy `pipeline/test_*.py` scripts

Superseded by `pipeline/evaluate.py`, and **none of them run unmodified.** Each
hardcodes an MLflow `RUN_ID` from a machine that no longer exists, and the TRPO
ones hardcode `file:///home/chrzanowski/mlflow_tracking` as the tracking store.
Additionally:

- `test_CEM.py` calls `rollout()` without the required `resample_fn` argument and
  raises `TypeError` regardless — it was never updated when the signature changed.
- `test_TRPO_N_cuda.py` hardcodes `DEVICE = "cuda"` and calls
  `torch.cuda.get_device_name(0)`, so it cannot run without CUDA.
- Both shell wrappers in `scripts/` use `taskset`, which is Linux-only.

To revive one anyway, edit its `RUN_ID` and `mlflow.set_tracking_uri(...)`, or
replace the MLflow lookup with a direct path to a local checkpoint.


## Experiment tracking

Both trainers default to the repo-local sqlite backend and honour
`MLFLOW_TRACKING_URI`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Experiments: `fiderer / omega_estimation / trpo` and
`fiderer / omega_estimation / cem`. (CEM previously logged into the experiment
literally named `... / trpo`; that is fixed.)

Logged per episode (TRPO) or per generation (CEM): `mean_reward`,
`mean_init_var`, `mean_final_var`, `mean_ess`, `mean_t`, `true_omega`, plus
`max_reward`, `sigma_mean`, `mu_norm` for CEM. Params capture the full
hyperparameter set plus git commit / branch / dirty flag.

Historical runs already in `mlflow.db` (no retraining needed to replot them):

| run | experiment | episodes logged | best `mean_final_var` | weights on disk |
|---|---|---|---|---|
| `16eda430…` | trpo_baseline | 270 764 | **9.28e-4** | ✗ crashed before saving |
| `caaabcb8…` | trpo_baseline | 1 003 | 2.85e-3 | ✓ `artifacts/trpo_sb3_policy.zip` |
| `809c8bf6…` | trpo_baseline | 100 | 2.44e-3 | ✓ in `mlruns/3/` |
| `b28a7dbc…` | trpo_baseline | 102 | 8.19e-2 | ✓ in `mlruns/3/` (barely learned) |
| `bc96386c…` | cem | 2 generations | — | ✓ `mlruns/1/models/…/model.pth` (untrained) |

Extract a metric history without loading MLflow:

```bash
sqlite3 -readonly mlflow.db \
  "select step, value from metrics
   where run_uuid='16eda43052fd436e8ff76d06ba9d3f0e' and key='mean_final_var'
   order by step;"
```

---

## Known issues and gotchas

These are real and worth reading before trusting any number out of this repo.

### 1. `FIXED_T2` is defined twice

`modules/simulation.py` sets `FIXED_T2 = 100.0` at line 3 and `FIXED_T2 = 10.0` at
line 31. **The second wins** — every consumer, including the value logged to
MLflow, sees $T_2 = 10$. Line 3 is dead and is a trap for anyone reading the file.

### 2. The action range is 300× the coherence time

$t_{\max} = 3000$ with $T_2 = 10$ means $e^{-t/T_2} = e^{-300}$, which underflows to
exactly `0.0` in floating point. Any $t \gtrsim 200$ yields $p_0 = 0.5$ identically —
a measurement with **literally zero** Fisher information. More than 99% of the
action range is physically useless, which makes exploration far harder than it
needs to be. `t_max = 3000` would be a sane "coherence-scale bound" only for the
dead $T_2 = 100$. Either delete line 31 or shrink `T_MAX` to $O(10\text{–}100)$ —
but note that changing either invalidates comparison with the existing checkpoints.

### 3. The observation vector is badly scaled

`[mean, var, *t_history]` mixes a variance of order $10^{-4}$ with interrogation
times of order $10^3$, and is fed unnormalised into the MLP. There is no
`VecNormalize` and no `check_env` call.

### 4. The reward signal vanishes after the first few steps

The return is capped at $V_0 = 1/12 \approx 0.083$, and the posterior variance falls
by two orders of magnitude within the first ~20 measurements. Per-step rewards
beyond that are $\sim 10^{-5}$ — numerically negligible against the early steps. The
adaptive-design literature optimises $-\log V$ or the information gain
$\log(V_{k-1}/V_k)$ precisely because those are scale-free and give constant
signal throughout an exponential collapse. **If late-episode policy behaviour
looks unlearned, this reward shaping is the first thing to examine, not the
network.**

### 5. The four rollout implementations are not equivalent

- Numerical floors differ: `+1e-12` inside the log (NumPy) vs
  `clamp(1e-8, 1-1e-8)` on the probability (Torch). CPU and GPU results are not
  bit-comparable.
- The CUDA path bypasses `model.predict` and reads
  `mlp_extractor → action_net` directly, skipping any SB3-side post-processing.
- `mean_t` used to mean different things in different files (the last 30 times vs
  all of them); `modules/rollout.py` is now fixed to match the env.
- `var_traj` / `var_list` in the legacy paths record the variance **before** each
  measurement, so their reported `final_posterior_variance` is one measurement
  stale. `rollout_generic.py` records `episode_len + 1` points and does not have
  this problem.

### 6. `artifacts/` is shared scratch, not per-run output

Every run writes plots there and then uploads them to MLflow, so files from
different runs interleave by episode number — `posterior_ep_050.png` and
`posterior_ep_250000.png` come from different runs. For anything you intend to
publish, use the per-run copies under `mlruns/<exp>/<run_id>/artifacts/`.

### 7. The best result in the repo is unreproducible

Run `16eda430…` trained for 67.9 hours, reached `mean_final_var` = 9.28e-4, and
ended `FAILED` before its single end-of-training `model.save()`. Its metric
history and posterior plots survive; **its weights do not.** The best loadable
policy (`caaabcb8…`, 1000 episodes) reaches ~2.85e-3, roughly 3× worse. Closing
that gap requires a retrain. Periodic checkpointing has since been added so this
cannot recur.

### 8. Reproducibility is incomplete

`init_particles` uses the *global* NumPy RNG, so `env.reset(seed=...)` does not
control particle initialisation. `TRUE_OMEGAS_LIST` is logged to MLflow as a
truncated numpy repr (`'[0.494 0.228 ... 0.520]'`), so the actual $\omega$ sequence
is recoverable only from `RANDOM_SEED`. Every historical run has
`git_dirty = True`, so exact code provenance is not recoverable either.

### 9. Dead code

`smc_step` (the pymc path) is called by nothing, yet its `import pymc` at
`seq_montecarlo.py:2` drags pymc/jax/pytensor into every entry point for zero
functionality. `utils/cov.py:posterior_cov_trace` is unused and would raise
`ValueError` for the 1-D parameter used here. `models/nn.TimePolicy_Fiderer` has
no `forward()` — which means archived checkpoints trained with that class (some
older MLflow runs) can no longer be executed.

### 10. Preliminary observation, not yet a validated result

A single 256-$\omega$ scan of the loadable TRPO policy against the baselines showed
the policy settling on $t \approx 0.18$ and being **beaten by a constant $t = 1$**
(final variance 7.8e-4 vs 4.5e-4), while constant $t = 10$ barely improved on the
prior — the phase-aliasing effect described above. This was one run at one seed
and has **not** been repeated with proper statistics; treat it as a hypothesis
about undertraining and reward shaping (gotcha 4), not a finding.

---

## License

See `LICENSE`.
