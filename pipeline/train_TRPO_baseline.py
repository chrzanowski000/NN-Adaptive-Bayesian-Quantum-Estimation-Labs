import os
from collections import deque

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from modules.algorithms.seq_montecarlo import (
    init_particles,
    smc_update_no_resample,
    ess,
    normalize,
    resample_liu_west,
    resample,
)
from modules.rewards import posterior_variance
from modules.simulation import measure, FIXED_T2
from utils.git_utils.git import get_git_branch, get_git_commit, git_is_dirty

try:
    from sb3_contrib import TRPO as SB3TRPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as e:
    raise ImportError(
        "Stable-Baselines TRPO is required. Install with: pip install stable-baselines3 sb3-contrib"
    ) from e


# ================= CONFIG =================
RESAMPLE_FN = resample_liu_west
# RESAMPLE_FN = resample

N_PARTICLES = 10000
EPISODE_LEN = 100
HISTORY_SIZE = 30
RANDOM_SEED = 50

N_TRAIN_EPISODES = 1000
TOTAL_TIMESTEPS = N_TRAIN_EPISODES * EPISODE_LEN

GAMMA = 0.99
GAE_LAMBDA = 0.95
TARGET_KL = 1e-2

PLOT_EVERY = 5
T_MIN = 0.1
T_MAX = 3000.0

np.random.seed(RANDOM_SEED)
TRUE_OMEGAS_LIST = np.random.uniform(0.0, 1.0, size=N_TRAIN_EPISODES)
# ==========================================


class AdaptiveSMCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        true_omegas,
        n_particles,
        episode_len,
        history_size,
        resample_fn,
        seed=0,
        t_min=0.1,
        t_max=3000.0,
    ):
        super().__init__()
        self.true_omegas = np.array(true_omegas, dtype=np.float64)
        self.n_particles = n_particles
        self.episode_len = episode_len
        self.history_size = history_size
        self.resample_fn = resample_fn
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        self.rng = np.random.default_rng(seed)
        self.omega_idx = 0

        self.action_space = spaces.Box(
            low=np.array([self.t_min], dtype=np.float32),
            high=np.array([self.t_max], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + history_size,),
            dtype=np.float32,
        )

        self._episode_reset_state()

    def _episode_reset_state(self):
        self.step_idx = 0
        self.t_history = deque([0.0] * self.history_size, maxlen=self.history_size)
        self.particles, self.logw = init_particles(self.n_particles)
        self.true_omega = float(self.true_omegas[self.omega_idx % len(self.true_omegas)])
        self.omega_idx += 1

        self.initial_var = posterior_variance(self.particles, self.logw)
        self.t_list = []
        self.var_list = []
        self.ess_list = []
        self.mean_list = []

    def _belief_stats(self):
        w = normalize(self.logw)
        mean = float(np.sum(w * self.particles[:, 0]))
        var = float(np.sum(w * (self.particles[:, 0] - mean) ** 2))
        return mean, var

    def _get_obs(self):
        mean, var = self._belief_stats()
        return np.array([mean, var] + list(self.t_history), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._episode_reset_state()
        return self._get_obs(), {}

    def step(self, action):
        t = float(np.clip(np.asarray(action).reshape(-1)[0], self.t_min, self.t_max))

        prev_var = posterior_variance(self.particles, self.logw)
        d = measure(self.true_omega, t, rng=self.rng)

        self.particles, self.logw = smc_update_no_resample(self.particles, self.logw, d, t)
        if ess(self.logw) < 0.75 * len(self.logw):
            self.particles, self.logw = self.resample_fn(self.particles, self.logw)

        mean, var = self._belief_stats()

        self.t_history.append(t)
        self.t_list.append(t)
        self.var_list.append(var)
        self.ess_list.append(float(ess(self.logw)))
        self.mean_list.append(mean)

        reward = prev_var - var

        self.step_idx += 1
        terminated = self.step_idx >= self.episode_len
        truncated = False

        info = {
            "true_omega": self.true_omega,
            "step_t": t,
            "step_var": var,
        }

        if terminated:
            info.update(
                {
                    "initial_var": float(self.initial_var),
                    "final_var": float(var),
                    "final_ess": float(ess(self.logw)),
                    "mean_t": float(np.mean(self.t_list)),
                    "reward_total": float(self.initial_var - var),
                    "logw": self.logw.copy(),
                    "particles": self.particles.copy(),
                }
            )

        return self._get_obs(), float(reward), terminated, truncated, info


class MlflowEpisodeCallback(BaseCallback):
    def __init__(self, artifacts_dir="artifacts", plot_every=5, verbose=0):
        super().__init__(verbose)
        self.episode_idx = 0
        self.artifacts_dir = artifacts_dir
        self.plot_every = plot_every

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            idx = self.episode_idx
            self.episode_idx += 1

            mlflow.log_metric("mean_reward", info["reward_total"], step=idx)
            mlflow.log_metric("mean_init_var", info["initial_var"], step=idx)
            mlflow.log_metric("mean_final_var", info["final_var"], step=idx)
            mlflow.log_metric("mean_ess", info["final_ess"], step=idx)
            mlflow.log_metric("mean_t", info["mean_t"], step=idx)
            mlflow.log_metric("true_omega", info["true_omega"], step=idx)

            if idx % self.plot_every == 0:
                w = normalize(info["logw"])
                particles = info["particles"]

                plt.figure(figsize=(6, 4))
                plt.hist(particles[:, 0], weights=w, bins=50, density=True)
                plt.axvline(info["true_omega"], color="red", linestyle="--", label="true ω")
                plt.xlabel("ω")
                plt.ylabel("posterior density")
                plt.title(f"Posterior (ep {idx})")
                plt.legend()

                fname = f"posterior_ep_{idx:03d}.png"
                fpath = os.path.join(self.artifacts_dir, fname)
                plt.savefig(fpath)
                plt.close()
                mlflow.log_artifact(fpath)

        return True


def evaluate_episode(model, env):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return info


mlflow.set_experiment("trpo_baseline_qubit_omega_only")
os.makedirs("artifacts", exist_ok=True)

if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run():
    mlflow.log_params(
        {
            "N_PARTICLES": N_PARTICLES,
            "EPISODE_LEN": EPISODE_LEN,
            "FIXED_T2": FIXED_T2,
            "TRUE_OMEGAS_LIST": TRUE_OMEGAS_LIST,
            "HISOTRY_SIZE": HISTORY_SIZE,
            "RANDOM_SEED": RANDOM_SEED,
            "resample_function": RESAMPLE_FN.__name__,
            "N_TRAIN_EPISODES": N_TRAIN_EPISODES,
            "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
            "TARGET_KL": TARGET_KL,
            "GAMMA": GAMMA,
            "GAE_LAMBDA": GAE_LAMBDA,
            "policy_name": "MlpPolicy",
            "git_commit": get_git_commit(),
            "git_branch": get_git_branch(),
            "git_dirty": git_is_dirty(),
        }
    )

    env = AdaptiveSMCEnv(
        true_omegas=TRUE_OMEGAS_LIST,
        n_particles=N_PARTICLES,
        episode_len=EPISODE_LEN,
        history_size=HISTORY_SIZE,
        resample_fn=RESAMPLE_FN,
        seed=RANDOM_SEED,
        t_min=T_MIN,
        t_max=T_MAX,
    )

    model = SB3TRPO(
        "MlpPolicy",
        env,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        target_kl=TARGET_KL,
        verbose=1,
        seed=RANDOM_SEED,
    )

    callback = MlflowEpisodeCallback(artifacts_dir="artifacts", plot_every=PLOT_EVERY)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    model_path = os.path.join("artifacts", "trpo_sb3_policy")
    model.save(model_path)
    mlflow.log_artifact(f"{model_path}.zip")

    eval_info = evaluate_episode(model, env)
    mlflow.log_metric("eval_final_var", eval_info["final_var"])
    mlflow.log_metric("eval_total_reward", eval_info["reward_total"])

    w = normalize(eval_info["logw"])
    particles = eval_info["particles"]
    plt.figure(figsize=(6, 4))
    plt.hist(particles[:, 0], weights=w, bins=50, density=True)
    plt.axvline(eval_info["true_omega"], color="red", linestyle="--", label="true ω")
    plt.xlabel("ω")
    plt.ylabel("posterior density")
    plt.title("Posterior (final eval)")
    plt.legend()
    final_plot = os.path.join("artifacts", "posterior_final_eval.png")
    plt.savefig(final_plot)
    plt.close()
    mlflow.log_artifact(final_plot)

    print("SB3 TRPO model saved")
