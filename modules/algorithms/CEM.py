import torch
import numpy as np
from models.nn import TimePolicy
from modules.simulation import measure


class CEM:
    def __init__(self, policy_cls, pop_size=40, elite_frac=0.2, init_std=1.0):
        dummy = policy_cls()
        self.dim = sum(p.numel() for p in dummy.parameters())

        self.mu = torch.zeros(self.dim)
        self.sigma = torch.ones(self.dim) * init_std

        self.pop = pop_size
        self.elite = int(pop_size * elite_frac)
        self.policy_cls = policy_cls

    def sample(self):
        return self.mu + self.sigma * torch.randn(self.pop, self.dim)

    def update(self, thetas, rewards):
        elite_idx = torch.topk(torch.tensor(rewards), self.elite).indices
        elite = thetas[elite_idx]
        self.mu = elite.mean(0)
        self.sigma = elite.std(0) + 1e-8

    def step(self, rollout_fn):
        thetas = self.sample()
        rewards = [rollout_fn(theta) for theta in thetas]
        self.update(thetas, rewards)
        return np.mean(rewards), np.max(rewards)

