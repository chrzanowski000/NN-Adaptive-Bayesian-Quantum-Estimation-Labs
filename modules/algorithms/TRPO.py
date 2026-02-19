import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class TRPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_type="continuous", hidden_sizes=(64, 64)):
        super().__init__()
        if action_type not in ("continuous", "discrete"):
            raise ValueError("action_type must be 'continuous' or 'discrete'")

        self.action_type = action_type
        self.action_dim = action_dim

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        if action_type == "continuous":
            self.mean_head = nn.Linear(in_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.logits_head = nn.Linear(in_dim, action_dim)

    def forward(self, states):
        x = self.backbone(states)
        if self.action_type == "continuous":
            mean = self.mean_head(x)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std
        logits = self.logits_head(x)
        return logits

    def _dist(self, states):
        if self.action_type == "continuous":
            mean, log_std = self.forward(states)
            std = torch.exp(log_std)
            return Normal(mean, std)
        logits = self.forward(states)
        return Categorical(logits=logits)

    def get_log_prob(self, states, actions):
        dist = self._dist(states)
        if self.action_type == "continuous":
            return dist.log_prob(actions).sum(dim=-1)
        return dist.log_prob(actions.long())

    def get_kl(self, states, old_policy):
        if self.action_type == "continuous":
            old_mean, old_log_std = old_policy.forward(states)
            new_mean, new_log_std = self.forward(states)

            old_std = torch.exp(old_log_std)
            new_std = torch.exp(new_log_std)

            kl = (
                new_log_std
                - old_log_std
                + (old_std.pow(2) + (old_mean - new_mean).pow(2)) / (2.0 * new_std.pow(2) + 1e-8)
                - 0.5
            ).sum(dim=-1)
            return kl

        old_logits = old_policy.forward(states)
        new_logits = self.forward(states)
        old_probs = torch.softmax(old_logits, dim=-1)
        old_log_probs = torch.log_softmax(old_logits, dim=-1)
        new_log_probs = torch.log_softmax(new_logits, dim=-1)
        return (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)

    def get_entropy(self, states):
        dist = self._dist(states)
        if self.action_type == "continuous":
            return dist.entropy().sum(dim=-1)
        return dist.entropy()


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states).squeeze(-1)


@dataclass
class TrajectoryBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs_old: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor


class TRPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_type="continuous",
        hidden_sizes=(64, 64),
        gamma=0.99,
        lam=0.95,
        max_kl=1e-2,
        damping=1e-2,
        cg_iters=10,
        cg_residual_tol=1e-10,
        line_search_max_steps=10,
        line_search_backtrack=0.5,
        accept_ratio=0.1,
        value_lr=1e-3,
        value_iters=80,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.cg_residual_tol = cg_residual_tol
        self.line_search_max_steps = line_search_max_steps
        self.line_search_backtrack = line_search_backtrack
        self.accept_ratio = accept_ratio
        self.value_iters = value_iters

        self.policy = TRPOPolicy(state_dim, action_dim, action_type, hidden_sizes).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_sizes).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)

    @staticmethod
    def flat_params(model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    @staticmethod
    def set_params(model, flat):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n

    @staticmethod
    def flat_grad(grads):
        chunks = []
        for g in grads:
            if g is None:
                continue
            chunks.append(g.reshape(-1))
        if not chunks:
            return torch.tensor([])
        return torch.cat(chunks)

    def collect_trajectories(self, env, batch_size, max_episode_steps=None):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs_old = []
        values = []
        next_values = []

        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        episode_steps = 0

        while len(states) < batch_size:
            s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                dist = self.policy._dist(s_t)
                action_t = dist.sample()
                log_prob_t = self.policy.get_log_prob(s_t, action_t)
                value_t = self.value_net(s_t)

            if self.policy.action_type == "continuous":
                env_action = action_t.squeeze(0).cpu().numpy()
            else:
                env_action = int(action_t.item())

            step_out = env.step(env_action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, _ = step_out

            episode_steps += 1
            if max_episode_steps is not None and episode_steps >= max_episode_steps:
                done = True

            ns_t = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                next_value_t = self.value_net(ns_t) if not done else torch.zeros(1, device=self.device)

            states.append(s_t.squeeze(0))
            actions.append(action_t.squeeze(0))
            rewards.append(float(reward))
            dones.append(float(done))
            log_probs_old.append(log_prob_t.squeeze(0))
            values.append(value_t.squeeze(0))
            next_values.append(next_value_t.squeeze(0))

            state = next_state
            if done:
                reset_out = env.reset()
                state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                episode_steps = 0

        return TrajectoryBatch(
            states=torch.stack(states),
            actions=torch.stack(actions),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            log_probs_old=torch.stack(log_probs_old).detach(),
            values=torch.stack(values).detach(),
            next_values=torch.stack(next_values).detach(),
        )

    def compute_advantages(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * (1.0 - dones) * next_values - values
        advantages = torch.zeros_like(deltas)
        gae = torch.zeros(1, device=self.device)

        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.detach(), returns.detach()

    def surrogate_loss(self, states, actions, log_probs_old, advantages):
        log_probs_new = self.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs_new - log_probs_old)
        return torch.mean(ratio * advantages)

    def mean_kl(self, policy, old_policy, states):
        return policy.get_kl(states, old_policy).mean()

    def fisher_vector_product(self, v, states, old_policy):
        kl = self.mean_kl(self.policy, old_policy, states)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = self.flat_grad(grads)

        kl_v = (flat_grad_kl * v).sum()
        grads2 = torch.autograd.grad(kl_v, self.policy.parameters(), retain_graph=True)
        flat_hvp = self.flat_grad(grads2).detach()

        return flat_hvp + self.damping * v

    def conjugate_gradient(self, Avp_fn, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for _ in range(self.cg_iters):
            Avp = Avp_fn(p)
            alpha = rdotr / (torch.dot(p, Avp) + 1e-8)
            x = x + alpha * p
            r = r - alpha * Avp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < self.cg_residual_tol:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def line_search(self, states, actions, log_probs_old, advantages, old_policy, full_step, expected_improve_rate):
        old_params = self.flat_params(self.policy)
        old_loss = self.surrogate_loss(states, actions, log_probs_old, advantages).detach()

        for i in range(self.line_search_max_steps):
            step_frac = self.line_search_backtrack ** i
            candidate_params = old_params + step_frac * full_step
            self.set_params(self.policy, candidate_params)

            with torch.no_grad():
                new_loss = self.surrogate_loss(states, actions, log_probs_old, advantages)
                kl = self.mean_kl(self.policy, old_policy, states)

            actual_improve = new_loss - old_loss
            expected_improve = expected_improve_rate * step_frac

            if (
                actual_improve > 0.0
                and kl <= self.max_kl
                and (actual_improve / (expected_improve + 1e-8)) > self.accept_ratio
            ):
                return True

        self.set_params(self.policy, old_params)
        return False

    def update(self, batch):
        states = batch.states
        actions = batch.actions
        log_probs_old = batch.log_probs_old
        rewards = batch.rewards
        dones = batch.dones
        values = batch.values
        next_values = batch.next_values

        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)

        old_policy = copy.deepcopy(self.policy).to(self.device)
        for p in old_policy.parameters():
            p.requires_grad_(False)

        loss = self.surrogate_loss(states, actions, log_probs_old, advantages)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        g = self.flat_grad(grads).detach()

        if g.numel() == 0 or torch.norm(g) < 1e-10:
            policy_updated = False
        else:
            Fvp = lambda v: self.fisher_vector_product(v, states, old_policy)
            step_dir = self.conjugate_gradient(Fvp, g)
            f_step = Fvp(step_dir)
            shs = 0.5 * torch.dot(step_dir, f_step)

            if shs <= 0:
                policy_updated = False
            else:
                step_scale = torch.sqrt(self.max_kl / (shs + 1e-8))
                full_step = step_dir * step_scale
                expected_improve_rate = torch.dot(g, full_step).detach()
                policy_updated = self.line_search(
                    states,
                    actions,
                    log_probs_old,
                    advantages,
                    old_policy,
                    full_step,
                    expected_improve_rate,
                )

        for _ in range(self.value_iters):
            v_pred = self.value_net(states)
            value_loss = ((v_pred - returns) ** 2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        with torch.no_grad():
            final_surrogate = self.surrogate_loss(states, actions, log_probs_old, advantages).item()
            final_kl = self.mean_kl(self.policy, old_policy, states).item()
            entropy = self.policy.get_entropy(states).mean().item()

        return {
            "policy_updated": policy_updated,
            "surrogate_loss": final_surrogate,
            "kl": final_kl,
            "entropy": entropy,
            "adv_mean": advantages.mean().item(),
            "adv_std": advantages.std().item(),
            "value_loss": value_loss.item(),
            "batch_reward_mean": rewards.mean().item(),
        }

    def train_step(self, env, batch_size, max_episode_steps=None):
        batch = self.collect_trajectories(env, batch_size, max_episode_steps)
        return self.update(batch)
