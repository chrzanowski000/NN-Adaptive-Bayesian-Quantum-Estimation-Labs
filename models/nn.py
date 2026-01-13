# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class network_CEM(nn.Module):
    """
    Simple fully-connected neural network:
    input -> hidden(16) -> output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 16,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming initialization for stable training.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softplus(self.net(x))
    

class TimePolicy(nn.Module):
    def __init__(self, history_len=30):
        super().__init__()
        self.history_len = history_len
        input_dim = 2 + history_len   # μ, σ + historia t

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        t = F.softplus(self.net(x)) + 1e-3
        # t=torch.clamp(t, 0.05, 10.0)
        return t
