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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # output measurement time t > 0
        t = F.softplus(self.net(x)) + 1e-3
        return torch.clamp(t, 0.01, 1000.0)
    
