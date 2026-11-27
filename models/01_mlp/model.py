"""MLP architecture: fully connected network with BatchNorm and Dropout, single logit output."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Input: (B, num_features) float tensor. Output: (B,) raw logits for P(income > 50K)."""

    def __init__(self, in_features: int, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
