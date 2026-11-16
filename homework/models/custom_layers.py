"""Differentiable CNN layers built with PyTorch operations."""
import torch
from torch import nn


class GatedConv(nn.Module):
    """Rescale each convolution output using a learned channel gate."""

    def __init__(self, channels, width):
        super().__init__()
        self.conv = nn.Conv2d(channels, width, 3, padding=1)
        self.gate = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, x):
        return self.conv(x) * (2 * self.gate.sigmoid())


class ChannelAttention(nn.Module):
    """Squeeze spatial dimensions and predict channel gates."""

    def __init__(self, width):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(width, max(1, width // 4), 1), nn.ReLU(),
                                  nn.Conv2d(max(1, width // 4), width, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.mean((2, 3), keepdim=True))


class SmoothActivation(nn.Module):
    """Apply x·sigmoid(x) with automatic differentiation."""

    def forward(self, x):
        return x * x.sigmoid()


class MixedPool(nn.Module):
    """Learn a convex mixture of global average and maximum pooling."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        weight = self.mix.sigmoid()
        return weight * x.mean((2, 3), keepdim=True) + (1 - weight) * x.amax((2, 3), keepdim=True)
