"""Fully connected image classifier."""
from torch import nn


class FC(nn.Sequential):
    """Flatten an image and predict logits with an MLP."""

    def __init__(self, channels, size, deep=False):
        widths = [channels * size * size, 256, 128, 64, 10] if deep else [
            channels * size * size, 128, 64, 10]
        layers = [nn.Flatten()]
        for left, right in zip(widths[:-2], widths[1:-1]):
            layers.extend((nn.Linear(left, right), nn.ReLU()))
        super().__init__(*layers, nn.Linear(widths[-2], widths[-1], bias=False))
