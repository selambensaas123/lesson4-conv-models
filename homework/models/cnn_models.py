"""CNN blocks for controlled comparisons."""
from torch import nn


class ConvUnit(nn.Sequential):
    """Convolution and activation with an optional normalization layer."""

    def __init__(self, left, right, kernel=3, norm=False):
        super().__init__(nn.Conv2d(left, right, kernel, padding=kernel // 2),
                         nn.BatchNorm2d(right) if norm else nn.Identity(), nn.ReLU())


class Residual(nn.Module):
    """Identity shortcut with a basic, bottleneck or widened branch."""

    def __init__(self, width, kind='basic'):
        super().__init__()
        if kind == 'bottleneck':
            hidden = max(1, width // 2)
            self.branch = nn.Sequential(ConvUnit(width, hidden, 1, True),
                                        ConvUnit(hidden, hidden, 3, True),
                                        nn.Conv2d(hidden, width, 1), nn.BatchNorm2d(width))
        else:
            hidden = width * 2 if kind == 'wide' else width
            self.branch = nn.Sequential(ConvUnit(width, hidden, 3, True),
                                        nn.Conv2d(hidden, width, 3, padding=1),
                                        nn.BatchNorm2d(width))
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class CNN(nn.Module):
    """Collect an activation map before global pooling and classification."""

    def __init__(self, channels, widths=(16, 24), kernels=(3, 3), residual=0,
                 kind='basic', dropout=0., pooling=None, layer=None):
        super().__init__()
        units = []
        for index, (width, kernel) in enumerate(zip(widths, kernels)):
            units.append(ConvUnit(channels, width, kernel) if layer is None or index else layer)
            if index == 1:
                units.append(nn.MaxPool2d(2))
            channels = width
        units.extend(Residual(channels, kind) for _ in range(residual))
        self.features = nn.Sequential(*units)
        self.pool = pooling if pooling is not None else nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 10))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
