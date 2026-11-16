"""Compare custom layers and residual branches with matched settings."""
from torch import nn
from .models.cnn_models import CNN
from .models.custom_layers import GatedConv, ChannelAttention, SmoothActivation, MixedPool
from .utils.comparison_utils import compare
from .utils.training_utils import setup


def convolution(activation):
    """Build a first convolution followed by the supplied activation."""
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), activation)


def main():
    """Train standard and custom CNNs plus three residual block variants."""
    setup()
    compare('custom', 'mnist', {
        'standard': lambda: CNN(1),
        'gated-conv': lambda: CNN(1, layer=nn.Sequential(GatedConv(1, 16), nn.ReLU())),
        'attention': lambda: CNN(1, layer=nn.Sequential(convolution(nn.ReLU()), ChannelAttention(16))),
        'smooth': lambda: CNN(1, layer=convolution(SmoothActivation())),
        'silu': lambda: CNN(1, layer=convolution(nn.SiLU())),
        'mixed-pool': lambda: CNN(1, pooling=MixedPool()),
        'max-pool': lambda: CNN(1, pooling=nn.AdaptiveMaxPool2d(1)),
    })
    compare('residual', 'cifar', {
        k: (lambda k=k: CNN(3, residual=2, kind=k)) for k in ('basic', 'bottleneck', 'wide')
    })


if __name__ == '__main__':
    main()
