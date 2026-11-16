"""Measure kernel sizes and depth while recording activations and gradients."""
from .models.cnn_models import CNN
from .utils.comparison_utils import compare
from .utils.training_utils import setup


def main():
    """Run the equal-budget kernel grid and convolutional depth grid."""
    setup()
    shapes = {'k3': ((11, 64), (3, 3)), 'k5': ((11, 24), (5, 5)),
              'k7': ((11, 12), (7, 7)), 'k1-k3': ((33, 23), (1, 3))}
    models = {name: (lambda w=w, k=k: CNN(1, w, k)) for name, (w, k) in shapes.items()}
    compare('kernel', 'mnist', models, show_maps='first',
            fields={name: sum(k) for name, (_, k) in shapes.items()})
    compare('depth', 'mnist', {
        **{str(n): (lambda n=n: CNN(1, (16,) * n, (3,) * n)) for n in (2, 4, 6)},
        'residual6': lambda: CNN(1, (16, 16), residual=2),
    }, show_maps='last')


if __name__ == '__main__':
    main()
