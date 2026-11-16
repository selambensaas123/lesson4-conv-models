"""Compare fully connected and convolutional models on MNIST and CIFAR-10."""
from convolutional_basics.models import SimpleCNN
from .models.cnn_models import CNN
from .models.fc_models import FC
from .utils.comparison_utils import compare
from .utils.training_utils import setup


def main():
    """Run both dataset comparisons with identical training settings."""
    setup()
    compare('mnist_comparison', 'mnist', {
        'FC': lambda: FC(1, 28), 'CNN': SimpleCNN,
        'Residual': lambda: CNN(1, residual=2),
    })
    compare('cifar_comparison', 'cifar', {
        'FC': lambda: FC(3, 32, deep=True),
        'Residual': lambda: CNN(3, residual=2),
        'Residual-dropout': lambda: CNN(3, residual=2, dropout=.35),
    }, show_confusion=True)


if __name__ == '__main__':
    main()
