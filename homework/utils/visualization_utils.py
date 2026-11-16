"""Save training curves, confusion matrices, gradient flow and feature maps."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .training_utils import ROOT


def save(fig, name):
    """Export a figure to the homework plots directory."""
    folder = ROOT / 'plots'
    folder.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / (name + '.png'), dpi=125)
    plt.close(fig)


def curves(rows, name):
    """Compare accuracy and validation loss by epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for label, row in rows.items():
        for ax, (split, metric) in zip(axes, [('train', 'accuracy'), ('val', 'accuracy'),
                                             ('val', 'loss')]):
            history = row['history']
            ax.plot(range(1, 1 + len(history)), [e[split][metric] for e in history], label=label)
            ax.set(xlabel='Epoch', title=f'{split} {metric}')
    axes[-1].legend(fontsize=7)
    save(fig, name)


def confusion(rows, name):
    """Plot test confusion matrices using class IDs on both axes."""
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 4), squeeze=False)
    for ax, (label, row) in zip(axes[0], rows.items()):
        ax.imshow(row['test']['confusion'], cmap='Blues')
        ax.set(xlabel='Predicted', ylabel='Observed', title=label,
               xticks=range(10), yticks=range(10))
    save(fig, name + '_confusion')


def gradient_flow(rows, name):
    """Plot final mean per-layer gradient RMS on a logarithmic axis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, row in rows.items():
        values = list(row['history'][-1]['train']['gradients'].values())
        ax.plot(range(1, len(values) + 1), values, marker='.', label=label)
    ax.set(xlabel='Weight tensor in model order', ylabel='Gradient RMS', yscale='log')
    ax.legend(fontsize=7)
    save(fig, name + '_gradients')


def maps(x, title):
    """Show the first six channels of a single held-out image."""
    fig, axes = plt.subplots(1, min(6, x.shape[0]), squeeze=False, figsize=(9, 2))
    for ax, channel in zip(axes[0], x):
        ax.imshow(channel, cmap='viridis')
        ax.axis('off')
    save(fig, title)
