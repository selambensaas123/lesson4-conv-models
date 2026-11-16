"""Deterministic comparisons on disjoint subsets of official image datasets."""
import logging
import os
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, CIFAR10

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    """Use training indices for train and validation, official test for test."""
    source = MNIST if name == 'mnist' else CIFAR10
    root = os.environ.get('LESSON_DATA_ROOT', 'data')
    train, test = [source(root, train=flag, download=True) for flag in (True, False)]
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train), generator=generator)
    test_ids = torch.randperm(len(test), generator=generator)

    def subset(dataset, ids):
        images = torch.as_tensor(dataset.data)[ids]
        images = images.unsqueeze(1) if name == 'mnist' else images.permute(0, 3, 1, 2)
        labels = torch.as_tensor(dataset.targets)[ids]
        return TensorDataset(images.contiguous().float().div_(255), labels)

    return subset(train, indices[:4096]), subset(train, indices[4096:5120]), subset(test, test_ids[:2048])


def epoch(model, loader, optimizer=None):
    """Measure sample-weighted loss, correct predictions and gradient RMS."""
    model.train(optimizer is not None)
    loss_sum = count = correct = 0
    matrix = torch.zeros(100, dtype=torch.long)
    tracked = [(name, p) for name, p in model.named_parameters() if p.ndim >= 2]
    gradients = dict.fromkeys((name for name, _ in tracked), 0.)
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            prediction = model(images)
            loss = nn.functional.cross_entropy(prediction, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in tracked:
                    gradients[name] += parameter.grad.square().mean().sqrt().item()
                optimizer.step()
            predicted = prediction.argmax(1)
            matrix += torch.bincount(10 * labels + predicted, minlength=100)
            count += len(labels)
            correct += (labels == predicted).sum().item()
            loss_sum += loss.item() * len(labels)
    return dict(loss=loss_sum / count, accuracy=correct / count,
                confusion=matrix.reshape(10, 10).tolist(),
                gradients={key: value / len(loader) for key, value in gradients.items()}
                if optimizer is not None else {})


def train(factory, splits, epochs=5):
    """Use identical Adam settings and batch order for each model."""
    torch.manual_seed(42)
    model = factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loaders = [DataLoader(split, batch_size=64, shuffle=index == 0,
                          generator=torch.Generator().manual_seed(42))
               for index, split in enumerate(splits)]
    train_loader, val_loader, test_loader = loaders
    history, seconds = [], 0.
    for index in range(epochs):
        start = perf_counter()
        train_stats = epoch(model, train_loader, optimizer)
        seconds += perf_counter() - start
        val = epoch(model, val_loader)
        history.append(dict(train=train_stats, val=val))
        logging.info('Epoch %d train=%.4f val=%.4f', index + 1,
                     train_stats['accuracy'], val['accuracy'])
    test = epoch(model, test_loader)
    inputs = splits[2].tensors[0]
    with torch.no_grad():
        model(inputs[:64])
        start = perf_counter()
        for images in inputs.split(64):
            model(images)
        inference = perf_counter() - start
    return model, dict(history=history, test=test, seconds=seconds, inference=inference,
                       params=sum(p.numel() for p in model.parameters()))


def setup():
    """Keep CPU thread count fixed and record training progress."""
    torch.set_num_threads(4)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
