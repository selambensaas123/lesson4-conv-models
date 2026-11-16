"""Run a family of models and persist comparison artifacts."""
import json
import logging

import torch
from .training_utils import ROOT, load, train
from .visualization_utils import curves, confusion, gradient_flow, maps


def compare(group, dataset, models, *, show_confusion=False, show_maps=None, fields=None):
    """Run each model on the same split and save numerical results and plots."""
    splits = load(dataset)
    rows = {}
    for name, factory in models.items():
        logging.info('%s / %s', group, name)
        model, rows[name] = train(factory, splits)
        if fields is not None:
            rows[name]['receptive_field'] = fields[name]
        if show_maps is not None:
            with torch.no_grad():
                layer = model.features[0] if show_maps == 'first' else model.features
                activations = layer(splits[2][0][0].unsqueeze(0))[0]
            maps(activations, f'{group}_{name}')
    folder = ROOT / 'results' / group
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    curves(rows, group)
    gradient_flow(rows, group)
    if show_confusion:
        confusion(rows, group)
