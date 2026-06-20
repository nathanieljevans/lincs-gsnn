"""Optimizer factory for GSNN training scripts."""

from __future__ import annotations

import torch

OPTIMIZER_CHOICES = ('adam', 'adamw', 'muon', 'rmsprop')


class MultiOptimizer:
    """Legacy wrapper for resume checkpoints built before MuonWithAuxAdam.

    New muon runs use :class:`muon.SingleDeviceMuonWithAuxAdam`, which is a
    single ``torch.optim.Optimizer`` with ``use_muon`` param groups.
    """

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt in self.optimizers:
            opt.step()


def scale_optimizer_lr(optimizer, factor):
    """Multiply lr in every param group (works for MultiOptimizer and Muon)."""
    for group in optimizer.param_groups:
        group['lr'] *= factor


def get_optimizer_lrs(optimizer):
    """Return current learning rates for logging / resume reconstruction."""
    return [float(group['lr']) for group in optimizer.param_groups]


def build_lr_scheduler(
    optimizer,
    *,
    patience,
    factor=0.5,
    threshold=1e-3,
):
    """Build a validation-metric plateau LR scheduler for the given optimizer."""
    if isinstance(optimizer, MultiOptimizer):
        raise TypeError(
            'MultiOptimizer is legacy-only; rebuild with optimizer="muon" to get '
            'SingleDeviceMuonWithAuxAdam before attaching a scheduler.'
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        threshold=threshold,
    )


def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _is_muon_param(name, param):
    if param.ndim != 2:
        return False
    if (
        name.endswith('_emb.weight')
        or name.endswith('.bias')
        or 'norm' in name.lower()
    ):
        return False
    return True


def _get_muon_with_aux_adam_cls():
    try:
        from muon import SingleDeviceMuonWithAuxAdam
    except ImportError as exc:
        raise ImportError(
            'optimizer="muon" requires the muon package: '
            'pip install git+https://github.com/KellerJordan/Muon'
        ) from exc
    return SingleDeviceMuonWithAuxAdam


def _build_adamw(model, lr, wd):
    return torch.optim.AdamW(_trainable_params(model), lr=lr, weight_decay=wd)


def _build_muon(model, lr, wd):
    """Build Muon + AdamW via the official single-optimizer param-group API."""
    muon_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_param(name, param):
            muon_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if muon_params:
        param_groups.append(
            dict(params=muon_params, lr=lr, weight_decay=wd, use_muon=True)
        )
    if other_params:
        param_groups.append(
            dict(params=other_params, lr=lr, weight_decay=wd, use_muon=False)
        )

    if not param_groups:
        raise ValueError('No trainable parameters found for optimizer.')

    return _get_muon_with_aux_adam_cls()(param_groups)


def build_optimizer(
    model,
    optimizer_type='adamw',
    lr=1e-4,
    wd=0.0,
):
    opt = optimizer_type.lower()
    if opt not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"Unknown optimizer {optimizer_type!r}. "
            f"Choose from: {', '.join(OPTIMIZER_CHOICES)}."
        )

    if opt == 'adamw':
        return _build_adamw(model, lr, wd)
    if opt == 'adam':
        return torch.optim.Adam(_trainable_params(model), lr=lr, weight_decay=wd)
    if opt == 'rmsprop':
        return torch.optim.RMSprop(_trainable_params(model), lr=lr, weight_decay=wd)
    if opt == 'muon':
        return _build_muon(model, lr, wd)

    raise ValueError(
        f"Unknown optimizer {optimizer_type!r}. "
        f"Choose from: {', '.join(OPTIMIZER_CHOICES)}."
    )


def optimizer_to_state(optimizer):
    if isinstance(optimizer, MultiOptimizer):
        return {'multi': True, 'states': [opt.state_dict() for opt in optimizer.optimizers]}
    return {'multi': False, 'states': [optimizer.state_dict()]}


def load_optimizer_state(optimizer, blob):
    states = blob['states']
    if isinstance(optimizer, MultiOptimizer):
        if len(states) != len(optimizer.optimizers):
            raise ValueError(
                f'optimizer count mismatch: saved {len(states)}, '
                f'current {len(optimizer.optimizers)}'
            )
        for opt, state in zip(optimizer.optimizers, states):
            opt.load_state_dict(state)
    else:
        optimizer.load_state_dict(states[0])
