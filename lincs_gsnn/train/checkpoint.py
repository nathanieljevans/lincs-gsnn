"""Checkpoint and resume helpers for GSNN training scripts."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import torch

from lincs_gsnn.train.optim import load_optimizer_state, optimizer_to_state


def checkpoint_dir(out_dir: str, model_id: str) -> str:
    return os.path.join(out_dir, 'checkpoints', model_id)


def last_model_path(out_dir: str, model_id: str) -> str:
    return os.path.join(checkpoint_dir(out_dir, model_id), 'last_model.pt')


def best_model_path(out_dir: str, model_id: str) -> str:
    return os.path.join(checkpoint_dir(out_dir, model_id), 'best_model.pt')


def last_optim_path(out_dir: str, model_id: str) -> str:
    return os.path.join(checkpoint_dir(out_dir, model_id), 'last_optim.pt')


def last_sched_path(out_dir: str, model_id: str) -> str:
    return os.path.join(checkpoint_dir(out_dir, model_id), 'last_sched.pt')


def train_state_path(out_dir: str, model_id: str) -> str:
    return os.path.join(checkpoint_dir(out_dir, model_id), 'train_state.json')


def infer_start_epoch(state: dict[str, Any] | None) -> int:
    if not state:
        return 0
    return int(state.get('last_epoch', -1)) + 1


def load_train_state(out_dir: str, model_id: str) -> dict[str, Any] | None:
    path = train_state_path(out_dir, model_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def save_train_state(
    out_dir: str,
    model_id: str,
    *,
    last_epoch: int,
    best_epoch: int,
    best_val_nll: float,
    best_val_mse: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    ckpt_dir = checkpoint_dir(out_dir, model_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        'last_epoch': int(last_epoch),
        'best_epoch': int(best_epoch),
        'best_val_nll': float(best_val_nll),
    }
    if best_val_mse is not None:
        payload['best_val_mse'] = float(best_val_mse)
    if extra:
        payload.update(extra)
    with open(train_state_path(out_dir, model_id), 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)


def save_epoch_checkpoint(
    out_dir: str,
    model_id: str,
    *,
    model,
    optimizer,
    scheduler,
    last_epoch: int,
    best_epoch: int,
    best_val_nll: float,
    best_val_mse: float | None = None,
    save_best: bool = False,
    extra_state: dict[str, Any] | None = None,
) -> None:
    ckpt_dir = checkpoint_dir(out_dir, model_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model, last_model_path(out_dir, model_id))
    torch.save(optimizer_to_state(optimizer), last_optim_path(out_dir, model_id))
    torch.save(scheduler.state_dict(), last_sched_path(out_dir, model_id))
    save_train_state(
        out_dir,
        model_id,
        last_epoch=last_epoch,
        best_epoch=best_epoch,
        best_val_nll=best_val_nll,
        best_val_mse=best_val_mse,
        extra=extra_state,
    )
    if save_best:
        torch.save(model, best_model_path(out_dir, model_id))


def try_load_resume(
    out_dir: str,
    model_id: str,
    *,
    resume_incomplete: bool,
    device,
    model,
    optimizer,
    scheduler,
):
    """Return (model, start_epoch, best_val_nll, best_epoch, best_val_mse, state)."""
    state = load_train_state(out_dir, model_id) if resume_incomplete else None
    start_epoch = infer_start_epoch(state) if resume_incomplete else 0
    best_val_nll = float(state.get('best_val_nll', float('inf'))) if state else float('inf')
    best_epoch = int(state.get('best_epoch', -1)) if state else -1
    best_val_mse = state.get('best_val_mse') if state else None

    if resume_incomplete and os.path.exists(last_model_path(out_dir, model_id)):
        model = torch.load(
            last_model_path(out_dir, model_id),
            map_location=device,
            weights_only=False,
        )
        model = model.to(device)
        opt_path = last_optim_path(out_dir, model_id)
        if os.path.exists(opt_path):
            load_optimizer_state(
                optimizer,
                torch.load(opt_path, map_location='cpu', weights_only=False),
            )
        sched_path = last_sched_path(out_dir, model_id)
        if os.path.exists(sched_path):
            scheduler.load_state_dict(
                torch.load(sched_path, map_location='cpu', weights_only=False)
            )
    elif resume_incomplete and start_epoch > 0:
        raise FileNotFoundError(
            f'resume_incomplete: train_state indicates {start_epoch} completed epochs '
            f'but checkpoint missing at {last_model_path(out_dir, model_id)}'
        )

    return model, start_epoch, best_val_nll, best_epoch, best_val_mse, state


def load_best_model(out_dir: str, model_id: str, device):
    path = best_model_path(out_dir, model_id)
    if not os.path.exists(path):
        path = last_model_path(out_dir, model_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No best/last checkpoint found under {checkpoint_dir(out_dir, model_id)}'
        )
    return torch.load(path, map_location=device, weights_only=False)


def append_history_row(csv_path: str, row: dict[str, Any], columns: list[str]) -> None:
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col) for col in columns})
