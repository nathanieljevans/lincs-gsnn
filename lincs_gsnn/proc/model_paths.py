"""Resolve GSNN checkpoint paths for train / explain workflows."""

from __future__ import annotations

import os


def gsnn_model_path(root_gsnn: str, model_id: str, *, model_path: str | None = None) -> str:
    """Return the GSNN checkpoint path for a model replicate.

    When ``model_path`` is provided (e.g. from Snakemake ``input.model``), it is
    returned as-is. Otherwise prefer ``train/trained_model_{model}.pt`` when
    present, falling back to ``pretrain/pretrained_model_{model}.pt``.
    """
    if model_path:
        return model_path
    trained = os.path.join(root_gsnn, "train", f"trained_model_{model_id}.pt")
    if os.path.isfile(trained):
        return trained
    return os.path.join(root_gsnn, "pretrain", f"pretrained_model_{model_id}.pt")
