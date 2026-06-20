"""Tests for aggregate_contrastive_results model directory discovery."""

from __future__ import annotations

import os

import pandas as pd
import torch

from workflow.scripts.aggregate_contrastive_results import (
    aggregate_out_dicts,
    find_model_dirs,
    pivot_long_to_wide,
)


def _write_model_result(root, model_id, score=0.5):
    model_dir = os.path.join(root, model_id)
    os.makedirs(model_dir, exist_ok=True)
    df = pd.DataFrame({
        "source": ["A"],
        "target": ["B"],
        "gsnn_score": [score],
    })
    df.to_csv(os.path.join(model_dir, f"contrastive_results_{model_id}.csv"), index=False)
    torch.save({"k": model_id}, os.path.join(model_dir, f"contrastive_results_{model_id}.pt"))


def test_find_model_dirs(tmp_path):
    _write_model_result(tmp_path, "model_0")
    _write_model_result(tmp_path, "model_1")
    found = find_model_dirs(str(tmp_path))
    assert [os.path.basename(p) for p in found] == ["model_0", "model_1"]


def test_pivot_suffixes_use_model_ids(tmp_path):
    long_csv = tmp_path / "long.csv"
    pd.DataFrame({
        "source": ["A", "A"],
        "target": ["B", "B"],
        "sample_id": ["model_0", "model_1"],
        "gsnn_score": [0.1, 0.2],
    }).to_csv(long_csv, index=False)
    wide_csv = tmp_path / "wide.csv"
    pivot_long_to_wide(str(long_csv), str(wide_csv), ["source", "target"])
    wide = pd.read_csv(wide_csv)
    assert "gsnn_score_model_0" in wide.columns
    assert "gsnn_score_model_1" in wide.columns


def test_aggregate_out_dict_keys(tmp_path):
    _write_model_result(tmp_path, "model_0", 0.1)
    _write_model_result(tmp_path, "model_1", 0.2)
    out_pt = tmp_path / "agg.pt"
    n = aggregate_out_dicts(find_model_dirs(str(tmp_path)), str(out_pt))
    assert n == 2
    loaded = torch.load(out_pt, weights_only=False)
    assert set(loaded.keys()) == {"model_0", "model_1"}
