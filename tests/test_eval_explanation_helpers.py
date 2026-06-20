"""Tests for eval_explanation helper functions."""

from __future__ import annotations

import json
import os

import pandas as pd

from workflow.scripts.eval_explanation import (
    _aggregate_val_metrics,
    _count_models_in_agg_csv,
    _discover_models,
)


def test_discover_models(tmp_path):
    pretrain = tmp_path / "pretrain"
    pretrain.mkdir()
    for mid, nll in [("model_0", 1.2), ("model_1", 0.8)]:
        with open(pretrain / f"val_metrics_pretrain_{mid}.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_nll": nll, "best_val_mse": 0.1, "best_val_r2": 0.5}, f)
    assert _discover_models(str(pretrain)) == ["model_0", "model_1"]


def test_count_models_in_agg_csv(tmp_path):
    csv_path = tmp_path / "agg.csv"
    pd.DataFrame(columns=["source", "target", "gsnn_score_model_0", "gsnn_score_model_1"]).to_csv(
        csv_path, index=False
    )
    assert _count_models_in_agg_csv(str(csv_path)) == 2


def test_aggregate_val_metrics_includes_nll(tmp_path):
    pretrain = tmp_path / "pretrain"
    pretrain.mkdir()
    for mid, nll in [("model_0", 1.0), ("model_1", 3.0)]:
        with open(pretrain / f"val_metrics_pretrain_{mid}.json", "w", encoding="utf-8") as f:
            json.dump({"best_val_nll": nll, "best_val_mse": 0.2, "best_val_r2": 0.4}, f)
    agg = _aggregate_val_metrics(str(pretrain), "val_metrics_pretrain", ["model_0", "model_1"])
    assert agg["mean_best_val_nll"] == 2.0
    assert "model_0" in agg["per_model"]
