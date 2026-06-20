"""Tests for agg_edge_scores and agg_node_scores."""

from __future__ import annotations

import pandas as pd
import pytest

from lincs_gsnn.explain.eval import agg_edge_scores, agg_node_scores


def test_agg_edge_scores_model_replicates(tmp_path):
    csv_path = tmp_path / "edge.csv"
    pd.DataFrame({
        "source": ["A", "B"],
        "target": ["B", "C"],
        "gsnn_score_model_0": [0.9, 0.1],
        "gsnn_score_model_1": [0.7, 0.3],
        "ig_score_model_0": [1.0, -1.0],
        "ig_score_model_1": [0.0, 0.0],
        "occlusion_score_model_0": [0.2, -0.2],
        "occlusion_score_model_1": [0.4, -0.4],
    }).to_csv(csv_path, index=False)

    agg = agg_edge_scores(str(csv_path))
    row = agg.set_index(["source", "target"]).loc[("A", "B")]

    assert row["mean_gsnn_score"] == 0.8
    assert row["mean_ig_score"] == 0.5
    assert row["mean_oc_score"] == pytest.approx(0.3)
    assert row["abs_ig_score"] == 0.5
    assert row["abs_oc_score"] == pytest.approx(0.3)


def test_agg_edge_scores_sample_replicates(tmp_path):
    csv_path = tmp_path / "edge_legacy.csv"
    pd.DataFrame({
        "source": ["A"],
        "target": ["B"],
        "gsnn_score_sample_0": [0.6],
        "ig_score_sample_0": [2.0],
        "occlusion_score_sample_0": [-1.0],
    }).to_csv(csv_path, index=False)

    agg = agg_edge_scores(str(csv_path))
    row = agg.set_index(["source", "target"]).loc[("A", "B")]

    assert row["mean_gsnn_score"] == 0.6
    assert row["mean_ig_score"] == 2.0
    assert row["mean_oc_score"] == -1.0


def test_agg_node_scores_model_replicates(tmp_path):
    csv_path = tmp_path / "node.csv"
    pd.DataFrame({
        "node": ["N1", "N2"],
        "gsnn_score_model_0": [0.9, 0.2],
        "gsnn_score_model_1": [0.7, 0.4],
        "ig_score_model_0": [1.0, 0.0],
        "occlusion_score_model_0": [0.5, -0.5],
    }).to_csv(csv_path, index=False)

    agg = agg_node_scores(str(csv_path))
    row = agg.set_index("node").loc["N1"]

    assert row["mean_gsnn_score"] == 0.8
    assert row["mean_ig_score"] == 1.0
    assert row["mean_oc_score"] == 0.5
