"""Tests for GSNN checkpoint path resolution."""

from __future__ import annotations

import os

import pytest

from lincs_gsnn.proc.model_paths import gsnn_model_path


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "exp_37"
    (root / "pretrain").mkdir(parents=True)
    (root / "train").mkdir(parents=True)
    return root


def test_explicit_path_wins(run_root):
    explicit = str(run_root / "custom" / "model.pt")
    trained = run_root / "train" / "trained_model_model_0.pt"
    trained.write_bytes(b"x")
    assert gsnn_model_path(str(run_root), "model_0", model_path=explicit) == explicit


def test_prefers_trained_when_present(run_root):
    model_id = "model_0"
    (run_root / "pretrain" / f"pretrained_model_{model_id}.pt").write_bytes(b"p")
    trained = run_root / "train" / f"trained_model_{model_id}.pt"
    trained.write_bytes(b"t")
    assert gsnn_model_path(str(run_root), model_id) == str(trained)


def test_falls_back_to_pretrain(run_root):
    model_id = "model_0"
    pretrain = run_root / "pretrain" / f"pretrained_model_{model_id}.pt"
    pretrain.write_bytes(b"p")
    assert gsnn_model_path(str(run_root), model_id) == str(pretrain)


def test_returns_pretrain_path_when_neither_exists(run_root):
    model_id = "model_0"
    expected = os.path.join(str(run_root), "pretrain", f"pretrained_model_{model_id}.pt")
    assert gsnn_model_path(str(run_root), model_id) == expected
