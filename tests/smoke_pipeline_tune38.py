"""Integration smoke for config_exp_37 / tune_37 predict_grid (dry-run)."""

from __future__ import annotations

import os
import subprocess

import pytest
import yaml

CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "workflow",
    "train",
    "configs",
    "config_exp_37.yaml",
)
SNAKEFILE = os.path.join(os.path.dirname(__file__), "..", "workflow", "train", "Snakefile")


def _preds_path():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["dirs"]["preds"]


@pytest.mark.skipif(not os.path.isfile(_preds_path() + "/obs.npy"), reason="tune_37 predict_grid absent")
def test_snakemake_dry_run_resolves_model_dag():
    cmd = [
        "snakemake",
        "-n",
        "-s",
        SNAKEFILE,
        "--configfile",
        CONFIG,
        "-j1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "model_0" in proc.stdout
    assert "obs.npy" in proc.stdout
