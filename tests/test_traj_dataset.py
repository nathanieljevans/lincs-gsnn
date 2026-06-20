"""Tests for :mod:`lincs_gsnn.data.TrajDataset`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lincs_gsnn.data.TrajDataset import TrajDataset
from lincs_gsnn.data.pred_store import STAT_MEAN, STAT_STD


def _build_pred_dir(tmp_path, n_rows=2, t=6, g=3):
    pred_dir = tmp_path / "predict_grid"
    pred_dir.mkdir()
    obs = np.zeros((n_rows, t, 2, g), dtype=np.float16)
    for r in range(n_rows):
        obs[r, :, STAT_MEAN, :] = r + 1
        obs[r, :, STAT_STD, :] = 0.0
    np.save(pred_dir / "obs.npy", obs)
    np.save(pred_dir / "dxdt.npy", np.zeros_like(obs))
    rows = []
    for r in range(n_rows):
        rows.append({
            "obs_row": r,
            "pert_id": "P0",
            "cell_iname": "C0",
            "dose": 10.0,
        })
    meta = pd.DataFrame(rows)
    input_names = ["DRUG__P0", "LINE__C0"] + [f"GENE__G{i}" for i in range(g)]
    return str(pred_dir), meta, input_names, obs


def test_traj_dataset_channels_and_horizon(tmp_path):
    pred_dir, meta, input_names, obs = _build_pred_dir(tmp_path)
    ds = TrajDataset(
        meta=meta,
        input_names=input_names,
        pred_dir=pred_dir,
        horizon=4,
        sigma_floor=1e-3,
    )
    obs_mu, x, obs_sigma = ds[0]
    assert obs_mu.shape == (4, 3)
    assert obs_sigma.shape == (4, 3)
    assert float(obs_mu[0, 0]) == float(obs[0, 0, STAT_MEAN, 0])
    assert (obs_sigma >= 1e-3).all()
    gene_ixs = [i for i, n in enumerate(input_names) if n.startswith("GENE__")]
    assert torch.allclose(x[gene_ixs], obs_mu[0])


def test_multiple_shooting_t0_in_range(tmp_path):
    pred_dir, meta, input_names, _ = _build_pred_dir(tmp_path, t=8)
    ds = TrajDataset(
        meta=meta.head(1),
        input_names=input_names,
        pred_dir=pred_dir,
        horizon=3,
        multiple_shooting=True,
    )
    torch.manual_seed(0)
    obs_mu, _, _ = ds[0]
    assert obs_mu.shape[0] == 3
