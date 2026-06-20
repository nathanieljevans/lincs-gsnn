"""Tests for :mod:`lincs_gsnn.data.pred_store`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from lincs_gsnn.data.pred_store import (
    STAT_MEAN,
    STAT_STD,
    get_pred_dxdt,
    get_pred_trajectory,
    open_pred_dxdt,
    open_pred_obs,
)


def _write_synthetic_pred_dir(tmp_path, n=3, t=4, g=5):
    pred_dir = tmp_path / "predict_grid"
    pred_dir.mkdir()

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, t, 2, g)).astype(np.float16)
    dxdt = (rng.standard_normal((n, t, 2, g)) + 2.0).astype(np.float16)
    # make mean != std channels deterministic for assertions
    obs[..., STAT_MEAN, :] = np.arange(n * t * g, dtype=np.float16).reshape(n, t, g)
    obs[..., STAT_STD, :] = obs[..., STAT_MEAN, :] + 1.0
    dxdt[..., STAT_MEAN, :] = np.arange(n * t * g, dtype=np.float16).reshape(n, t, g) * 0.1
    dxdt[..., STAT_STD, :] = dxdt[..., STAT_MEAN, :] + 0.5

    np.save(pred_dir / "obs.npy", obs)
    np.save(pred_dir / "dxdt.npy", dxdt)

    pd.DataFrame({"gene_names": [f"G{i}" for i in range(g)]}).to_csv(
        pred_dir / "gene_names.csv", index=False
    )
    return pred_dir, obs, dxdt


def test_open_pred_arrays_shapes_and_dtype(tmp_path):
    pred_dir, obs, dxdt = _write_synthetic_pred_dir(tmp_path)
    obs_mm = open_pred_obs(str(pred_dir))
    dxdt_mm = open_pred_dxdt(str(pred_dir))
    assert obs_mm.shape == obs.shape == (3, 4, 2, 5)
    assert dxdt_mm.shape == dxdt.shape == (3, 4, 2, 5)
    assert obs_mm.dtype == np.float16


def test_get_pred_trajectory_mean_vs_std(tmp_path):
    pred_dir, obs, _ = _write_synthetic_pred_dir(tmp_path)
    mu = get_pred_trajectory(obs, obs_row=1, stat="mean")
    sigma = get_pred_trajectory(obs, obs_row=1, stat="std")
    assert mu.shape == (4, 5)
    assert sigma.shape == (4, 5)
    assert not torch.allclose(mu, sigma)
    assert float(mu[0, 0]) == float(obs[1, 0, STAT_MEAN, 0])
    assert float(sigma[0, 0]) == float(obs[1, 0, STAT_STD, 0])


def test_get_pred_dxdt_vector(tmp_path):
    pred_dir, _, dxdt = _write_synthetic_pred_dir(tmp_path)
    mu = get_pred_dxdt(dxdt, obs_row=2, time_idx=3, stat="mean")
    sigma = get_pred_dxdt(dxdt, obs_row=2, time_idx=3, stat="std")
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
    assert float(mu[0]) == float(dxdt[2, 3, STAT_MEAN, 0])
    assert float(sigma[0]) == float(dxdt[2, 3, STAT_STD, 0])


def test_missing_array_raises(tmp_path):
    pred_dir = tmp_path / "empty"
    pred_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        open_pred_obs(str(pred_dir))


def test_bad_stat_raises(tmp_path):
    pred_dir, obs, _ = _write_synthetic_pred_dir(tmp_path)
    with pytest.raises(ValueError, match="stat"):
        get_pred_trajectory(obs, obs_row=0, stat="median")
