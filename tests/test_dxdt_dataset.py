"""Tests for :mod:`lincs_gsnn.data.DXDTDataset`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from lincs_gsnn.data.DXDTDataset import DXDTDataset
from lincs_gsnn.data.pred_store import STAT_MEAN, STAT_STD


def _build_pred_dir(tmp_path, n_pert=2, n_cell=2, n_time=2, g=4):
    pred_dir = tmp_path / "predict_grid"
    pred_dir.mkdir()
    n_cond = n_pert * n_cell
    t = n_time
    obs = np.zeros((n_cond, t, 2, g), dtype=np.float16)
    dxdt = np.zeros((n_cond, t, 2, g), dtype=np.float16)
    for c in range(n_cond):
        obs[c, :, STAT_MEAN, :] = c + 1
        obs[c, :, STAT_STD, :] = 0.0
        dxdt[c, :, STAT_MEAN, :] = (c + 1) * 0.1
        dxdt[c, :, STAT_STD, :] = 0.0
    np.save(pred_dir / "obs.npy", obs)
    np.save(pred_dir / "dxdt.npy", dxdt)
    genes = [f"G{i}" for i in range(g)]
    pd.DataFrame({"gene_names": genes}).to_csv(pred_dir / "gene_names.csv", index=False)

    rows = []
    obs_row = 0
    for pert in ["P0", "P1"][:n_pert]:
        for cell in ["C0", "C1"][:n_cell]:
            for time_idx in range(n_time):
                rows.append({
                    "obs_row": obs_row,
                    "time_idx": time_idx,
                    "pert_id": pert,
                    "cell_iname": cell,
                    "dose": 10.0,
                    "time": float(time_idx),
                })
            obs_row += 1
    meta = pd.DataFrame(rows)
    return str(pred_dir), genes, meta


def _input_output_names(genes):
    input_names = ["DRUG__P0", "DRUG__P1", "LINE__C0", "LINE__C1"] + [f"GENE__{g}" for g in genes]
    output_names = [f"GENE__{g}" for g in genes]
    return input_names, output_names


def test_dxdt_dataset_layout_and_sigma_floor(tmp_path):
    pred_dir, genes, meta = _build_pred_dir(tmp_path)
    input_names, output_names = _input_output_names(genes)
    scale = 2.0
    ds = DXDTDataset(
        meta=meta,
        input_names=input_names,
        output_names=output_names,
        src_names=genes,
        pred_dir=pred_dir,
        scale=scale,
        sigma_floor=1e-3,
    )
    X, mu, sigma = ds[0]
    assert X[input_names.index("LINE__C0")] == 1.0
    assert X[input_names.index("DRUG__P0")] > 0
    assert (sigma >= 1e-3).all()
    assert torch.allclose(mu, mu)  # finite
    assert mu.shape == (len(output_names),)
    assert sigma.shape == (len(output_names),)


def test_drops_missing_pert(tmp_path):
    pred_dir, genes, meta = _build_pred_dir(tmp_path)
    meta = pd.concat([
        meta,
        pd.DataFrame([{
            "obs_row": 0, "time_idx": 0, "pert_id": "MISSING",
            "cell_iname": "C0", "dose": 10.0, "time": 0.0,
        }]),
    ], ignore_index=True)
    input_names, output_names = _input_output_names(genes)
    ds = DXDTDataset(
        meta=meta,
        input_names=input_names,
        output_names=output_names,
        src_names=genes,
        pred_dir=pred_dir,
        scale=1.0,
    )
    assert len(ds) == len(meta) - 1


def test_return_time_appends_t(tmp_path):
    pred_dir, genes, meta = _build_pred_dir(tmp_path)
    input_names, output_names = _input_output_names(genes)
    ds = DXDTDataset(
        meta=meta.head(1),
        input_names=input_names,
        output_names=output_names,
        src_names=genes,
        pred_dir=pred_dir,
        scale=1.0,
        return_time=True,
    )
    out = ds[0]
    assert len(out) == 4
    assert out[-1].shape == (1,)
