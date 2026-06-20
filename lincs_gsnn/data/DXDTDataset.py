from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from DeepTraj.models.DeepTraj import dose_transform

from lincs_gsnn.data.pred_store import (
    STAT_MEAN,
    open_pred_obs,
    open_pred_dxdt,
    materialize_dxdt_rows,
)
from lincs_gsnn.utils.nll import clamp_sigma


class DXDTDataset(Dataset):
    """
    Dataset that returns (X, dxdt_mu, dxdt_sigma, [t], [x_fn]) tensors for
    training the GSNN on dX/dt targets from the consolidated predict_grid
    memmaps (obs.npy / dxdt.npy).
    """

    def __init__(
        self,
        meta,
        input_names,
        output_names,
        src_names,
        pred_dir: str = "",
        scale: Optional[float] = None,
        sigma_floor: float = 1e-4,
        return_time: bool = False,
        x_fn_lookup: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.x_fn_lookup = None
        if x_fn_lookup is not None:
            self.x_fn_lookup = {}
            for k, v in x_fn_lookup.items():
                t = torch.as_tensor(v, dtype=torch.float32)
                if t.dim() == 1:
                    t = t.unsqueeze(-1)
                self.x_fn_lookup[k] = t.contiguous()

            n_before = len(meta)
            present = meta["cell_iname"].isin(self.x_fn_lookup.keys())
            dropped_inames = sorted(set(meta.loc[~present, "cell_iname"].astype(str).tolist()))
            meta = meta.loc[present].reset_index(drop=True)
            if dropped_inames:
                sample = dropped_inames[:5]
                warnings.warn(
                    f"DXDTDataset: dropped {n_before - len(meta)} sample(s) "
                    f"covering {len(dropped_inames)} cell_iname(s) absent from the "
                    f"x_fn_lookup (showing up to 5): {sample}.",
                    RuntimeWarning,
                )

        drug_nodes = {n.split("__", 1)[1] for n in input_names if n.startswith("DRUG__")}
        n_before = len(meta)
        meta = meta[meta["pert_id"].astype(str).isin(drug_nodes)].reset_index(drop=True)
        if len(meta) < n_before:
            warnings.warn(
                f"DXDTDataset: dropped {n_before - len(meta)} row(s) whose pert_id "
                f"is absent from input_names DRUG__ nodes.",
                RuntimeWarning,
            )

        required = {"obs_row", "time_idx", "pert_id", "dose", "cell_iname"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"DXDTDataset meta missing columns: {sorted(missing)}")

        self.meta = meta.reset_index(drop=True)
        self.pred_dir = pred_dir
        self.input_names = input_names
        self.output_names = output_names
        self.sigma_floor = float(sigma_floor)
        self.return_time = return_time

        self.src_names = src_names
        src_ixs, dst_ixs, dst2_ixs = [], [], []
        for i_src, name in enumerate(src_names):
            gene_sym = name.split("__")[1] if name.startswith("GENE__") else name
            gene_node = f"GENE__{gene_sym}"
            try:
                i_dst = self.input_names.index(gene_node)
                i_dst2 = self.output_names.index(gene_node)
            except ValueError:
                continue
            src_ixs.append(i_src)
            dst_ixs.append(i_dst)
            dst2_ixs.append(i_dst2)
        self.src_ixs = torch.tensor(src_ixs, dtype=torch.long)
        self.dst_ixs = torch.tensor(dst_ixs, dtype=torch.long)
        self.dst2_ixs = torch.tensor(dst2_ixs, dtype=torch.long)

        self._obs = open_pred_obs(pred_dir)
        self._dxdt = open_pred_dxdt(pred_dir)

        obs_rows = self.meta["obs_row"].to_numpy(dtype=np.int64)
        time_idxs = self.meta["time_idx"].to_numpy(dtype=np.int64)

        x_genes = np.empty((len(self.meta), self._obs.shape[-1]), dtype=np.float32)
        for i, (obs_row, time_idx) in enumerate(zip(obs_rows, time_idxs)):
            x_genes[i] = self._obs[obs_row, time_idx, STAT_MEAN, :]
        self._x_genes = torch.from_numpy(x_genes)

        mu_arr, sigma_arr = materialize_dxdt_rows(self._obs, self._dxdt, obs_rows, time_idxs)
        self._dxdt_mu = torch.from_numpy(mu_arr)
        self._dxdt_sigma = torch.from_numpy(sigma_arr)

        if scale is None:
            self._scale_value = self.estimate_dxdt_std(n_samples=min(10000, len(self)))
        else:
            self._scale_value = float(scale)

        self._dxdt_mu = self._dxdt_mu / self._scale_value
        self._dxdt_sigma = clamp_sigma(self._dxdt_sigma / self._scale_value, self.sigma_floor)

    @property
    def _scale(self):
        return self._scale_value

    def estimate_dxdt_std(self, n_samples: int = 250) -> float:
        n = min(n_samples, len(self))
        if n == 0:
            raise ValueError("DXDTDataset is empty; cannot estimate scale")
        idxs = torch.randint(0, len(self), (n,))
        samples = self._dxdt_mu[idxs].view(n, -1)
        scale = samples.std().item()
        assert scale > 0, f"scale is {scale}"
        return scale

    def get(self, idx: int):
        row = self.meta.iloc[idx]
        x_gene = self._x_genes[idx]

        X = torch.zeros(len(self.input_names), dtype=torch.float32)
        drug_ix = self.input_names.index("DRUG__" + str(row["pert_id"]))
        line_ix = self.input_names.index("LINE__" + str(row["cell_iname"]))
        X[drug_ix] = dose_transform(torch.tensor([row["dose"]], dtype=torch.float32))
        X[line_ix] = 1.0
        X[self.dst_ixs] = x_gene[self.src_ixs]

        dxdt_mu = torch.zeros(len(self.output_names), dtype=torch.float32)
        dxdt_sigma = torch.zeros(len(self.output_names), dtype=torch.float32)
        mu_g = self._dxdt_mu[idx]
        sig_g = self._dxdt_sigma[idx]
        dxdt_mu[self.dst2_ixs] = mu_g[self.src_ixs]
        dxdt_sigma[self.dst2_ixs] = sig_g[self.src_ixs]

        t = torch.tensor([row["time"]], dtype=torch.float32) if "time" in row else torch.tensor([float(row["time_idx"])])
        return X.clone().detach(), dxdt_mu.clone().detach(), dxdt_sigma.clone().detach(), t.clone().detach()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int):
        X, dxdt_mu, dxdt_sigma, t = self.get(idx)

        assert not torch.isnan(X).any(), f"X has nans at index {idx}"
        assert not torch.isnan(dxdt_mu).any(), f"dxdt_mu has nans at index {idx}"
        assert not torch.isnan(dxdt_sigma).any(), f"dxdt_sigma has nans at index {idx}"

        if self.x_fn_lookup is not None:
            row = self.meta.iloc[idx]
            x_fn = self.x_fn_lookup[row["cell_iname"]]
            if self.return_time:
                return X, dxdt_mu, dxdt_sigma, t, x_fn
            return X, dxdt_mu, dxdt_sigma, x_fn

        return (X, dxdt_mu, dxdt_sigma, t) if self.return_time else (X, dxdt_mu, dxdt_sigma)
