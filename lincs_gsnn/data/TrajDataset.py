from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset

from DeepTraj.models.DeepTraj import dose_transform

from lincs_gsnn.data.pred_store import open_pred_obs, materialize_trajectories
from lincs_gsnn.utils.nll import clamp_sigma


class TrajDataset(Dataset):
    """
    Dataset that returns (obs_mu, x, obs_sigma, [x_fn]) tensors for training/
    evaluating the GSNN against observed trajectories from predict_grid memmaps.
    """

    def __init__(
        self,
        meta,
        input_names,
        pred_dir: str = "",
        horizon: Optional[int] = None,
        multiple_shooting: bool = False,
        sigma_floor: float = 1e-4,
        x_fn_lookup: Optional[Dict[str, torch.Tensor]] = None,
        src_names: Optional[Sequence[str]] = None,
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
                    f"TrajDataset: dropped {n_before - len(meta)} sample(s) "
                    f"covering {len(dropped_inames)} cell_iname(s) absent from the "
                    f"x_fn_lookup (showing up to 5): {sample}.",
                    RuntimeWarning,
                )

        drug_nodes = {n.split("__", 1)[1] for n in input_names if n.startswith("DRUG__")}
        n_before = len(meta)
        meta = meta[meta["pert_id"].astype(str).isin(drug_nodes)].reset_index(drop=True)
        if len(meta) < n_before:
            warnings.warn(
                f"TrajDataset: dropped {n_before - len(meta)} row(s) whose pert_id "
                f"is absent from input_names DRUG__ nodes.",
                RuntimeWarning,
            )

        if "obs_row" not in meta.columns:
            raise ValueError("TrajDataset meta must include obs_row column")

        self.meta = meta.reset_index(drop=True)
        self.pred_dir = pred_dir
        self.input_names = input_names
        self.gene_ixs = [i for i, name in enumerate(input_names) if name.startswith("GENE__")]
        self.horizon = horizon
        self.multiple_shooting = multiple_shooting
        self.sigma_floor = float(sigma_floor)

        obs = open_pred_obs(pred_dir)
        obs_rows = self.meta["obs_row"].to_numpy(dtype=int)
        mu_arr, sigma_arr = materialize_trajectories(obs, obs_rows)
        self._obs_mu = torch.from_numpy(mu_arr)
        self._obs_sigma = clamp_sigma(torch.from_numpy(sigma_arr), self.sigma_floor)

        # The predict_grid obs.npy gene axis is in its native (gene_names.csv /
        # ``src_names``) order, which is NOT the bionetwork's GENE__ input order
        # used by ODEFunc.gene_ixs / accessible_gene_ix / x[self.gene_ixs].
        # Reindex the gene axis into self.gene_ixs order so obs_mu lines up
        # column-for-column with the integrated gene slice (gene_hat). Without
        # this, every gene is supervised against a different gene's trajectory
        # (huge NLL/MSE). When src_names is None we assume obs is already in the
        # GENE__ input order (e.g. synthetic test fixtures).
        if src_names is not None:
            src_pos = {}
            for i_src, name in enumerate(src_names):
                sym = name.split("__", 1)[1] if str(name).startswith("GENE__") else str(name)
                src_pos.setdefault(sym, i_src)

            sel, missing = [], []
            for ix in self.gene_ixs:
                sym = self.input_names[ix].split("__", 1)[1]
                if sym in src_pos:
                    sel.append(src_pos[sym])
                else:
                    missing.append(sym)
            if missing:
                raise ValueError(
                    f"TrajDataset: {len(missing)} bionetwork GENE__ node(s) have no "
                    f"matching column in src_names/gene_names.csv (showing up to 5): "
                    f"{missing[:5]}. The predict_grid was built from a different gene "
                    "set than the bionetwork."
                )
            sel = torch.as_tensor(sel, dtype=torch.long)
            self._obs_mu = self._obs_mu[:, :, sel].contiguous()
            self._obs_sigma = self._obs_sigma[:, :, sel].contiguous()

    def set_horizon(self, horizon):
        self.horizon = horizon

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        obs_mu = self._obs_mu[idx].type(torch.float32)
        obs_sigma = self._obs_sigma[idx].type(torch.float32)

        if self.horizon is not None:
            if self.multiple_shooting:
                t0 = torch.randint(0, obs_mu.shape[0] - self.horizon + 1, size=(1,)).item()
            else:
                t0 = 0
            tT = t0 + self.horizon
            obs_mu = obs_mu[t0:tT, :]
            obs_sigma = obs_sigma[t0:tT, :]

        t0_mu = obs_mu[0, :]

        x = torch.zeros(len(self.input_names), dtype=torch.float32)
        x[self.input_names.index("DRUG__" + str(row["pert_id"]))] = dose_transform(
            torch.tensor([row["dose"]], dtype=torch.float32)
        )
        x[self.input_names.index("LINE__" + str(row["cell_iname"]))] = 1.0
        x[self.gene_ixs] = t0_mu.type(torch.float32)
        x = x.contiguous().detach()

        if self.x_fn_lookup is None:
            return obs_mu, x, obs_sigma

        x_fn = self.x_fn_lookup[row["cell_iname"]]
        return obs_mu, x, obs_sigma, x_fn
