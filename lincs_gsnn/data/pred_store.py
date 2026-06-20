"""Shared fp16 predict_grid store (obs.npy + dxdt.npy summary statistics).

Layout in ``predict_grid/``:

- ``obs.npy``:   (N_cond, T, 2, G) fp16 — ch0=mean, ch1=std of expression
- ``dxdt.npy``:  (N_cond, T, 2, G) fp16 — ch0=mean, ch1=std of dx/dt

Indexed via ``obs_row`` / ``time_idx`` columns in ``pred_meta.csv`` /
``dxdt_meta.csv``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch

OBS_ARRAY = "obs.npy"
DXDT_ARRAY = "dxdt.npy"

STAT_MEAN = 0
STAT_STD = 1


@lru_cache(maxsize=16)
def open_pred_obs(pred_dir: str) -> np.ndarray:
    """Load (N_cond, T, 2, G) fp16 expression summary memmap."""
    pred_dir = os.path.abspath(pred_dir)
    obs_path = os.path.join(pred_dir, OBS_ARRAY)
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Missing {obs_path}. Re-run predict_grid.")
    return np.load(obs_path, mmap_mode="r")


@lru_cache(maxsize=16)
def open_pred_dxdt(pred_dir: str) -> np.ndarray:
    """Load (N_cond, T, 2, G) fp16 dx/dt summary memmap."""
    pred_dir = os.path.abspath(pred_dir)
    dxdt_path = os.path.join(pred_dir, DXDT_ARRAY)
    if not os.path.isfile(dxdt_path):
        raise FileNotFoundError(f"Missing {dxdt_path}. Re-run predict_grid.")
    return np.load(dxdt_path, mmap_mode="r")


def _stat_index(stat: str) -> int:
    if stat == "mean":
        return STAT_MEAN
    if stat == "std":
        return STAT_STD
    raise ValueError(f"stat must be 'mean' or 'std', got {stat!r}")


def get_pred_trajectory(
    obs: np.ndarray,
    obs_row: int,
    stat: str = "mean",
) -> torch.Tensor:
    """Trajectory (T, G) as fp16 torch tensor for the requested statistic."""
    stat_idx = _stat_index(stat)
    return torch.from_numpy(obs[obs_row, :, stat_idx].copy())


def get_pred_dxdt(
    dxdt: np.ndarray,
    obs_row: int,
    time_idx: int,
    stat: str = "mean",
) -> torch.Tensor:
    """dx/dt vector (G,) as fp16 torch tensor for the requested statistic."""
    stat_idx = _stat_index(stat)
    return torch.from_numpy(dxdt[obs_row, time_idx, stat_idx].copy())


def materialize_obs_gene_slice(
    obs: np.ndarray,
    obs_rows: np.ndarray,
    time_idx: int,
    stat: str = "mean",
) -> np.ndarray:
    """Gather (n_rows, G) float32 gene expression at one time point."""
    stat_idx = _stat_index(stat)
    rows = obs[obs_rows, time_idx, stat_idx, :]
    return np.asarray(rows, dtype=np.float32)


def materialize_dxdt_rows(
    obs: np.ndarray,
    dxdt: np.ndarray,
    obs_rows: np.ndarray,
    time_idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather (n_rows, G) mean and std dx/dt vectors for aligned meta rows."""
    mu = dxdt[obs_rows, time_idxs, STAT_MEAN, :]
    sigma = dxdt[obs_rows, time_idxs, STAT_STD, :]
    return (
        np.asarray(mu, dtype=np.float32),
        np.asarray(sigma, dtype=np.float32),
    )


def materialize_trajectories(
    obs: np.ndarray,
    obs_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather (n_rows, T, G) mean and std expression trajectories."""
    mu = obs[obs_rows, :, STAT_MEAN, :]
    sigma = obs[obs_rows, :, STAT_STD, :]
    return (
        np.asarray(mu, dtype=np.float32),
        np.asarray(sigma, dtype=np.float32),
    )
