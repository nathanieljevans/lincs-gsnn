"""Tests for trajectory NLL evaluation in :mod:`lincs_gsnn.train.metrics`."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from lincs_gsnn.train.metrics import evaluate_traj
from lincs_gsnn.utils.GaussianNLL import GaussianNLL
from lincs_gsnn.utils.nll import gaussian_nll


class _StubFunc:
    gene_ixs = slice(0, 2)

    def set_edge_mask(self, *_):
        pass

    def set_node_mask(self, *_):
        pass

    def set_x_fn(self, *_):
        pass

    def __call__(self, t, y):
        return torch.zeros_like(y)


def test_evaluate_traj_returns_nll():
    obs_mu = torch.tensor([[[1.0, 2.0], [1.1, 2.1]]])
    obs_sigma = torch.tensor([[[0.2, 0.3], [0.2, 0.3]]])
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])  # last two are non-gene inputs
    ds = TensorDataset(obs_mu, x, obs_sigma)
    loader = DataLoader(ds, batch_size=1)

    model = torch.nn.Linear(1, 1)
    func = _StubFunc()
    crit = GaussianNLL()
    t = torch.tensor([0.0, 1.0])
    accessible = torch.tensor([0, 1])

    metrics = evaluate_traj(
        model,
        func,
        loader,
        crit,
        t,
        torch.device("cpu"),
        accessible,
        method="euler",
        tol=1e-4,
    )

    pred = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    expected_nll = gaussian_nll(pred, obs_mu[:, :, accessible], obs_sigma[:, :, accessible])
    assert "nll" in metrics
    assert abs(metrics["nll"] - expected_nll.item()) < 1e-5
    assert "mse" in metrics
    assert "time_series_r" in metrics
