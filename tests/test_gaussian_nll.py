"""Tests for Gaussian NLL helpers."""

from __future__ import annotations

import torch

from lincs_gsnn.utils.GaussianNLL import GaussianNLL
from lincs_gsnn.utils.nll import clamp_sigma, gaussian_nll


def test_gaussian_nll_matches_torch_distribution():
    crit = GaussianNLL()
    mu = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    sigma = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    pred = torch.tensor([[0.1, 0.9], [2.2, 2.8]])
    manual = -torch.distributions.Normal(mu, sigma).log_prob(pred).mean()
    assert torch.allclose(crit(pred, mu, sigma), manual)


def test_minimized_at_prediction_equals_mean():
    crit = GaussianNLL()
    mu = torch.zeros(4, 6)
    sigma = torch.ones(4, 6) * 0.3
    at_mu = crit(mu, mu, sigma)
    off_mu = crit(mu + 0.5, mu, sigma)
    assert at_mu < off_mu


def test_sigma_floor_makes_zero_sigma_finite():
    pred = torch.zeros(2, 3)
    mu = torch.zeros(2, 3)
    sigma = torch.zeros(2, 3)
    loss = gaussian_nll(pred, mu, sigma, sigma_floor=1e-4)
    assert torch.isfinite(loss)


def test_gradient_points_toward_mean():
    pred = torch.tensor([[1.0]], requires_grad=True)
    mu = torch.tensor([[0.0]])
    sigma = torch.tensor([[1.0]])
    loss = gaussian_nll(pred, mu, sigma)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.item() > 0


def test_clamp_sigma():
    x = torch.tensor([0.0, 0.5])
    out = clamp_sigma(x, 1e-3)
    assert out.min().item() >= 1e-3
