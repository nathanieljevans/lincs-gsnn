"""Gaussian NLL helpers with optional sigma floor."""

from __future__ import annotations

import torch

from lincs_gsnn.utils.GaussianNLL import GaussianNLL


def clamp_sigma(sigma: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    """Clamp target standard deviations to a positive floor."""
    floor = float(sigma_floor)
    if floor <= 0.0:
        return sigma
    return sigma.clamp_min(floor)


def gaussian_nll(
    pred: torch.Tensor,
    target_mu: torch.Tensor,
    target_sigma: torch.Tensor,
    *,
    sigma_floor: float = 0.0,
) -> torch.Tensor:
    """Scalar mean NLL of ``pred`` under Normal(target_mu, target_sigma)."""
    crit = GaussianNLL()
    sigma = clamp_sigma(target_sigma, sigma_floor)
    return crit(pred, target_mu, sigma)
