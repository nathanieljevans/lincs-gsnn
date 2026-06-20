"""Training/validation metrics for GSNN pretrain and odeint fine-tune."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from lincs_gsnn.utils.nll import gaussian_nll


def mean_time_series_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    dim: int = 1,
    eps: float = 1e-8,
) -> tuple[float, int]:
    """Pearson r along time for each (batch, gene) pair; return global sum and count."""
    y_true_c = y_true - y_true.mean(dim=dim, keepdim=True)
    y_pred_c = y_pred - y_pred.mean(dim=dim, keepdim=True)
    num = (y_true_c * y_pred_c).sum(dim=dim)
    denom = torch.sqrt((y_true_c ** 2).sum(dim=dim) * (y_pred_c ** 2).sum(dim=dim))
    valid = denom > eps
    r = torch.where(valid, num / denom.clamp_min(eps), torch.zeros_like(num))
    return float(r[valid].sum().item()), int(valid.sum().item())


def _unpack_dxdt_batch(batch, device):
    if len(batch) == 3:
        X, dxdt_mu, dxdt_sigma = batch
        x_fn = None
    elif len(batch) == 4:
        X, dxdt_mu, dxdt_sigma, x_fn = batch
        x_fn = x_fn.to(device)
    else:
        raise ValueError(
            f"unexpected batch arity {len(batch)}; expected 3 or 4 (X, mu, sigma[, x_fn])"
        )
    return X.to(device), dxdt_mu.to(device), dxdt_sigma.to(device), x_fn


@torch.no_grad()
def evaluate_dxdt(
    model: torch.nn.Module,
    dataloader: DataLoader,
    crit: Any,
    device: torch.device,
    accessible_out_ix: torch.Tensor,
    *,
    alpha_decay: float = 0.0,
    na_module: Any = None,
    sigma_floor: float = 0.0,
    amp: bool = False,
) -> dict[str, float]:
    """Evaluate NLL, MSE (vs mean), and R² on drug-accessible dx/dt targets."""
    model.eval()
    use_alpha = na_module is not None and alpha_decay > 0.0
    losses = 0.0
    nll_sum = 0.0
    mse_sum = 0.0
    r2_sum = 0.0
    n_batches = 0
    mse_crit = torch.nn.MSELoss(reduction="mean")

    device_type = device.type if isinstance(device, torch.device) else str(device)

    for batch in dataloader:
        X, dxdt_mu, dxdt_sigma, x_fn = _unpack_dxdt_batch(batch, device)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            dxdt_hat = model(X) if x_fn is None else model(X, x_fn=x_fn)

        mu_sub = dxdt_mu[:, accessible_out_ix]
        sigma_sub = dxdt_sigma[:, accessible_out_ix]
        hat_sub = dxdt_hat[:, accessible_out_ix]

        nll_loss = crit(hat_sub, mu_sub, sigma_sub)
        mse_loss = mse_crit(hat_sub, mu_sub)
        loss = nll_loss
        if use_alpha:
            loss = loss + alpha_decay * na_module.get_alpha_mean().mean()

        losses += loss.item()
        nll_sum += nll_loss.item()
        mse_sum += mse_loss.item()
        r2_sum += r2_score(
            mu_sub.cpu().numpy(),
            hat_sub.cpu().numpy(),
            multioutput="uniform_average",
        )
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": losses / n,
        "nll": nll_sum / n,
        "mse": mse_sum / n,
        "r2": r2_sum / n,
        "n_batches": n_batches,
    }


def _unpack_traj_batch(batch, device):
    if len(batch) == 3:
        obs_mu, x, obs_sigma = batch
        x_fn = None
    elif len(batch) == 4:
        obs_mu, x, obs_sigma, x_fn = batch
        x_fn = x_fn.to(device)
    else:
        raise ValueError(
            f"unexpected batch arity {len(batch)}; expected 3 or 4 (obs_mu, x, sigma[, x_fn])"
        )
    return obs_mu.to(device), x.to(device), obs_sigma.to(device), x_fn


@torch.no_grad()
def evaluate_traj(
    model: torch.nn.Module,
    func: Any,
    dataloader: DataLoader,
    crit: Any,
    t: torch.Tensor,
    device: torch.device,
    accessible_gene_ix: torch.Tensor,
    *,
    method: str,
    tol: float,
    alpha_decay: float = 0.0,
    na_module: Any = None,
    sigma_floor: float = 0.0,
    amp: bool = False,
) -> dict[str, float]:
    """Evaluate trajectory NLL, MSE (vs mean), delta-R², and raw time-series Pearson r."""
    from torchdiffeq import odeint

    model.eval()
    use_alpha = na_module is not None and alpha_decay > 0.0
    losses = 0.0
    nll_sum = 0.0
    mse_sum = 0.0
    r2_sum = 0.0
    ts_r_sum = 0.0
    ts_r_count = 0
    n_batches = 0
    mse_crit = torch.nn.MSELoss(reduction="mean")
    device_type = device.type if isinstance(device, torch.device) else str(device)

    for batch in dataloader:
        obs_mu, x, obs_sigma, x_fn = _unpack_traj_batch(batch, device)
        func.set_edge_mask(None)
        func.set_node_mask(None)
        func.set_x_fn(x_fn)

        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=amp,
        ):
            xt_hat = odeint(
                func=func, y0=x, t=t, method=method, atol=tol, rtol=tol
            ).transpose(0, 1)
        gene_hat = xt_hat[:, :, func.gene_ixs]
        gene_hat_sub = gene_hat[:, :, accessible_gene_ix]
        obs_sub = obs_mu[:, :, accessible_gene_ix]
        sigma_sub = obs_sigma[:, :, accessible_gene_ix]

        nll_loss = crit(gene_hat_sub, obs_sub, sigma_sub)
        mse_loss = mse_crit(gene_hat_sub, obs_sub)
        if use_alpha:
            loss = nll_loss + alpha_decay * na_module.get_alpha_mean().mean()
        else:
            loss = nll_loss

        losses += loss.item()
        nll_sum += nll_loss.item()
        mse_sum += mse_loss.item()
        delta = (obs_sub - obs_sub[:, [0], :]).detach().cpu().numpy().ravel()
        delta_hat = (gene_hat_sub - obs_sub[:, [0], :]).detach().cpu().numpy().ravel()
        r2_sum += r2_score(delta, delta_hat, multioutput="uniform_average")
        batch_r_sum, batch_r_n = mean_time_series_correlation(obs_sub, gene_hat_sub)
        ts_r_sum += batch_r_sum
        ts_r_count += batch_r_n
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": losses / n,
        "nll": nll_sum / n,
        "mse": mse_sum / n,
        "r2": r2_sum / n,
        "time_series_r": ts_r_sum / max(ts_r_count, 1),
        "n_batches": n_batches,
    }
