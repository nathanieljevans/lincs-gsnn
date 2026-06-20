"""
train_gsnn_with_odeint.py - Fine-tune a pretrained GSNN against observed
trajectories using ``torchdiffeq.odeint``.

This script complements :mod:`pretrain_gsnn_with_dxdt`: rather than supervising
on instantaneous dx/dt values, it integrates the GSNN-defined ODE (wrapped by
:class:`lincs_gsnn.models.ODEFunc.ODEFunc`) over a horizon of time-points and
matches the gene-slice of the trajectory to the observed expression rollout.

Inputs:
    * ``--bionet``           : directory containing ``bionetwork.pt``
    * ``--pretrained_model`` : warm-start GSNN checkpoint from ``pretrain_gsnn``
    * ``--pretrained_scale`` : dxdt_scale tensor from ``pretrain_gsnn`` (reused
                               by ``ODEFunc`` to undo the pretrain-time scale)
    * ``--model_id``         : replicate id (e.g. ``model_0``)
    * ``--data``             : predict_grid root (``obs.npy``, ``pred_meta.csv``)
    * ``--seed``             : RNG seed for this replicate's cell-drug val split

Outputs:
    * ``trained_model_{model_id}.pt`` written under ``--out``
    * ``val_metrics_history_train_{model_id}.csv`` per-epoch train/val metrics
      (nll, mse, r2, time_series_r)
    * ``val_metrics_train_{model_id}.json`` best-epoch validation summary
      (includes ``best_val_nll``, ``best_val_time_series_r``, ``final_val_nll``)
"""

import argparse
import copy
import json
import os
import time
import warnings

import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torchdiffeq import odeint, odeint_adjoint

from lincs_gsnn.data.TrajDataset import TrajDataset
from lincs_gsnn.models.BIOGSNN import BIOGSNN
from lincs_gsnn.models.ODEFunc import ODEFunc
from lincs_gsnn.proc.cell_drug_split import (
    build_cell_drug_split,
    filter_meta_by_partition,
)
from lincs_gsnn.utils.GaussianNLL import GaussianNLL
from lincs_gsnn.proc.node_activity import load_node_activity_artifact
from lincs_gsnn.proc.gene_norm import load_gene_norm_artifact
from lincs_gsnn.proc.drug_accessibility import (
    accessible_gene_slice_mask,
    accessible_indices,
    get_or_compute_drug_accessible_mask,
)
from lincs_gsnn.train.metrics import evaluate_traj, mean_time_series_correlation
from lincs_gsnn.train.optim import OPTIMIZER_CHOICES, build_optimizer, build_lr_scheduler
from lincs_gsnn.train.console import (
    configure_cuda_performance,
    erase_progress_line,
    format_train_batch_progress,
    train_epoch_table_columns,
    table_header,
    table_row,
    peak_mem_gb,
    reset_peak_mem,
)
from lincs_gsnn.train.checkpoint import (
    append_history_row,
    load_best_model,
    save_epoch_checkpoint,
    try_load_resume,
)


TRAIN_HISTORY_COLUMNS = (
    'epoch', 'train_nll', 'train_mse', 'train_r2', 'train_time_series_r',
    'val_nll', 'val_mse', 'val_r2', 'val_time_series_r',
    'lr', 'time_s', 'max_mem_gb', 'train_gamma_prior',
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str, required=True,                              help="path to predict_grid root (obs.npy, pred_meta.csv)")
    parser.add_argument("--out",                type=str, required=True,                              help="path to output directory")
    parser.add_argument("--bionet",             type=str, required=True,                              help="path to bionetwork directory (contains bionetwork.pt)")
    parser.add_argument("--pretrained_model",   type=str, required=True,                              help="path to pretrained GSNN checkpoint (warm-start)")
    parser.add_argument("--pretrained_scale",   type=str, required=True,                              help="path to dxdt_scale_{model_id}.pt produced by pretrain")
    parser.add_argument("--model_id",           type=str, required=True,                              help="replicate id (e.g., model_0)")
    parser.add_argument("--seed",               type=int, default=0,                                  help="RNG seed for this replicate's cell-drug val split")
    parser.add_argument("--sigma_floor",        type=float, default=1e-4,                            help="minimum target std for GaussianNLL")
    parser.add_argument("--objective",          type=str,   default='nll', choices=['nll', 'mse'],   help="trajectory fine-tune objective on the observed gene rollout: 'nll' = Gaussian NLL under Normal(obs_mu, obs_sigma) (default); 'mse' = plain MSE against obs_mu only, ignoring obs_sigma. Controls the optimized loss AND the metric used for best-checkpoint/LR-scheduler selection.")
    parser.add_argument("--clip_grad_norm",     type=float, default=0.0,                             help="clip gradients to this max L2-norm (torch.nn.utils.clip_grad_norm_) before each optimizer step. <=0 disables (default). Helps stabilize the NLL objective, where small obs_sigma can produce very large gradients.")
    parser.add_argument("--val_cells_per_drug", type=int, default=1,                                  help="number of cell lines held out per drug for validation")
    parser.add_argument("--batch_size",         type=int,   default=32,                               help="batch size for training")
    parser.add_argument("--num_workers",        type=int,   default=4,                                help="number of workers for dataloader")
    parser.add_argument("--epochs",             type=int,   default=25,                               help="number of epochs to train for")
    parser.add_argument("--lr",                 type=float, default=1e-3,                             help="learning rate for optimizer")
    parser.add_argument("--wd",                 type=float, default=1e-6,                             help="weight decay for optimizer")
    parser.add_argument("--patience",           type=int,   default=3,                                help="patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--horizon",            type=int,   default=24,                               help="number of time-points integrated per training sample")
    parser.add_argument("--multiple_shooting",  action='store_true', default=False,                   help="random t0 windows within each trajectory")
    parser.add_argument("--method",             type=str,   default='euler',                          help="torchdiffeq integration method")
    parser.add_argument("--tol",                type=float, default=1e-4,                             help="atol/rtol for adaptive integration methods")
    parser.add_argument("--adjoint",            action='store_true', default=False,                   help="use torchdiffeq.odeint_adjoint for the training rollout. Reconstructs gradients by integrating backward instead of storing the forward graph, making memory O(1) in the number of solver steps. Recommended with adaptive solvers (e.g. dopri5) on stiff systems that otherwise OOM. Eval always uses plain odeint under no_grad (no graph stored).")
    parser.add_argument("--split_path",         type=str,   default=None,                             help="(deprecated) ignored; split is built in-process from --seed")

    # Node-activity flags (must match the ones used at pretrain-time so the
    # loaded GSNN's node_activity gate receives the expected per-cell x_fn).
    parser.add_argument("--node_activity",          action='store_true', default=False,               help="enable per-function-node activity gating; required when the warm-start GSNN was trained with --node_activity")
    parser.add_argument("--node_activity_path",     type=str,            default=None,                help="path to the node_activity.pt artifact (defaults to <bionet>/node_activity.pt)")
    parser.add_argument("--node_activity_hidden",   type=int,            default=16,                  help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--node_activity_temperature", type=float,       default=1.0,                 help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--node_activity_transform", type=str,           default='sigmoid', choices=['sigmoid', 'softmax', 'tanh'], help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--node_activity_mass",     type=float,           default=1.0,                 help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--node_activity_dropout",  type=float,          default=0.0,                 help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--node_activity_mode",     type=str,            default='per-node', choices=['per-node', 'per-channel'], help="(unused at fine-tune time; kept for CLI parity with pretrain)")
    parser.add_argument("--alpha_decay",            type=float,          default=0.0,                 help="L1 sparsity penalty on the NodeActivity gate mean; matches pretrain semantics. 0 disables (default).")

    parser.add_argument("--gene_norm_path",         type=str,            default=None,                help="path to gene_norm.pt (per-gene mu/sigma). Validated against the loaded BIOGSNN's output_names; the loaded model already carries its own mu/sigma buffers, so this flag exists mainly to catch artifact-bundle mismatches between pretrain and train. Defaults to <bionet>/gene_norm.pt.")
    parser.add_argument(
        "--gamma_prior_weight",
        type=float,
        default=0.0,
        help="BIOGSNN only: soft log-rate L2 weight on gamma toward the init prior "
        "carried in the warm-start checkpoint. No-op when 0 or when the checkpoint "
        "has no gamma_prior buffer.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='adamw',
        choices=OPTIMIZER_CHOICES,
        help="optimizer for training",
    )
    parser.add_argument("--amp", action='store_true', default=False,
                        help="mixed-precision training on CUDA (bfloat16 autocast)")
    parser.add_argument("--grad_scaler", action='store_true', default=False,
                        help="use torch.cuda.amp.GradScaler on CUDA; skips optimizer steps when unscaled gradients are non-finite")
    parser.add_argument("--tf32", action='store_true', default=False,
                        help="enable TF32 matmul on CUDA")
    parser.add_argument("--resume_incomplete", action='store_true', default=False,
                        help="resume from checkpoints under <out>/checkpoints/<model_id>/")

    args = parser.parse_args()
    return args


def _node_activity_penalty_args(args, model):
    alpha_decay = float(getattr(args, 'alpha_decay', 0.0) or 0.0)
    na_module = getattr(model, 'node_activity_model', None)
    use_alpha_decay = (
        bool(getattr(args, 'node_activity', False))
        and (na_module is not None)
        and (alpha_decay > 0.0)
    )
    return alpha_decay, na_module, use_alpha_decay


_gamma_prior_warned = False


def _gamma_prior_penalty_args(args, model):
    global _gamma_prior_warned
    weight = float(getattr(args, 'gamma_prior_weight', 0.0) or 0.0)
    has_prior = isinstance(model, BIOGSNN) and hasattr(model, 'gamma_prior')
    use_prior = has_prior and weight > 0.0
    if (
        weight > 0.0
        and isinstance(model, BIOGSNN)
        and not has_prior
        and not _gamma_prior_warned
    ):
        warnings.warn(
            "--gamma_prior_weight > 0 but loaded BIOGSNN has no gamma_prior buffer "
            "(warm-start was not built with --init_rna_half_life). The prior penalty is a no-op.",
            RuntimeWarning,
        )
        _gamma_prior_warned = True
    return weight, use_prior


def train_epoch(args, model, func, dataloader, optim, crit, t, device, accessible_gene_ix, use_amp=False, scaler=None):
    """Run one ODE-integrated training epoch."""
    model.train()
    alpha_decay, na_module, use_alpha_decay = _node_activity_penalty_args(args, model)
    gamma_prior_weight, use_gamma_prior = _gamma_prior_penalty_args(args, model)
    device_type = device.type if isinstance(device, torch.device) else str(device)
    # odeint_adjoint keeps memory O(1) in solver steps (it re-integrates
    # backward), which is what prevents OOM when an adaptive solver takes many
    # tiny steps. Plain odeint stores the full forward graph per step.
    integrate = odeint_adjoint if getattr(args, 'adjoint', False) else odeint

    losses = 0.0
    nll_losses = 0.0
    mse_losses = 0.0
    prior_losses = 0.0
    r2s = 0.0
    ts_r_sum = 0.0
    ts_r_count = 0
    n_batches = 0
    n_loader = len(dataloader)
    epoch_start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        optim.zero_grad()

        if len(batch) == 3:
            obs_mu, x, obs_sigma = batch
            x_fn = None
        elif len(batch) == 4:
            obs_mu, x, obs_sigma, x_fn = batch
            x_fn = x_fn.to(device)
        else:
            raise ValueError(
                f"train_epoch: unexpected batch arity {len(batch)}; "
                "expected 3 or 4 (obs_mu, x, obs_sigma[, x_fn])"
            )

        obs_mu = obs_mu.to(device)
        obs_sigma = obs_sigma.to(device)
        x = x.to(device)
        func.set_edge_mask(None)
        func.set_node_mask(None)
        func.set_x_fn(x_fn)

        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            xt_hat = integrate(
                func=func, y0=x, t=t, method=args.method, atol=args.tol, rtol=args.tol,
            ).transpose(0, 1)
        gene_hat = xt_hat[:, :, func.gene_ixs]
        gene_hat_sub = gene_hat[:, :, accessible_gene_ix]
        obs_sub = obs_mu[:, :, accessible_gene_ix]
        sigma_sub = obs_sigma[:, :, accessible_gene_ix]
        nll_loss = crit(gene_hat_sub, obs_sub, sigma_sub)
        mse_crit = torch.nn.MSELoss(reduction='mean')
        mse_loss = mse_crit(gene_hat_sub, obs_sub)
        # 'mse' trains on obs_mu only (obs_sigma ignored); 'nll' is the
        # inverse-variance weighted Gaussian NLL. Both metrics are always logged.
        loss = mse_loss if getattr(args, 'objective', 'nll') == 'mse' else nll_loss
        if use_alpha_decay:
            loss = loss + alpha_decay * na_module.get_alpha_mean().mean()
        if use_gamma_prior:
            gamma_pen = model.gamma_prior_loss()
            loss = loss + gamma_prior_weight * gamma_pen
            prior_losses += gamma_pen.item()

        clip_grad_norm = float(getattr(args, 'clip_grad_norm', 0.0) or 0.0)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0.0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optim.step()

        losses += loss.item()
        nll_losses += nll_loss.item()
        mse_losses += mse_loss.item()
        with torch.no_grad():
            delta = (obs_sub - obs_sub[:, [0], :]).detach().cpu().numpy().ravel()
            delta_hat = (gene_hat_sub - obs_sub[:, [0], :]).detach().cpu().numpy().ravel()
            batch_r2 = r2_score(delta, delta_hat, multioutput='uniform_average')
            r2s += batch_r2
            batch_r_sum, batch_r_n = mean_time_series_correlation(obs_sub, gene_hat_sub)
            ts_r_sum += batch_r_sum
            ts_r_count += batch_r_n
        n_batches += 1
        print(
            format_train_batch_progress(
                i, n_loader,
                epoch_start=epoch_start,
                loss=loss.item(),
                r2=batch_r2,
            ),
            end='\r',
            flush=True,
        )

    n = max(n_batches, 1)
    metrics = {
        'nll': nll_losses / n,
        'mse': mse_losses / n,
        'r2': r2s / n,
        'time_series_r': ts_r_sum / max(ts_r_count, 1),
        'loss': losses / n,
    }
    if use_gamma_prior:
        metrics['gamma_prior'] = prior_losses / n
    return metrics


def _train_epoch_row_values(
    *,
    epoch,
    train_metrics,
    val_metrics,
    lr,
    best_val,
    time_s,
    max_mem_gb,
):
    values = {
        'epoch': epoch + 1,
        'train_nll': train_metrics['nll'],
        'train_mse': train_metrics['mse'],
        'train_r2': train_metrics['r2'],
        'train_time_series_r': train_metrics['time_series_r'],
        'val_nll': val_metrics['nll'],
        'val_mse': val_metrics['mse'],
        'val_r2': val_metrics['r2'],
        'val_time_series_r': val_metrics['time_series_r'],
        'lr': lr,
        'best_val': best_val,
        'time_s': time_s,
        'max_mem_gb': max_mem_gb,
    }
    if 'gamma_prior' in train_metrics:
        values['train_gamma_prior'] = train_metrics['gamma_prior']
    return values


def train_with_validation(
    args, model, func, train_loader, val_loader, optim, scheduler, crit, t, device, accessible_gene_ix,
    *, out_dir, model_id, start_epoch=0, best_val_nll=float('inf'), best_epoch=-1,
    best_val_mse=float('inf'), use_amp=False, scaler=None,
):
    """Train with per-epoch validation; checkpoint on lowest val NLL."""
    alpha_decay, na_module, _ = _node_activity_penalty_args(args, model)
    _, use_gamma_prior = _gamma_prior_penalty_args(args, model)
    history = []
    best_state_dict = None
    history_path = os.path.join(out_dir, f'val_metrics_history_train_{model_id}.csv')

    # Select best checkpoint / step the LR scheduler on the objective being
    # optimized: val MSE for 'mse', val NLL for 'nll'. best_monitored seeds
    # from the resumed best of the matching metric.
    objective = getattr(args, 'objective', 'nll')
    monitor_key = 'mse' if objective == 'mse' else 'nll'
    best_monitored = best_val_mse if monitor_key == 'mse' else best_val_nll

    table_cols = train_epoch_table_columns(use_gamma_prior=use_gamma_prior)
    table_header_line = table_header(table_cols)
    print(f'Training model (odeint, train/val split, objective={objective}, monitoring val {monitor_key})...')
    print(table_header_line)
    print('-' * len(table_header_line))

    for epoch in range(start_epoch, args.epochs):
        reset_peak_mem(device)
        tic = time.time()
        train_metrics = train_epoch(
            args, model, func, train_loader, optim, crit, t, device, accessible_gene_ix,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = evaluate_traj(
            model, func, val_loader, crit, t, device, accessible_gene_ix,
            method=args.method, tol=args.tol,
            alpha_decay=alpha_decay, na_module=na_module,
            sigma_floor=args.sigma_floor,
            amp=use_amp,
        )
        scheduler.step(val_metrics[monitor_key])
        lr = scheduler.get_last_lr()[0]
        time_s = time.time() - tic
        max_mem_gb = peak_mem_gb(device)

        row = {
            'epoch': epoch + 1,
            'train_nll': train_metrics['nll'],
            'train_mse': train_metrics['mse'],
            'train_r2': train_metrics['r2'],
            'train_time_series_r': train_metrics['time_series_r'],
            'val_nll': val_metrics['nll'],
            'val_mse': val_metrics['mse'],
            'val_r2': val_metrics['r2'],
            'val_time_series_r': val_metrics['time_series_r'],
            'lr': lr,
            'time_s': time_s,
            'max_mem_gb': max_mem_gb,
        }
        if 'gamma_prior' in train_metrics:
            row['train_gamma_prior'] = train_metrics['gamma_prior']
        history.append(row)
        append_history_row(history_path, row, list(TRAIN_HISTORY_COLUMNS))

        save_best = val_metrics[monitor_key] < best_monitored
        if save_best:
            best_monitored = val_metrics[monitor_key]
            best_val_nll = val_metrics['nll']
            best_val_mse = val_metrics['mse']
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        save_epoch_checkpoint(
            out_dir,
            model_id,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            last_epoch=epoch,
            best_epoch=best_epoch,
            best_val_nll=best_val_nll,
            best_val_mse=best_val_mse,
            save_best=save_best,
        )

        erase_progress_line(len(table_header_line))
        print(
            table_row(
                table_cols,
                _train_epoch_row_values(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=lr,
                    best_val=best_monitored,
                    time_s=time_s,
                    max_mem_gb=max_mem_gb,
                ),
            ),
            flush=True,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, pd.DataFrame(history), {
        'best_epoch': best_epoch,
        'best_val_nll': best_val_nll,
        'best_val_mse': best_val_mse,
        'best_val_r2': history[best_epoch - 1]['val_r2'] if best_epoch > 0 else None,
        'best_val_time_series_r': (
            history[best_epoch - 1]['val_time_series_r'] if best_epoch > 0 else None
        ),
        'final_val_nll': history[-1]['val_nll'] if history else None,
        'final_val_mse': history[-1]['val_mse'] if history else None,
        'final_val_r2': history[-1]['val_r2'] if history else None,
        'final_val_time_series_r': (
            history[-1]['val_time_series_r'] if history else None
        ),
        'n_train_batches': len(train_loader),
        'n_val_batches': len(val_loader),
    }


def _save_train_metrics(out_dir, model_id, history_df, summary):
    os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, f'val_metrics_history_train_{model_id}.csv')
    summary_path = os.path.join(out_dir, f'val_metrics_train_{model_id}.json')
    history_df.to_csv(history_path, index=False)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved validation metrics to {history_path} and {summary_path}')


if __name__ == '__main__':

    args = get_args()
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp and device.type == 'cuda'
    use_grad_scaler = args.grad_scaler and device.type == 'cuda'
    use_tf32 = args.tf32 and device.type == 'cuda'
    if args.amp and device.type != 'cuda':
        print('--amp ignored (CUDA not available)')
    if args.grad_scaler and device.type != 'cuda':
        print('--grad_scaler ignored (CUDA not available)')
    if args.tf32 and device.type != 'cuda':
        print('--tf32 ignored (CUDA not available)')
    configure_cuda_performance(use_tf32)
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler) if use_grad_scaler else None
    print(f'Using device: {device}')
    print(f'AMP: {use_amp}')
    print(f'GradScaler: {use_grad_scaler}')
    print(f'TF32: {use_tf32}')
    print(f'optimizer: {args.optimizer}')

    data = torch.load(f'{args.bionet}/bionetwork.pt', weights_only=False)
    model = torch.load(args.pretrained_model, weights_only=False, map_location=device)
    dxdt_scale = torch.load(args.pretrained_scale, weights_only=False).item()

    gn_path = args.gene_norm_path or os.path.join(args.bionet, 'gene_norm.pt')
    if not os.path.exists(gn_path):
        raise FileNotFoundError(
            f"gene_norm.pt not found at {gn_path}. Rebuild the bionetwork with "
            "`make_bio_network.py --gene_stats_path <gene_stats.dict>` or pass "
            "--gene_norm_path explicitly."
        )
    _ = load_gene_norm_artifact(gn_path, output_names=data.node_names_dict['output'])
    print(f"gene_norm: validated artifact against current bionetwork ({gn_path})")

    if hasattr(model, 'hnet') or getattr(model, 'is_hypernetwork', False):
        raise ValueError(
            "Hypernetwork-conditioned models are not supported by "
            "train_gsnn_with_odeint in v1. Disable hypernetwork.enabled or "
            "set train.enabled=false."
        )

    model = model.to(device).train()

    if getattr(model, 'node_activity', False) and not args.node_activity:
        raise ValueError(
            "Loaded GSNN was pretrained with node_activity=True, but "
            "--node_activity was not passed to train_gsnn_with_odeint. "
            "Re-run with --node_activity (and a matching --node_activity_path "
            "if non-default)."
        )

    x_fn_lookup = None
    if args.node_activity:
        na_path = args.node_activity_path or os.path.join(args.bionet, 'node_activity.pt')
        if not os.path.exists(na_path):
            raise FileNotFoundError(
                f"--node_activity requested but artifact not found at {na_path}. "
                "Rebuild the bionetwork with `make_bio_network.py --node_activity` "
                "or pass --node_activity_path explicitly."
            )
        na_payload = load_node_activity_artifact(na_path, node_names_dict=data.node_names_dict)
        x_fn_lookup = na_payload['x_fn_by_ciname']
        print(f"node_activity: loaded {len(x_fn_lookup)} cell-line activity rows "
              f"(activity_dim={int(na_payload['activity_dim'])}) from {na_path}")

    pred_meta = pd.read_csv(f'{args.data}/pred_meta.csv')
    # predict_grid obs.npy gene axis is in this (gene_names.csv) order; TrajDataset
    # uses it to reindex obs into the bionetwork GENE__ input order so the gene
    # slice aligns with gene_hat/accessible_gene_ix during the rollout loss.
    src_gene_names = pd.read_csv(f'{args.data}/gene_names.csv')['gene_names'].astype(str).tolist()
    pert_ids_net = [n.split('__')[1] for n in data.node_names_dict['input'] if n.startswith('DRUG__')]
    missing = set(pred_meta['pert_id'].unique()) - set(pert_ids_net)
    if missing:
        print(f'Dropping {len(missing)} drugs missing from bionet (first few): {sorted(missing)[:5]}')
    pred_meta = pred_meta[pred_meta['pert_id'].isin(pert_ids_net)].reset_index(drop=True)

    pert_ids = sorted(pred_meta['pert_id'].unique().tolist())
    split_df = build_cell_drug_split(
        pert_ids=pert_ids,
        cell_inames=sorted(pred_meta['cell_iname'].unique().tolist()),
        n_val=int(args.val_cells_per_drug),
        seed=int(args.seed),
    )
    train_meta = filter_meta_by_partition(pred_meta, split_df, 'train')
    val_meta = filter_meta_by_partition(pred_meta, split_df, 'val')
    if len(train_meta) == 0:
        raise ValueError("train partition is empty after applying cell-drug split")
    if len(val_meta) == 0:
        raise ValueError("val partition is empty after applying cell-drug split")

    print(
        f'cell_drug_split (seed={args.seed}): train rows={len(train_meta)}, '
        f'val rows={len(val_meta)}'
    )

    pred_dir = args.data
    print(f'Training replicate: {args.model_id}')

    train_dataset = TrajDataset(
        meta=train_meta,
        input_names=data.node_names_dict['input'],
        pred_dir=pred_dir,
        horizon=args.horizon,
        multiple_shooting=args.multiple_shooting,
        x_fn_lookup=x_fn_lookup,
        sigma_floor=args.sigma_floor,
        src_names=src_gene_names,
    )
    val_dataset = TrajDataset(
        meta=val_meta,
        input_names=data.node_names_dict['input'],
        pred_dir=pred_dir,
        horizon=args.horizon,
        multiple_shooting=False,
        x_fn_lookup=x_fn_lookup,
        sigma_floor=args.sigma_floor,
        src_names=src_gene_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    accessible_mask = get_or_compute_drug_accessible_mask(data)
    # gene_ixs come from node names; compute once before building ODEFunc.
    _probe_func = ODEFunc(model, data.node_names_dict['input'], scale=dxdt_scale)
    accessible_gene_ix = accessible_indices(
        accessible_gene_slice_mask(
            accessible_mask,
            data.node_names_dict['output'],
            data.node_names_dict['input'],
            _probe_func.gene_ixs,
        )
    ).to(device)
    del _probe_func
    n_acc = int(accessible_mask.sum())
    n_total = len(data.node_names_dict['output'])
    print(f'Drug-accessible output genes: {n_acc} / {n_total}')

    # The pretrained GSNN field is parameterized in NORMALIZED time: the
    # predict_grid spans the full trajectory over tau in [0, 1] across its
    # n_time_pts grid points, and dx/dt is d(z)/d(tau) (its empirical scale
    # ~O(1) per unit-tau, not per-hour). odeint must therefore step on the same
    # normalized grid with dt = 1/(n_time_pts - 1); integrating on the raw hour
    # axis (or with dt=1) drives the rollout to |z|~1e2-1e3 and MSE>1e4. The
    # grid is uniform and ODEFunc is autonomous (forward ignores t), so
    # arange * dt lands on the observed points and also covers
    # --multiple_shooting.
    n_time_pts_grid = int(pred_meta['n_time_pts'].iloc[0])
    if n_time_pts_grid < 2:
        raise ValueError(
            f"predict_grid n_time_pts={n_time_pts_grid} (<2); cannot derive a time step."
        )
    dt_grid = 1.0 / (n_time_pts_grid - 1)
    t = torch.arange(args.horizon, dtype=torch.float32, device=device) * dt_grid
    print(
        f'integration time grid (normalized): dt={dt_grid:.5f} '
        f'(n_time_pts={n_time_pts_grid}); '
        f't[0..{args.horizon - 1}] spans [0, {float(t[-1]):.5f}]'
    )

    optim = build_optimizer(model, args.optimizer, lr=args.lr, wd=args.wd)
    scheduler = build_lr_scheduler(optim, patience=args.patience)
    crit = GaussianNLL()

    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model, start_epoch, best_val_nll, best_epoch, best_val_mse, _ = try_load_resume(
        args.out,
        args.model_id,
        resume_incomplete=args.resume_incomplete,
        device=device,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
    )
    func = ODEFunc(model, data.node_names_dict['input'], scale=dxdt_scale).to(device)
    model_dxdt_clip = float(getattr(model, 'dxdt_clip', 0.0) or 0.0)
    print(f'integrator: {"odeint_adjoint" if args.adjoint else "odeint"} '
          f'(method={args.method}, tol={args.tol}); '
          f'model dxdt_clip={model_dxdt_clip if model_dxdt_clip > 0.0 else None}')
    if args.resume_incomplete and start_epoch > 0:
        print(
            f'resuming incomplete train from epoch {start_epoch} '
            f'(best_val_nll={best_val_nll:.4E} at epoch {best_epoch})'
        )

    if start_epoch >= args.epochs:
        print(
            f'train already complete ({start_epoch} epochs logged, '
            f'epochs={args.epochs}); writing final artifacts from checkpoint'
        )
        model = load_best_model(args.out, args.model_id, device)
        summary = {
            'best_epoch': best_epoch,
            'best_val_nll': best_val_nll,
            'best_val_mse': best_val_mse,
            'best_val_r2': None,
            'best_val_time_series_r': None,
            'final_val_nll': best_val_nll,
            'final_val_mse': best_val_mse,
            'final_val_r2': None,
            'final_val_time_series_r': None,
            'n_train_batches': len(train_loader),
            'n_val_batches': len(val_loader),
        }
        history_path = os.path.join(args.out, f'val_metrics_history_train_{args.model_id}.csv')
        history_df = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
        if not history_df.empty and best_epoch > 0:
            best_row = history_df[history_df['epoch'] == best_epoch]
            if not best_row.empty:
                summary['best_val_r2'] = float(best_row.iloc[0]['val_r2'])
                summary['best_val_time_series_r'] = float(best_row.iloc[0]['val_time_series_r'])
        _save_train_metrics(args.out, args.model_id, history_df, summary)
        out_path = f'{args.out}/trained_model_{args.model_id}.pt'
        torch.save(model, out_path)
        print(f'Saved fine-tuned model to {out_path}')
    else:
        model, history_df, summary = train_with_validation(
            args, model, func, train_loader, val_loader, optim, scheduler, crit, t, device,
            accessible_gene_ix,
            out_dir=args.out,
            model_id=args.model_id,
            start_epoch=start_epoch,
            best_val_nll=best_val_nll,
            best_epoch=best_epoch,
            best_val_mse=best_val_mse if best_val_mse is not None else float('inf'),
            use_amp=use_amp,
            scaler=scaler,
        )
        _save_train_metrics(args.out, args.model_id, history_df, summary)

        out_path = f'{args.out}/trained_model_{args.model_id}.pt'
        torch.save(model, out_path)
        print(f'Saved fine-tuned model to {out_path}')
