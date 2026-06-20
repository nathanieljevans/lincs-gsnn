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

from gsnn.models.GSNN import GSNN
from lincs_gsnn.models.BIOGSNN import BIOGSNN
from lincs_gsnn.data.DXDTDataset import DXDTDataset
from lincs_gsnn.proc.cell_drug_split import (
    build_cell_drug_split,
    filter_meta_by_partition,
)
from lincs_gsnn.utils.GaussianNLL import GaussianNLL
from lincs_gsnn.proc.node_activity import load_node_activity_artifact
from lincs_gsnn.proc.gene_norm import load_gene_norm_artifact
from lincs_gsnn.proc.drug_accessibility import (
    accessible_indices,
    get_or_compute_drug_accessible_mask,
)
from lincs_gsnn.train.metrics import evaluate_dxdt
from lincs_gsnn.train.optim import OPTIMIZER_CHOICES, build_optimizer, build_lr_scheduler
from lincs_gsnn.train.console import (
    configure_cuda_performance,
    erase_progress_line,
    format_train_batch_progress,
    pretrain_epoch_table_columns,
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


PRETRAIN_HISTORY_COLUMNS = (
    'epoch', 'train_nll', 'train_mse', 'train_r2',
    'val_nll', 'val_mse', 'val_r2', 'lr', 'time_s', 'max_mem_gb',
    'train_gamma_prior',
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                   help="path to data directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                help="path to output directory")
    parser.add_argument("--bionet",             type=str,               default='../../proc/bionetwork.pt',       help="path to bionetwork file")
    parser.add_argument("--model_id",             type=str,               required=True,                             help="model replicate id (e.g., model_0)")
    parser.add_argument("--seed",                 type=int,               default=0,                                 help="RNG seed for this replicate's cell-drug val split")
    parser.add_argument("--sigma_floor",          type=float,             default=1e-4,                              help="minimum clamp on target dxdt std for NLL")
    parser.add_argument("--objective",            type=str,               default='nll', choices=['nll', 'mse'],     help="training objective on drug-accessible dx/dt targets: 'nll' = Gaussian NLL under Normal(target_mu, target_sigma) (inverse-variance weighted; default); 'mse' = plain MSE against target_mu only, ignoring target_sigma. Controls the optimized loss AND the metric used for best-checkpoint/LR-scheduler selection.")
    parser.add_argument("--clip_grad_norm",       type=float,             default=0.0,                               help="clip gradients to this max L2-norm (torch.nn.utils.clip_grad_norm_) before each optimizer step. <=0 disables (default). Helps stabilize the NLL objective, where small target_sigma can produce very large gradients.")
    parser.add_argument("--batch_size",         type=int,               default=64,                                help="batch size for training")
    parser.add_argument("--num_workers",        type=int,               default=4,                                 help="number of workers for dataloader")
    parser.add_argument("--epochs",             type=int,               default=100,                               help="number of epochs to train for")
    parser.add_argument("--lr",                 type=float,             default=1e-3,                              help="learning rate for optimizer")
    parser.add_argument("--wd",                 type=float,             default=1e-4,                              help="weight decay for optimizer")
    parser.add_argument("--patience",           type=int,               default=10,                                help="patience for learning rate scheduler")
    parser.add_argument("--channels",           type=int,               default=64,                                help="number of channels in the model")
    parser.add_argument("--layers",             type=int,               default=3,                                 help="number of layers in the model")
    parser.add_argument("--share_layers",       action='store_true',    default=False,                              help="whether to share layers between input and output nodes")
    parser.add_argument("--dropout",            type=float,             default=0.1,                               help="dropout rate for the model")
    parser.add_argument("--norm",               type=str,               default='batch',                           help="normalization type for the model [batch, layer, none]")
    parser.add_argument("--checkpoint",         action='store_true',    default=False,                              help="whether to use checkpointing in the model")
    parser.add_argument("--init",               type=str,               default='degree_normalized',               help="initialization type for the model")
    parser.add_argument("--add_function_self_edges", action=argparse.BooleanOptionalAction, default=True,           help="whether to add function->function self-edges in the GSNN")
    parser.add_argument("--bias",               action=argparse.BooleanOptionalAction, default=True,                help="whether SparseLinear layers learn an additive bias")
    parser.add_argument("--residual",           action=argparse.BooleanOptionalAction, default=True,                help="whether ResBlocks use residual connections")
    parser.add_argument("--node_mlp",           action=argparse.BooleanOptionalAction, default=True,                help="whether each function node has its own per-node MLP")
    parser.add_argument("--node_mlp_dim",       type=int,               default=128,                                help="hidden dimension of the per-node MLP (mapped to GSNN's node_mlp_hidden)")
    parser.add_argument("--node_attn",          action=argparse.BooleanOptionalAction, default=False,               help="whether each function node uses per-node attention (passed as node_attn to GSNN)")
    parser.add_argument("--attn_mlp_hidden",    type=int,               default=32,                                 help="hidden dimension of the per-node attention MLP (passed as attn_mlp_hidden to GSNN)")

    # ------------------------------------------------------------------
    # Optional hypernetwork conditioning on cell line. When --use_hypernetwork
    # is NOT passed, all behavior below is identical to the legacy pipeline.
    # ------------------------------------------------------------------
    parser.add_argument("--use_hypernetwork",   action='store_true',    default=False,                              help="train a cell-line-conditioned hypernetwork wrapping the GSNN (default: off)")
    parser.add_argument("--hnet_stochastic_channels", type=int,         default=4,                                  help="dimension of latent z (hypernetwork only)")
    parser.add_argument("--hnet_width",         type=int,               default=8,                                  help="hidden width of f_phi (hypernetwork only)")
    parser.add_argument("--hnet_pz",            type=str,               default='normal',                           help="latent prior: normal | uniform | bernoulli | categorical")
    parser.add_argument("--hnet_learn_pz",      action='store_true',    default=False,                              help="learn p(z) via RealNVP normalizing flow")
    parser.add_argument("--hnet_affine",        action='store_true',    default=False,                              help="learnable scale on theta output of f_phi")
    parser.add_argument("--hnet_norm",          type=str,               default='none',                             help="normalization inside f_phi: none | layer")
    parser.add_argument("--hnet_dropout",       type=float,             default=0.0,                                help="dropout inside f_phi")
    parser.add_argument("--hnet_bias",          action='store_true',    default=False,                              help="bias in f_phi linear layers")
    parser.add_argument("--hnet_embed_dim",     type=int,               default=0,                                  help="0 => one-hot over LINE__ entries; >0 => learned cell embedding (currently treated as 0)")
    parser.add_argument("--hnet_n_train_samples", type=int,             default=1,                                  help="number of theta samples per batch; 1 avoids vmap")
    parser.add_argument("--hnet_loss",          type=str,               default='mse',                              help="loss function: mse | edl (edl requires n_train_samples>1)")
    parser.add_argument("--hnet_pretrain_init", action='store_true',    default=False,                              help="warm-start f_phi via hnet.train.hnet.init_hnet using a fresh GSNN's init dict")

    # ------------------------------------------------------------------
    # Optional per-function-node activity conditioning (node_activity).
    # When --node_activity is NOT passed, behavior below is byte-identical
    # to the pre-node-activity pipeline. Mutually exclusive with
    # --use_hypernetwork (see validation in __main__).
    # ------------------------------------------------------------------
    parser.add_argument("--node_activity",          action='store_true',    default=False,                              help="enable per-function-node activity gating; requires a node_activity.pt artifact built by make_bio_network.py --node_activity")
    parser.add_argument("--node_activity_path",     type=str,               default=None,                               help="path to the node_activity.pt artifact (defaults to <bionet>/node_activity.pt)")
    parser.add_argument("--node_activity_hidden",   type=int,               default=16,                                 help="hidden width of the NodeActivity MLP (passed to GSNN; node_activity only)")
    parser.add_argument("--node_activity_temperature", type=float,          default=1.0,                                help="sigmoid temperature for the node-activity gate (passed to GSNN; node_activity only)")
    parser.add_argument("--node_activity_transform", type=str,              default='sigmoid', choices=['sigmoid', 'softmax', 'tanh'], help="transformation applied to node-activity logits (passed to GSNN; node_activity only)")
    parser.add_argument("--node_activity_mass",     type=float,             default=1.0,                                help="mass parameter for NodeActivity (passed to GSNN; node_activity only)")
    parser.add_argument("--node_activity_dropout", type=float,              default=0.0,                                help="dropout probability inside the NodeActivity MLP (passed to GSNN; node_activity only)")
    parser.add_argument("--node_activity_mode",     type=str,               default='per-node', choices=['per-node', 'per-channel'], help="NodeActivity gating mode passed to GSNN (per-node or per-channel)")
    parser.add_argument("--alpha_decay",            type=float,             default=1e-2,                               help="L1-style sparsity penalty on NodeActivity gate activations; added to the MSE objective as `alpha_decay * mean(node_activity_model.get_alpha_mean())`. No-op when --node_activity is not set or --alpha_decay <= 0.")

    parser.add_argument("--gene_norm_path",         type=str,               default=None,                               help="path to gene_norm.pt (per-gene control-population mu/sigma artifact built by make_bio_network.py --gene_stats_path). Required by BIOGSNN; defaults to <bionet>/gene_norm.pt.")
    parser.add_argument("--split_path",             type=str,               default=None,                               help="(deprecated) ignored; split is built in-process from --seed")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--GSNN",               action='store_true',    default=False,                              help="train a GSNN (default when neither --GSNN nor --BIOGSNN is passed)")
    model_group.add_argument("--BIOGSNN",            action='store_true',    default=False,                              help="train a BIOGSNN with log1p-back mRNA degradation (requires gene_norm.pt)")
    parser.add_argument(
        "--dxdt_nonlin",
        type=str,
        default=None,
        choices=['relu', 'leaky_relu', 'sigmoid', 'softplus', 'elu', 'selu', 'gelu', 'swish'],
        help="optional nonlinearity on GSNN output before degradation (BIOGSNN only; default: identity)",
    )
    parser.add_argument(
        "--init_rna_half_life",
        type=float,
        default=None,
        help="BIOGSNN only: mRNA half-life prior in hours for per-gene gamma init "
        "(Schwanhäusser et al. 2011 median ~9h). Requires --BIOGSNN.",
    )
    parser.add_argument(
        "--dxdt_clip",
        type=float,
        default=0.0,
        help="BIOGSNN only: intrinsic soft tanh bound on the net dx/dt magnitude "
        "(model's native scaled-dxdt units), fixed at construction and carried in "
        "the checkpoint so pretrain and odeint fine-tune share identical bounded "
        "dynamics. Reduces ODE stiffness/step count downstream. <=0 disables (default).",
    )
    parser.add_argument(
        "--gamma_prior_weight",
        type=float,
        default=0.0,
        help="BIOGSNN only: soft log-rate L2 weight on gamma toward the init prior. "
        "No-op when 0 or when init_rna_half_life was not used at construction.",
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
    parser.add_argument("--tf32", action='store_true', default=False,
                        help="enable TF32 matmul on CUDA")
    parser.add_argument("--resume_incomplete", action='store_true', default=False,
                        help="resume from checkpoints under <out>/checkpoints/<model_id>/")

    args = parser.parse_args()
    return args


def _gsnn_kwargs_from_args(args, data, node_activity_dim: int = 1) -> dict:
    """Centralize GSNN construction kwargs so train- and explain-time models
    agree on architecture.

    When ``args.node_activity`` is true, the returned dict enables the GSNN's
    NodeActivity gate and threads the per-channel size through. Callers must
    pass ``x_fn`` of shape ``(B, n_function_nodes, node_activity_dim)`` on
    every forward call.
    """
    kw = dict(
        edge_index_dict=data.edge_index_dict,
        node_names_dict=data.node_names_dict,
        channels=args.channels,
        layers=args.layers,
        share_layers=args.share_layers,
        dropout=args.dropout,
        checkpoint=args.checkpoint,
        init=args.init,
        norm=args.norm,
        add_function_self_edges=args.add_function_self_edges,
        bias=args.bias,
        residual=args.residual,
        node_mlp=args.node_mlp,
        node_mlp_hidden=args.node_mlp_dim,
        node_attn=args.node_attn,
        attn_mlp_hidden=args.attn_mlp_hidden,
    )
    if getattr(args, 'node_activity', False):
        kw.update(
            node_activity=True,
            node_activity_dim=int(node_activity_dim),
            node_activity_hidden=int(args.node_activity_hidden),
            node_activity_temperature=float(args.node_activity_temperature),
            node_activity_transform=str(getattr(args, 'node_activity_transform', 'sigmoid')),
            node_activity_mass=float(getattr(args, 'node_activity_mass', 1.0)),
            node_activity_dropout=float(getattr(args, 'node_activity_dropout', 0.0) or 0.0),
            node_activity_mode=str(getattr(args, 'node_activity_mode', 'per-node')),
        )
    return kw


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
            "--gamma_prior_weight > 0 but BIOGSNN has no gamma_prior buffer "
            "(pass --init_rna_half_life at construction). The prior penalty is a no-op.",
            RuntimeWarning,
        )
        _gamma_prior_warned = True
    return weight, use_prior


def train_epoch(args, model, dataloader, optim, crit, device, accessible_out_ix, mse_crit, use_amp=False):
    """Run one training epoch; return dict with train nll/mse/r2."""
    model.train()
    alpha_decay, na_module, use_alpha_decay = _node_activity_penalty_args(args, model)
    gamma_prior_weight, use_gamma_prior = _gamma_prior_penalty_args(args, model)
    device_type = device.type if isinstance(device, torch.device) else str(device)

    losses = 0.0
    nll_losses = 0.0
    mse_losses = 0.0
    prior_losses = 0.0
    r2s = 0.0
    n_batches = 0
    n_loader = len(dataloader)
    epoch_start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        optim.zero_grad()

        if len(batch) == 3:
            X, dxdt_mu, dxdt_sigma = batch
            x_fn = None
        elif len(batch) == 4:
            X, dxdt_mu, dxdt_sigma, x_fn = batch
            x_fn = x_fn.to(device)
        else:
            raise ValueError(
                f"train_epoch: unexpected batch arity {len(batch)}; expected 3 or 4"
            )

        X = X.to(device)
        dxdt_mu = dxdt_mu.to(device)
        dxdt_sigma = dxdt_sigma.to(device)
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            dxdt_hat = model(X) if x_fn is None else model(X, x_fn=x_fn)

        mu_sub = dxdt_mu[:, accessible_out_ix]
        sigma_sub = dxdt_sigma[:, accessible_out_ix]
        hat_sub = dxdt_hat[:, accessible_out_ix]
        nll_loss = crit(hat_sub, mu_sub, sigma_sub)
        mse_loss = mse_crit(hat_sub, mu_sub)
        # 'mse' trains on target_mu only (target_sigma ignored); 'nll' is the
        # inverse-variance weighted Gaussian NLL. Both metrics are always logged.
        loss = mse_loss if getattr(args, 'objective', 'nll') == 'mse' else nll_loss
        if use_alpha_decay:
            alpha_pen = na_module.get_alpha_mean().mean()
            loss = loss + alpha_decay * alpha_pen
        if use_gamma_prior:
            gamma_pen = model.gamma_prior_loss()
            loss = loss + gamma_prior_weight * gamma_pen
            prior_losses += gamma_pen.item()

        loss.backward()
        clip_grad_norm = float(getattr(args, 'clip_grad_norm', 0.0) or 0.0)
        if clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optim.step()

        losses += loss.item()
        nll_losses += nll_loss.item()
        mse_losses += mse_loss.item()
        batch_r2 = r2_score(
            mu_sub.cpu().numpy(),
            hat_sub.detach().cpu().numpy(),
            multioutput='uniform_average',
        )
        r2s += batch_r2
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
        'loss': losses / n,
    }
    if use_gamma_prior:
        metrics['gamma_prior'] = prior_losses / n
    return metrics


def _pretrain_epoch_row_values(
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
        'val_nll': val_metrics['nll'],
        'val_mse': val_metrics['mse'],
        'val_r2': val_metrics['r2'],
        'lr': lr,
        'best_val': best_val,
        'time_s': time_s,
        'max_mem_gb': max_mem_gb,
    }
    if 'gamma_prior' in train_metrics:
        values['train_gamma_prior'] = train_metrics['gamma_prior']
    return values


def train_with_validation(
    args, model, train_loader, val_loader, optim, scheduler, crit, device, accessible_out_ix,
    *, out_dir, model_id, start_epoch=0, best_val_nll=float('inf'), best_epoch=-1,
    best_val_mse=float('inf'), use_amp=False,
):
    """Train with per-epoch validation; checkpoint on lowest val NLL."""
    alpha_decay, na_module, _ = _node_activity_penalty_args(args, model)
    _, use_gamma_prior = _gamma_prior_penalty_args(args, model)
    mse_crit = torch.nn.MSELoss(reduction='mean')
    history = []
    best_state_dict = None
    history_path = os.path.join(out_dir, f'val_metrics_history_pretrain_{model_id}.csv')

    # Select best checkpoint / step the LR scheduler on the objective being
    # optimized: val MSE for 'mse', val NLL for 'nll'. best_monitored seeds
    # from the resumed best of the matching metric.
    objective = getattr(args, 'objective', 'nll')
    monitor_key = 'mse' if objective == 'mse' else 'nll'
    best_monitored = best_val_mse if monitor_key == 'mse' else best_val_nll

    table_cols = pretrain_epoch_table_columns(use_gamma_prior=use_gamma_prior)
    table_header_line = table_header(table_cols)
    print(f'Training model (train/val split, objective={objective}, monitoring val {monitor_key})...')
    print(table_header_line)
    print('-' * len(table_header_line))

    for epoch in range(start_epoch, args.epochs):
        reset_peak_mem(device)
        tic = time.time()
        train_metrics = train_epoch(
            args, model, train_loader, optim, crit, device, accessible_out_ix, mse_crit,
            use_amp=use_amp,
        )
        val_metrics = evaluate_dxdt(
            model, val_loader, crit, device, accessible_out_ix,
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
            'val_nll': val_metrics['nll'],
            'val_mse': val_metrics['mse'],
            'val_r2': val_metrics['r2'],
            'lr': lr,
            'time_s': time_s,
            'max_mem_gb': max_mem_gb,
        }
        if 'gamma_prior' in train_metrics:
            row['train_gamma_prior'] = train_metrics['gamma_prior']
        history.append(row)
        append_history_row(history_path, row, list(PRETRAIN_HISTORY_COLUMNS))

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
                _pretrain_epoch_row_values(
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
        'best_val_mse': best_val_mse if best_epoch > 0 else None,
        'best_val_r2': history[best_epoch - 1]['val_r2'] if best_epoch > 0 else None,
        'final_val_nll': history[-1]['val_nll'] if history else None,
        'final_val_mse': history[-1]['val_mse'] if history else None,
        'final_val_r2': history[-1]['val_r2'] if history else None,
        'n_train_batches': len(train_loader),
        'n_val_batches': len(val_loader),
    }


def _build_per_cell_dataloaders(
    args, dxdt_meta, input_names, output_names, src_gene_names,
    pred_dir, scale, x_fn_lookup=None, shuffle=True,
):
    """Group ``dxdt_meta`` by ``cell_iname`` and build a DXDTDataset/DataLoader
    per cell line."""
    cell_inames = sorted(dxdt_meta['cell_iname'].unique().tolist())
    dataloaders = {}
    for cell in cell_inames:
        meta_cell = dxdt_meta[dxdt_meta['cell_iname'] == cell]
        if len(meta_cell) == 0:
            continue
        ds = DXDTDataset(
            meta=meta_cell,
            input_names=input_names,
            output_names=output_names,
            src_names=src_gene_names,
            pred_dir=pred_dir,
            scale=scale,
            sigma_floor=args.sigma_floor,
            x_fn_lookup=x_fn_lookup,
        )
        if len(ds) == 0:
            continue
        dataloaders[cell] = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
        )
    return dataloaders


def _hnet_forward_batch(hnet, args, X, dxdt, C, crit_mse, crit_edl, accessible_out_ix):
    """Single hnet train/val forward; returns (loss, mse, r2)."""
    if args.hnet_n_train_samples <= 1:
        z = hnet._sample_z()
        state_dict = hnet.sample(C=C, z=z)
        dxdt_hat = torch.func.functional_call(hnet.model, state_dict, X)
        dxdt_sub = dxdt[:, accessible_out_ix]
        dxdt_hat_sub = dxdt_hat[:, accessible_out_ix]
        mse_loss = crit_mse(dxdt_hat_sub, dxdt_sub)
        r2_val = r2_score(
            dxdt_sub.detach().cpu().numpy(),
            dxdt_hat_sub.detach().cpu().numpy(),
            multioutput='uniform_average',
        )
        return mse_loss, mse_loss, r2_val

    K = args.hnet_n_train_samples
    yhat = hnet(X, samples=K, C=C)
    dxdt_sub = dxdt[:, accessible_out_ix]
    yhat_sub = yhat[:, :, accessible_out_ix]
    if args.hnet_loss == 'edl':
        loss = crit_edl(yhat_sub, dxdt_sub)
    else:
        loss = crit_mse(
            yhat_sub,
            dxdt_sub.unsqueeze(0).expand(K, -1, -1),
        )
    mse_loss = crit_mse(yhat_sub.mean(0), dxdt_sub)
    r2_val = r2_score(
        dxdt_sub.detach().cpu().numpy(),
        yhat_sub.mean(0).detach().cpu().numpy(),
        multioutput='uniform_average',
    )
    return loss, mse_loss, r2_val


@torch.no_grad()
def _evaluate_hnet_per_cell(hnet, args, val_loaders, hnet_cell_index, crit_mse, device, accessible_out_ix):
    """Aggregate validation metrics over per-cell val loaders."""
    hnet.eval()
    mse_sum, r2_sum, n_batches = 0.0, 0.0, 0

    for cell, loader in val_loaders.items():
        C = hnet_cell_index[cell].to(device)
        for X, dxdt in loader:
            X = X.to(device)
            dxdt = dxdt.to(device)
            if args.hnet_n_train_samples <= 1:
                z = hnet._sample_z()
                state_dict = hnet.sample(C=C, z=z)
                dxdt_hat = torch.func.functional_call(hnet.model, state_dict, X)
            else:
                K = args.hnet_n_train_samples
                yhat = hnet(X, samples=K, C=C)
                dxdt_hat = yhat.mean(0)
            dxdt_sub = dxdt[:, accessible_out_ix]
            dxdt_hat_sub = dxdt_hat[:, accessible_out_ix]
            mse_loss = crit_mse(dxdt_hat_sub, dxdt_sub)
            mse_sum += mse_loss.item()
            r2_sum += r2_score(
                dxdt_sub.cpu().numpy(),
                dxdt_hat_sub.cpu().numpy(),
                multioutput='uniform_average',
            )
            n_batches += 1

    n = max(n_batches, 1)
    return {'mse': mse_sum / n, 'r2': r2_sum / n, 'n_batches': n_batches}


def train_hnet_with_validation(
    args, hnet, hnet_cell_index, train_loaders, val_loaders, device, accessible_out_ix,
):
    """Hypernetwork training with per-epoch validation; checkpoint on val MSE."""
    from hnet.train.hnet import EnergyDistanceLoss

    crit_mse = torch.nn.MSELoss(reduction='mean')
    crit_edl = EnergyDistanceLoss() if args.hnet_loss == 'edl' else None

    if args.hnet_loss == 'edl' and args.hnet_n_train_samples <= 1:
        raise ValueError("hnet_loss='edl' requires hnet_n_train_samples > 1")

    optim = torch.optim.AdamW(hnet.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = build_lr_scheduler(optim, patience=args.patience)

    train_cells = sorted(train_loaders.keys())
    print(
        f'Hypernet training on {len(train_cells)} train cell lines; '
        f'{len(val_loaders)} val cell lines'
    )

    history = []
    best_val_mse = float('inf')
    best_state_dict = None
    best_epoch = -1

    for epoch in range(args.epochs):
        tic = time.time()
        hnet.train()
        train_mse, train_r2, n_train_batches = 0.0, 0.0, 0

        for cell in train_cells:
            C = hnet_cell_index[cell].to(device)
            for X, dxdt in train_loaders[cell]:
                optim.zero_grad()
                X = X.to(device)
                dxdt = dxdt.to(device)
                loss, mse_loss, r2_val = _hnet_forward_batch(
                    hnet, args, X, dxdt, C, crit_mse, crit_edl, accessible_out_ix,
                )
                loss.backward()
                optim.step()
                train_mse += mse_loss.item()
                train_r2 += r2_val
                n_train_batches += 1

        n_tr = max(n_train_batches, 1)
        train_metrics = {'mse': train_mse / n_tr, 'r2': train_r2 / n_tr}
        val_metrics = _evaluate_hnet_per_cell(
            hnet, args, val_loaders, hnet_cell_index, crit_mse, device, accessible_out_ix,
        )
        scheduler.step(val_metrics['mse'])
        lr = scheduler.get_last_lr()[0]

        row = {
            'epoch': epoch + 1,
            'train_mse': train_metrics['mse'],
            'train_r2': train_metrics['r2'],
            'val_mse': val_metrics['mse'],
            'val_r2': val_metrics['r2'],
            'lr': lr,
            'time_s': time.time() - tic,
        }
        history.append(row)

        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_state_dict = copy.deepcopy(hnet.state_dict())
            best_epoch = epoch + 1

        print(
            f'--> epoch {epoch+1}/{args.epochs} | train mse: {train_metrics["mse"]:.4E} '
            f'r2: {train_metrics["r2"]:.3f} | val mse: {val_metrics["mse"]:.4E} '
            f'r2: {val_metrics["r2"]:.3f} | lr: {lr:.2E} | '
            f'best_val_mse: {best_val_mse:.4E} (epoch {best_epoch}) | '
            f'time: {row["time_s"]:.2f}s'
        )

    if best_state_dict is not None:
        hnet.load_state_dict(best_state_dict)
    return hnet, pd.DataFrame(history), {
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'best_val_r2': history[best_epoch - 1]['val_r2'] if best_epoch > 0 else None,
        'final_val_mse': history[-1]['val_mse'] if history else None,
        'final_val_r2': history[-1]['val_r2'] if history else None,
    }


def _save_pretrain_metrics(out_dir, model_id, history_df, summary):
    os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, f'val_metrics_history_pretrain_{model_id}.csv')
    summary_path = os.path.join(out_dir, f'val_metrics_pretrain_{model_id}.json')
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

    # ------------------------------------------------------------------
    # Mutual-exclusion guard: node_activity and the hypernetwork both
    # supply cell-line conditioning information through different
    # mechanisms; mixing them is currently unsupported.
    # ------------------------------------------------------------------
    if args.node_activity and args.use_hypernetwork:
        raise ValueError(
            "--node_activity and --use_hypernetwork are mutually exclusive. "
            "Pick one cell-line conditioning mechanism."
        )

    dxdt_meta = pd.read_csv(f'{args.data}/dxdt_meta.csv')
    src_gene_names = pd.read_csv(f'{args.data}/gene_names.csv')['gene_names'].tolist()
    data = torch.load(f'{args.bionet}/bionetwork.pt', weights_only=False)

    # ------------------------------------------------------------------
    # Load the per-gene mu/sigma artifact required by BIOGSNN's log1p-back
    # degradation term.
    # ------------------------------------------------------------------
    gn_path = args.gene_norm_path or os.path.join(args.bionet, 'gene_norm.pt')
    if not os.path.exists(gn_path):
        raise FileNotFoundError(
            f"gene_norm.pt not found at {gn_path}. Rebuild the bionetwork with "
            "`make_bio_network.py --gene_stats_path <gene_stats.dict>` or pass "
            "--gene_norm_path explicitly."
        )
    gene_norm = load_gene_norm_artifact(gn_path, output_names=data.node_names_dict['output'])
    print(f"gene_norm: loaded {len(gene_norm['gene_names'])} per-gene mu/sigma rows from {gn_path}")

    # ------------------------------------------------------------------
    # Load the per-cell-line x_fn artifact when node_activity is enabled.
    # Building it inline here would re-read the full DepMap expression
    # matrix every run, so we require it to be precomputed by
    # make_bio_network.py --node_activity.
    # ------------------------------------------------------------------
    x_fn_lookup = None
    node_activity_dim = 1
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
        node_activity_dim = int(na_payload['activity_dim'])
        print(f"node_activity: loaded {len(x_fn_lookup)} cell-line activity rows "
              f"(activity_dim={node_activity_dim}) from {na_path}")

    pert_ids_net = [x.split('__')[1] for x in data.node_names_dict['input'] if 'DRUG__' in x]
    pert_ids_meta = dxdt_meta['pert_id'].unique().tolist()
    missing_ids = set(pert_ids_meta) - set(pert_ids_net)
    print(f'Some perts in meta are not in network: {len(missing_ids)} (showing up to 5): {sorted(missing_ids)[:5]}')
    dxdt_meta = dxdt_meta[dxdt_meta['pert_id'].isin(pert_ids_net)]

    pert_ids = sorted(dxdt_meta['pert_id'].unique().tolist())
    cell_inames = sorted(dxdt_meta['cell_iname'].unique().tolist())
    n_val = int(getattr(args, 'val_cells_per_drug', 1) or 1)
    split_df = build_cell_drug_split(
        pert_ids=pert_ids,
        cell_inames=cell_inames,
        n_val=n_val,
        seed=int(args.seed),
    )
    train_meta = filter_meta_by_partition(dxdt_meta, split_df, 'train')
    val_meta = filter_meta_by_partition(dxdt_meta, split_df, 'val')
    if len(train_meta) == 0:
        raise ValueError("train partition is empty after applying per-replicate cell_drug_split")
    if len(val_meta) == 0:
        raise ValueError(
            "val partition is empty after applying per-replicate cell_drug_split "
            "(check node_activity x_fn_lookup did not drop all val cells)"
        )
    print(
        f'cell_drug_split (seed={args.seed}): train rows={len(train_meta)}, val rows={len(val_meta)} '
        f'(pairs train={split_df[split_df.partition=="train"].shape[0]}, '
        f'val={split_df[split_df.partition=="val"].shape[0]})'
    )

    print('# output nodes', len(data.node_names_dict['output']))
    print(f'Training model replicate: {args.model_id}')

    pred_dir = args.data

    # Estimate normalization scale on train only to avoid val leakage.
    train_scale_dataset = DXDTDataset(
        train_meta,
        input_names=data.node_names_dict['input'],
        output_names=data.node_names_dict['output'],
        src_names=src_gene_names,
        pred_dir=pred_dir,
        sigma_floor=args.sigma_floor,
        x_fn_lookup=x_fn_lookup,
    )
    if len(train_scale_dataset) == 0:
        raise ValueError("train DXDTDataset is empty after filtering")
    dxdt_scale = train_scale_dataset._scale

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.amp and device.type == 'cuda'
    use_tf32 = args.tf32 and device.type == 'cuda'
    if args.amp and device.type != 'cuda':
        print('--amp ignored (CUDA not available)')
    if args.tf32 and device.type != 'cuda':
        print('--tf32 ignored (CUDA not available)')
    configure_cuda_performance(use_tf32)
    print(f'Using device: {device}')
    print(f'AMP: {use_amp}')
    print(f'TF32: {use_tf32}')
    print(f'optimizer: {args.optimizer}')

    accessible_mask = get_or_compute_drug_accessible_mask(data)
    accessible_out_ix = accessible_indices(accessible_mask).to(device)
    n_acc = int(accessible_mask.sum())
    n_total = len(data.node_names_dict['output'])
    print(f'Drug-accessible output genes: {n_acc} / {n_total}')

    gsnn_kwargs = _gsnn_kwargs_from_args(args, data, node_activity_dim=node_activity_dim)
    # gsnn_kwargs contains the bionetwork dicts — strip them when serializing
    # since they are already part of bionetwork.pt and would bloat the artifact.
    gsnn_kwargs_serializable = {
        k: v for k, v in gsnn_kwargs.items()
        if k not in ("edge_index_dict", "node_names_dict")
    }

    if not args.use_hypernetwork:
        # ----------------------------------------------------------------
        # Legacy GSNN path with train/val split and best-checkpoint selection.
        # ----------------------------------------------------------------
        train_dataset = DXDTDataset(
            train_meta,
            input_names=data.node_names_dict['input'],
            output_names=data.node_names_dict['output'],
            src_names=src_gene_names,
            pred_dir=pred_dir,
            scale=dxdt_scale,
            sigma_floor=args.sigma_floor,
            x_fn_lookup=x_fn_lookup,
        )
        val_dataset = DXDTDataset(
            val_meta,
            input_names=data.node_names_dict['input'],
            output_names=data.node_names_dict['output'],
            src_names=src_gene_names,
            pred_dir=pred_dir,
            scale=dxdt_scale,
            sigma_floor=args.sigma_floor,
            x_fn_lookup=x_fn_lookup,
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

        if args.BIOGSNN:
            model = BIOGSNN(
                gsnn_kwargs=gsnn_kwargs,
                gene_norm=gene_norm,
                dxdt_nonlin=args.dxdt_nonlin,
                init_rna_half_life=args.init_rna_half_life,
                dxdt_scale=(dxdt_scale if args.init_rna_half_life is not None else None),
                dxdt_clip=args.dxdt_clip,
            ).to(device)
        else:
            model = GSNN(**gsnn_kwargs).to(device)

        print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

        optim = build_optimizer(model, args.optimizer, lr=args.lr, wd=args.wd)
        scheduler = build_lr_scheduler(optim, patience=args.patience)
        crit = GaussianNLL()

        model, start_epoch, best_val_nll, best_epoch, best_val_mse, _ = try_load_resume(
            args.out,
            args.model_id,
            resume_incomplete=args.resume_incomplete,
            device=device,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
        )
        if args.resume_incomplete and start_epoch > 0:
            print(
                f'resuming incomplete pretrain from epoch {start_epoch} '
                f'(best_val_nll={best_val_nll:.4E} at epoch {best_epoch})'
            )

        if start_epoch >= args.epochs:
            print(
                f'pretrain already complete ({start_epoch} epochs logged, '
                f'epochs={args.epochs}); writing final artifacts from checkpoint'
            )
            model = load_best_model(args.out, args.model_id, device)
            summary = {
                'best_epoch': best_epoch,
                'best_val_nll': best_val_nll,
                'best_val_mse': best_val_mse,
                'final_val_nll': best_val_nll,
                'final_val_mse': best_val_mse,
                'n_train_batches': len(train_loader),
                'n_val_batches': len(val_loader),
            }
            history_df = pd.read_csv(
                os.path.join(args.out, f'val_metrics_history_pretrain_{args.model_id}.csv')
            ) if os.path.exists(
                os.path.join(args.out, f'val_metrics_history_pretrain_{args.model_id}.csv')
            ) else pd.DataFrame()
            _save_pretrain_metrics(args.out, args.model_id, history_df, summary)
            torch.save(model, f'{args.out}/pretrained_model_{args.model_id}.pt')
            torch.save(torch.tensor([dxdt_scale]), f'{args.out}/dxdt_scale_{args.model_id}.pt')
        else:
            model, history_df, summary = train_with_validation(
                args, model, train_loader, val_loader, optim, scheduler, crit, device,
                accessible_out_ix,
                out_dir=args.out,
                model_id=args.model_id,
                start_epoch=start_epoch,
                best_val_nll=best_val_nll,
                best_epoch=best_epoch,
                best_val_mse=best_val_mse if best_val_mse is not None else float('inf'),
                use_amp=use_amp,
            )
            _save_pretrain_metrics(args.out, args.model_id, history_df, summary)

            torch.save(model, f'{args.out}/pretrained_model_{args.model_id}.pt')
            torch.save(torch.tensor([dxdt_scale]), f'{args.out}/dxdt_scale_{args.model_id}.pt')

    else:
        raise ValueError(
            "Hypernetwork mode is not supported with the consolidated predict_grid "
            "format and NLL objective in this adaptation. Disable hypernetwork.enabled."
        )
        # Legacy hypernetwork path below is unreachable; kept for reference during migration.
        # ----------------------------------------------------------------
        # Hypernetwork path. Imports are local so legacy users don't need
        # the `hnet` package installed.
        # ----------------------------------------------------------------
        from lincs_gsnn.models.HnetGSNN import (
            build_gsnn_template,
            build_hnet,
            cell_lines_from_bionet,
            cell_onehot,
            gsnn_init_dict,
            materialize_gsnn,
            save_hnet_artifact,
            soft_mean_C,
        )
        from hnet.train.hnet import init_hnet

        cell_lines = cell_lines_from_bionet(data)
        if len(cell_lines) == 0:
            raise ValueError(
                "Hypernetwork mode requires LINE__ entries in "
                "data.node_names_dict['input'] (cell-line vocabulary). "
                "Found none in the bionetwork."
            )

        hnet_cfg = dict(
            stochastic_channels=args.hnet_stochastic_channels,
            width=args.hnet_width,
            pz=args.hnet_pz,
            learn_pz=args.hnet_learn_pz,
            affine=args.hnet_affine,
            norm=args.hnet_norm,
            dropout=args.hnet_dropout,
            bias=args.hnet_bias,
            embed_dim=args.hnet_embed_dim,
        )

        gsnn_template = build_gsnn_template(data, gsnn_kwargs_serializable)
        n_gsnn_params = sum(p.numel() for p in gsnn_template.parameters() if p.requires_grad)
        n_fphi_out = n_gsnn_params  # f_phi outputs nparams entries
        print(f'# GSNN parameters: {n_gsnn_params}')
        print(f'# f_phi output features: {n_fphi_out} (last linear: width x n_params = '
              f'{args.hnet_width} x {n_gsnn_params} = {args.hnet_width * n_gsnn_params})')

        hnet = build_hnet(gsnn_template, n_cell_lines=len(cell_lines), hnet_cfg=hnet_cfg).to(device)

        if args.hnet_pretrain_init:
            print('Warm-starting f_phi to match GSNN initialization distribution...')
            init_dict = gsnn_init_dict(gsnn_template)
            hnet = init_hnet(hnet, init_dict, samples=64, iters=100, lr=1e-3, verbose=True)
            print()

        print('# total hnet parameters:', sum(p.numel() for p in hnet.parameters() if p.requires_grad))

        # Pre-compute one-hot vectors per cell (kept on CPU; .to(device) at use).
        hnet_cell_index = {
            cell: cell_onehot(cell, cell_lines, device=None) for cell in cell_lines
        }

        train_loaders = _build_per_cell_dataloaders(
            args, train_meta,
            input_names=data.node_names_dict['input'],
            output_names=data.node_names_dict['output'],
            src_gene_names=src_gene_names,
            obs_dir=obs_dir,
            dxdt_dir=dxdt_dir,
            scale=dxdt_scale,
            x_fn_lookup=x_fn_lookup,
            shuffle=True,
        )
        val_loaders = _build_per_cell_dataloaders(
            args, val_meta,
            input_names=data.node_names_dict['input'],
            output_names=data.node_names_dict['output'],
            src_gene_names=src_gene_names,
            obs_dir=obs_dir,
            dxdt_dir=dxdt_dir,
            scale=dxdt_scale,
            x_fn_lookup=x_fn_lookup,
            shuffle=False,
        )
        if not val_loaders:
            raise ValueError("hypernetwork val loaders are empty after split")

        hnet, history_df, summary = train_hnet_with_validation(
            args, hnet, hnet_cell_index, train_loaders, val_loaders, device, accessible_out_ix,
        )
        _save_pretrain_metrics(args.out, args.sample, history_df, summary)

        # ----------------------------------------------------------------
        # Save artifacts.
        # ----------------------------------------------------------------
        hnet_path = f'{args.out}/pretrained_hnet_{args.sample}.pt'
        save_hnet_artifact(
            path=hnet_path,
            hnet=hnet.cpu(),
            cell_lines=cell_lines,
            gsnn_kwargs=gsnn_kwargs_serializable,
            hnet_cfg=hnet_cfg,
        )
        # Restore device so the legacy artifact is materialized correctly below.
        hnet = hnet.to(device)

        # Backward-compat: produce a vanilla GSNN at a "mean cell line" so any
        # downstream code that just torch.loads pretrained_model_{sample}.pt
        # still gets a usable, deterministic GSNN.
        mean_C = soft_mean_C(cell_lines, device=device)
        mean_template = build_gsnn_template(data, gsnn_kwargs_serializable).to(device)
        mean_gsnn = materialize_gsnn(hnet, mean_C, template=mean_template)
        mean_gsnn = mean_gsnn.eval()
        torch.save(mean_gsnn, f'{args.out}/pretrained_model_{args.sample}.pt')

        torch.save(torch.tensor([dxdt_scale]), f'{args.out}/dxdt_scale_{args.sample}.pt')
        print(f'Saved hypernetwork artifact to {hnet_path}')
