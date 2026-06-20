"""
pretrain_gsnn_with_fei.py - Pretrain a GSNN on dx/dt with per-epoch FEI validation.

Extends the dx/dt pretrain loop so FunctionEdgeInferer runs on held-out validation
dxdt batches each epoch. The checkpoint with the best FEI val metric (configurable:
mrr, top1, top10, top100, auroc, aupr) is saved along with the dense W matrix.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from gsnn.models.GSNN import GSNN
from gsnn.optim.FunctionEdgeInferer import FunctionEdgeInferer, mrr as fei_mrr
from torch.utils.data import DataLoader

from lincs_gsnn.data.DXDTDataset import DXDTDataset
from lincs_gsnn.data.dxdt_meta import filter_min_dose, subsample as subsample_dxdt_meta
from lincs_gsnn.models.BIOGSNN import BIOGSNN
from lincs_gsnn.proc.cell_drug_split import (
    filter_meta_by_partition,
    load_cell_drug_split,
)
from lincs_gsnn.proc.drug_accessibility import (
    accessible_indices,
    get_or_compute_drug_accessible_mask,
)
from lincs_gsnn.proc.gene_norm import load_gene_norm_artifact
from lincs_gsnn.train.metrics import evaluate_dxdt
from lincs_gsnn.utils.eval import eval_edge_inference_any

# Reuse helpers from the legacy pretrain script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pretrain_gsnn_with_dxdt import (  # noqa: E402
    _gamma_prior_penalty_args,
    _gsnn_kwargs_from_args,
    _node_activity_penalty_args,
    train_epoch,
)

FEI_SELECT_METRICS = {
    'mrr': 'mrr',
    'top1': 'top1_acc',
    'top10': 'top10_acc',
    'top100': 'top100_acc',
    'auroc': 'auroc',
    'aupr': 'aupr',
}


def get_args():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # Shared pretrain flags (mirrors pretrain_gsnn_with_dxdt.py).
    # ------------------------------------------------------------------
    parser.add_argument('--data', type=str, default='../../../data/')
    parser.add_argument('--out', type=str, default='../../proc/')
    parser.add_argument('--bionet', type=str, default='../../proc/bionetwork.pt')
    parser.add_argument('--sample', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--share_layers', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='batch')
    parser.add_argument('--checkpoint', action='store_true', default=False)
    parser.add_argument('--init', type=str, default='degree_normalized')
    parser.add_argument('--add_function_self_edges', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bias', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--residual', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--node_mlp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--node_mlp_dim', type=int, default=128)
    parser.add_argument('--node_attn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--attn_mlp_hidden', type=int, default=32)
    parser.add_argument('--use_hypernetwork', action='store_true', default=False)
    parser.add_argument('--node_activity', action='store_true', default=False)
    parser.add_argument('--node_activity_path', type=str, default=None)
    parser.add_argument('--node_activity_hidden', type=int, default=16)
    parser.add_argument('--node_activity_temperature', type=float, default=1.0)
    parser.add_argument('--node_activity_dropout', type=float, default=0.0)
    parser.add_argument('--node_activity_mode', type=str, default='per-node',
                        choices=['per-node', 'per-channel'])
    parser.add_argument('--alpha_decay', type=float, default=1e-2)
    parser.add_argument('--gene_norm_path', type=str, default=None)
    parser.add_argument('--split_path', type=str, default=None)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--GSNN', action='store_true', default=False)
    model_group.add_argument('--BIOGSNN', action='store_true', default=False)
    parser.add_argument('--dxdt_nonlin', type=str, default=None,
                        choices=['relu', 'leaky_relu', 'sigmoid', 'softplus', 'elu', 'selu', 'gelu', 'swish'])
    parser.add_argument('--init_rna_half_life', type=float, default=None)
    parser.add_argument('--gamma_prior_weight', type=float, default=0.0)
    parser.add_argument('--removed_edges_path', type=str, default=None,
                        help='Path to removed_edges.csv (defaults to <bionet>/removed_edges.csv)')

    # ------------------------------------------------------------------
    # FEI-specific flags.
    # ------------------------------------------------------------------
    parser.add_argument('--fei_score_method', type=str, default='spearman',
                        choices=['spearman', 'pearson'])
    parser.add_argument('--fei_norm', type=str, default='l1', choices=['l1', 'l2', 'none'])
    parser.add_argument('--fei_agg', type=str, default='sum', choices=['sum', 'mean', 'max'])
    parser.add_argument('--fei_use_prenorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fei_penalty_factor', type=float, default=0.0)
    parser.add_argument('--fei_scale_by_act_mean', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fei_estimate', action=argparse.BooleanOptionalAction, default=False,
                        help='Average FEI matrices from estimate_iters subsampled passes')
    parser.add_argument('--fei_estimate_iters', type=int, default=10,
                        help='Number of estimate passes to average when --fei_estimate')
    parser.add_argument('--fei_estimate_n_samples', type=int, default=2500,
                        help='Observations sampled per estimate pass (with replacement)')
    parser.add_argument('--fei_row_chunk', type=int, default=None)
    parser.add_argument('--fei_subsample', type=float, default=1.0,
                        help='Fraction of val dxdt_meta rows for FEI (1.0 = all)')
    parser.add_argument('--fei_subsample_seed', type=int, default=0)
    parser.add_argument('--fei_min_dose_um', type=float, default=None,
                        help='Keep val rows with dose >= this (µM); omit for no filter')
    parser.add_argument('--fei_max_val_rows', type=int, default=None,
                        help='Hard cap on val observations stacked for FEI')
    parser.add_argument('--fei_select_metric', type=str, default='mrr',
                        choices=sorted(FEI_SELECT_METRICS.keys()))
    parser.add_argument('--fei_select_freq', type=int, default=1,
                        help='Evaluate FEI every N epochs')
    parser.add_argument('--fei_verbose', action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def _edges_df_to_index(edge_df: pd.DataFrame, node_names: list[str]) -> torch.Tensor:
    name2idx = {n: i for i, n in enumerate(node_names)}
    src_idx, dst_idx = [], []
    for _, row in edge_df.iterrows():
        s, d = row['source'], row['target']
        if s in name2idx and d in name2idx:
            src_idx.append(name2idx[s])
            dst_idx.append(name2idx[d])
    if not src_idx:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.stack([
        torch.tensor(src_idx, dtype=torch.long),
        torch.tensor(dst_idx, dtype=torch.long),
    ], dim=0)


def _load_holdout_edges(removed_edges_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = pd.read_csv(removed_edges_path, low_memory=False)
    holdout = holdout.rename(columns={'src_name': 'source', 'dst_name': 'target'})
    if 'split' in holdout.columns:
        val_df = holdout[holdout['split'] == 'val'][['source', 'target']].copy()
        test_df = holdout[holdout['split'] == 'test'][['source', 'target']].copy()
    else:
        val_df = holdout.iloc[0:0][['source', 'target']].copy()
        test_df = holdout[['source', 'target']].copy()
    return holdout[['source', 'target']], val_df, test_df


def _load_train_edges_df(data) -> pd.DataFrame:
    node_names = np.array(data.node_names_dict['function'])
    src = data.edge_index_dict['function', 'to', 'function'][0]
    dst = data.edge_index_dict['function', 'to', 'function'][1]
    return pd.DataFrame({
        'source': node_names[src.cpu().numpy()],
        'target': node_names[dst.cpu().numpy()],
    })


def _build_edge_context(data, removed_edges_path: str):
    node_names = list(data.node_names_dict['function'])
    train_edges_df = _load_train_edges_df(data)
    _, val_df, test_df = _load_holdout_edges(removed_edges_path)

    val_edge_index = _edges_df_to_index(val_df, node_names)
    test_edge_index = _edges_df_to_index(test_df, node_names)
    train_edge_index = data.edge_index_dict['function', 'to', 'function'].cpu()

    all_true_edges = torch.cat(
        [train_edge_index, val_edge_index, test_edge_index],
        dim=1,
    )
    return {
        'node_names': node_names,
        'train_edges_df': train_edges_df,
        'val_df': val_df,
        'test_df': test_df,
        'val_edge_index': val_edge_index,
        'all_true_edges': all_true_edges,
    }


def _prepare_fei_val_meta(val_meta: pd.DataFrame, args) -> pd.DataFrame:
    meta = val_meta.copy()
    if args.fei_min_dose_um is not None:
        n_before = len(meta)
        meta = filter_min_dose(meta, args.fei_min_dose_um)
        if args.fei_verbose:
            print(
                f'FEI min_dose_um filter: >= {args.fei_min_dose_um} '
                f'rows={len(meta)}/{n_before}',
                flush=True,
            )
    n_before = len(meta)
    meta = subsample_dxdt_meta(meta, args.fei_subsample, seed=args.fei_subsample_seed)
    if args.fei_verbose:
        print(
            f'FEI subsample: frac={args.fei_subsample} seed={args.fei_subsample_seed} '
            f'rows={len(meta)}/{n_before}',
            flush=True,
        )
    if len(meta) == 0:
        raise ValueError('FEI val meta is empty after min_dose/subsample filtering')
    return meta.reset_index(drop=True)


def _stack_fei_val_tensors(
    fei_val_meta: pd.DataFrame,
    dataset_kwargs: dict,
    batch_size: int,
    max_val_rows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = DXDTDataset(meta=fei_val_meta, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    x_parts, y_parts = [], []
    for batch in loader:
        if len(batch) == 2:
            x_batch, y_batch = batch
        elif len(batch) == 3:
            x_batch, y_batch, _x_fn = batch
        else:
            raise ValueError(f'Unexpected batch arity {len(batch)} for FEI val loader')
        x_parts.append(x_batch)
        y_parts.append(y_batch)
        if max_val_rows is not None and sum(t.shape[0] for t in x_parts) >= max_val_rows:
            break

    x_val = torch.cat(x_parts, dim=0)
    y_val = torch.cat(y_parts, dim=0)
    if max_val_rows is not None:
        x_val = x_val[:max_val_rows]
        y_val = y_val[:max_val_rows]
    return x_val.contiguous(), y_val.contiguous()


def _W_to_eval_df(W: np.ndarray, node_names: list[str]) -> pd.DataFrame:
    n = len(node_names)
    rows = []
    for i in range(n):
        src = node_names[i]
        if not src.startswith('PROTEIN__'):
            continue
        for j in range(n):
            if i == j:
                continue
            dst = node_names[j]
            if not (dst.startswith('PROTEIN__') or dst.startswith('RNA__')):
                continue
            rows.append({'source': src, 'target': dst, 'score': float(W[i, j])})
    return pd.DataFrame(rows)


def _tag_eval_df(
    preds: pd.DataFrame,
    train_edges_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    res = preds.copy()
    train_tag = train_edges_df.assign(train_edge=True)[['source', 'target', 'train_edge']]
    val_tag = val_df.assign(val_edge=True)[['source', 'target', 'val_edge']]
    test_tag = test_df.assign(test_edge=True)[['source', 'target', 'test_edge']]
    res = res.merge(train_tag, on=['source', 'target'], how='left')
    res = res.merge(val_tag, on=['source', 'target'], how='left')
    res = res.merge(test_tag, on=['source', 'target'], how='left')
    for col in ('train_edge', 'val_edge', 'test_edge'):
        res[col] = res[col].fillna(False).astype(bool)
    return res


def _score_W(
    W: np.ndarray,
    edge_ctx: dict,
    data,
    select_metric: str,
    device: torch.device,
) -> float:
    metric_key = FEI_SELECT_METRICS[select_metric]

    if select_metric == 'mrr':
        val_ei = edge_ctx['val_edge_index']
        if val_ei.shape[1] == 0:
            return float('nan')
        w_t = torch.as_tensor(W, dtype=torch.float32, device=device)
        all_true = edge_ctx['all_true_edges'].to(device)
        val_ei = val_ei.to(device)
        return float(fei_mrr(w_t, val_ei, all_true))

    preds = _W_to_eval_df(W, edge_ctx['node_names'])
    if preds.empty:
        return float('nan')
    tagged = _tag_eval_df(
        preds,
        edge_ctx['train_edges_df'],
        edge_ctx['val_df'],
        edge_ctx['test_df'],
    )
    results = eval_edge_inference_any(tagged, data, 'score')
    val_metrics = results['overall'].get('val', {})
    value = val_metrics.get(metric_key)
    if value is None:
        return float('nan')
    return float(value)


def _run_fei(
    model,
    crit,
    data,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args,
    device: torch.device,
) -> np.ndarray:
    if getattr(model, 'checkpoint', False):
        model.checkpoint = False

    edge_index = data.edge_index_dict['function', 'to', 'function']
    inferer = FunctionEdgeInferer(
        model,
        crit,
        edge_index=edge_index,
        use_prenorm=args.fei_use_prenorm,
        device=device,
        norm=args.fei_norm,
        agg=args.fei_agg,
    )
    x_dev = x_val.to(device)
    y_dev = y_val.to(device)
    return inferer.fit(
        x_dev,
        y_dev,
        method=args.fei_score_method,
        penalty_factor=args.fei_penalty_factor,
        scale_by_act_mean=args.fei_scale_by_act_mean,
        estimate=args.fei_estimate,
        estimate_iters=args.fei_estimate_iters,
        estimate_n_samples=args.fei_estimate_n_samples,
        row_chunk=args.fei_row_chunk,
        verbose=args.fei_verbose,
    )


def train_with_fei_validation(
    args,
    model,
    train_loader,
    val_loader,
    optim,
    scheduler,
    crit,
    device,
    accessible_out_ix,
    x_val_fei,
    y_val_fei,
    edge_ctx,
    data,
):
    alpha_decay, na_module, _ = _node_activity_penalty_args(args, model)
    history = []
    best_fei_score = float('-inf')
    best_state_dict = None
    best_W = None
    best_epoch = -1
    best_val_mse = float('inf')
    best_val_r2 = None

    select_metric = args.fei_select_metric
    metric_col = f'fei_{select_metric}'

    print('Training model (dxdt + FEI val selection)...')
    print()

    for epoch in range(args.epochs):
        tic = time.time()
        train_metrics = train_epoch(
            args, model, train_loader, optim, crit, device, accessible_out_ix,
        )
        val_metrics = evaluate_dxdt(
            model, val_loader, crit, device, accessible_out_ix,
            alpha_decay=alpha_decay, na_module=na_module,
        )
        scheduler.step(val_metrics['mse'])
        lr = scheduler.get_last_lr()[0]

        fei_score = float('nan')
        fei_n_obs = int(x_val_fei.shape[0])
        run_fei = ((epoch + 1) % args.fei_select_freq == 0) or (epoch == args.epochs - 1)

        if run_fei:
            model.eval()
            with torch.set_grad_enabled(True):
                W = _run_fei(model, crit, data, x_val_fei, y_val_fei, args, device)
            fei_score = _score_W(W, edge_ctx, data, select_metric, device)
            if args.fei_verbose:
                print(
                    f'    FEI val {select_metric}={fei_score:.4f} '
                    f'(n_obs={fei_n_obs})',
                    flush=True,
                )
            if np.isfinite(fei_score) and fei_score > best_fei_score:
                best_fei_score = fei_score
                best_state_dict = copy.deepcopy(model.state_dict())
                best_W = W.copy()
                best_epoch = epoch + 1
                best_val_mse = val_metrics['mse']
                best_val_r2 = val_metrics['r2']

        row = {
            'epoch': epoch + 1,
            'train_mse': train_metrics['mse'],
            'train_r2': train_metrics['r2'],
            'val_mse': val_metrics['mse'],
            'val_r2': val_metrics['r2'],
            'lr': lr,
            'time_s': time.time() - tic,
            metric_col: fei_score,
            'fei_n_obs': fei_n_obs,
        }
        if 'gamma_prior' in train_metrics:
            row['train_gamma_prior'] = train_metrics['gamma_prior']
        history.append(row)

        prior_msg = (
            f' | gamma_prior: {train_metrics["gamma_prior"]:.2E}'
            if 'gamma_prior' in train_metrics else ''
        )
        fei_msg = (
            f' | fei {select_metric}: {fei_score:.4f}'
            if np.isfinite(fei_score) else ' | fei: n/a'
        )
        print(
            f'--> epoch {epoch+1}/{args.epochs} | train mse: {train_metrics["mse"]:.4E} '
            f'r2: {train_metrics["r2"]:.3f} | val mse: {val_metrics["mse"]:.4E} '
            f'r2: {val_metrics["r2"]:.3f}{fei_msg} | lr: {lr:.2E}{prior_msg} | '
            f'best_fei_{select_metric}: {best_fei_score:.4f} (epoch {best_epoch}) | '
            f'time: {row["time_s"]:.2f}s',
            flush=True,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    elif best_W is None:
        warnings.warn(
            'No FEI checkpoint was selected (all scores NaN?). '
            'Running a final FEI pass on the last epoch weights.',
            RuntimeWarning,
        )
        model.eval()
        best_W = _run_fei(model, crit, data, x_val_fei, y_val_fei, args, device)
        best_epoch = args.epochs
        best_fei_score = _score_W(best_W, edge_ctx, data, select_metric, device)
        best_val_mse = history[-1]['val_mse'] if history else float('nan')
        best_val_r2 = history[-1]['val_r2'] if history else None

    summary = {
        'best_epoch': best_epoch,
        'best_val_mse': best_val_mse,
        'best_val_r2': best_val_r2,
        'best_fei_metric': select_metric,
        'best_fei_score': best_fei_score,
        'best_fei_epoch': best_epoch,
        'fei_n_obs': int(x_val_fei.shape[0]),
        'final_val_mse': history[-1]['val_mse'] if history else None,
        'final_val_r2': history[-1]['val_r2'] if history else None,
        'n_train_batches': len(train_loader),
        'n_val_batches': len(val_loader),
    }
    return model, best_W, pd.DataFrame(history), summary


def _save_fei_artifacts(out_dir, sample, history_df, summary, W):
    os.makedirs(out_dir, exist_ok=True)
    history_path = os.path.join(out_dir, f'val_metrics_history_pretrain_{sample}.csv')
    fei_history_path = os.path.join(out_dir, f'fei_history_{sample}.csv')
    summary_path = os.path.join(out_dir, f'val_metrics_pretrain_{sample}.json')
    w_path = os.path.join(out_dir, f'fei_W_{sample}.pt')

    history_df.to_csv(history_path, index=False)
    history_df.to_csv(fei_history_path, index=False)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    torch.save({'W': W, 'summary': summary}, w_path)
    print(f'Saved validation metrics to {history_path} and {summary_path}')
    print(f'Saved FEI history to {fei_history_path}')
    print(f'Saved FEI W matrix to {w_path}')


if __name__ == '__main__':
    args = get_args()
    print('--' * 40)
    print('Arguments:')
    print(args)
    print('--' * 40)

    if args.use_hypernetwork:
        raise ValueError(
            'pretrain_gsnn_with_fei does not support --use_hypernetwork in v1 '
            '(FunctionEdgeInferer requires a direct GSNN during training).'
        )
    if args.node_activity:
        raise ValueError(
            'pretrain_gsnn_with_fei does not support --node_activity in v1 '
            '(FunctionEdgeInferer calls model(x) without x_fn).'
        )
    if args.fei_select_freq < 1:
        raise ValueError('--fei_select_freq must be >= 1')

    dxdt_meta = pd.read_csv(f'{args.data}/dxdt_meta.csv')
    src_gene_names = pd.read_csv(f'{args.data}/gene_names.csv')['gene_names'].tolist()
    data = torch.load(f'{args.bionet}/bionetwork.pt', weights_only=False)

    gn_path = args.gene_norm_path or os.path.join(args.bionet, 'gene_norm.pt')
    if not os.path.exists(gn_path):
        raise FileNotFoundError(
            f'gene_norm.pt not found at {gn_path}. Rebuild the bionetwork with '
            '`make_bio_network.py --gene_stats_path <gene_stats.dict>` or pass '
            '--gene_norm_path explicitly.'
        )
    gene_norm = load_gene_norm_artifact(gn_path, output_names=data.node_names_dict['output'])
    print(f'gene_norm: loaded {len(gene_norm["gene_names"])} per-gene mu/sigma rows from {gn_path}')

    pert_ids_net = [x.split('__')[1] for x in data.node_names_dict['input'] if 'DRUG__' in x]
    missing_ids = set(dxdt_meta['pert_id'].unique()) - set(pert_ids_net)
    if missing_ids:
        print(f'Some drugs in meta are not in network: {missing_ids}')
    dxdt_meta = dxdt_meta[dxdt_meta['pert_id'].isin(pert_ids_net)]

    split_path = args.split_path or os.path.join(args.bionet, 'cell_drug_split.csv')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f'cell_drug_split not found at {split_path}. Rebuild the bionetwork '
            'with make_bio_network.py or pass --split_path explicitly.'
        )
    split_df = load_cell_drug_split(split_path)
    train_meta = filter_meta_by_partition(dxdt_meta, split_df, 'train')
    val_meta = filter_meta_by_partition(dxdt_meta, split_df, 'val')
    if len(train_meta) == 0:
        raise ValueError('train partition is empty after applying cell_drug_split')
    if len(val_meta) == 0:
        raise ValueError('val partition is empty after applying cell_drug_split')

    print(
        f'cell_drug_split: train rows={len(train_meta)}, val rows={len(val_meta)} '
        f'(pairs train={split_df[split_df.partition=="train"].shape[0]}, '
        f'val={split_df[split_df.partition=="val"].shape[0]})'
    )

    removed_edges_path = args.removed_edges_path or os.path.join(args.bionet, 'removed_edges.csv')
    if not os.path.exists(removed_edges_path):
        raise FileNotFoundError(
            f'removed_edges.csv not found at {removed_edges_path}. '
            'Rebuild the bionetwork with holdout edges or pass --removed_edges_path.'
        )
    edge_ctx = _build_edge_context(data, removed_edges_path)
    if edge_ctx['val_edge_index'].shape[1] == 0:
        raise ValueError(
            'No val holdout edges found in removed_edges.csv (split=val). '
            'FEI model selection requires val edges.'
        )
    print(
        f'FEI edge context: val_edges={edge_ctx["val_edge_index"].shape[1]}, '
        f'train_edges={edge_ctx["train_edges_df"].shape[0]}',
        flush=True,
    )

    print('# output nodes', len(data.node_names_dict['output']))
    print(f'Training on sample: {args.sample}')

    obs_dir = f'{args.data}/{args.sample}/obs/'
    dxdt_dir = f'{args.data}/{args.sample}/dxdt/'

    train_scale_dataset = DXDTDataset(
        train_meta,
        input_names=data.node_names_dict['input'],
        output_names=data.node_names_dict['output'],
        src_names=src_gene_names,
        obs_dir=obs_dir,
        dxdt_dir=dxdt_dir,
    )
    if len(train_scale_dataset) == 0:
        raise ValueError('train DXDTDataset is empty after filtering')
    dxdt_scale = train_scale_dataset._scale

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accessible_mask = get_or_compute_drug_accessible_mask(data)
    accessible_out_ix = accessible_indices(accessible_mask).to(device)
    n_acc = int(accessible_mask.sum())
    n_total = len(data.node_names_dict['output'])
    print(f'Drug-accessible output genes: {n_acc} / {n_total}')

    dataset_kwargs = dict(
        input_names=data.node_names_dict['input'],
        output_names=data.node_names_dict['output'],
        src_names=src_gene_names,
        obs_dir=obs_dir,
        dxdt_dir=dxdt_dir,
        scale=dxdt_scale,
    )

    fei_val_meta = _prepare_fei_val_meta(val_meta, args)
    x_val_fei, y_val_fei = _stack_fei_val_tensors(
        fei_val_meta,
        dataset_kwargs,
        batch_size=args.batch_size,
        max_val_rows=args.fei_max_val_rows,
    )
    print(
        f'FEI val tensor: X={tuple(x_val_fei.shape)}, y={tuple(y_val_fei.shape)}',
        flush=True,
    )

    train_dataset = DXDTDataset(meta=train_meta, **dataset_kwargs)
    val_dataset = DXDTDataset(meta=val_meta, **dataset_kwargs)
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

    gsnn_kwargs = _gsnn_kwargs_from_args(args, data, node_activity_dim=1)
    if args.BIOGSNN:
        model = BIOGSNN(
            gsnn_kwargs=gsnn_kwargs,
            gene_norm=gene_norm,
            dxdt_nonlin=args.dxdt_nonlin,
            init_rna_half_life=args.init_rna_half_life,
            dxdt_scale=(dxdt_scale if args.init_rna_half_life is not None else None),
        ).to(device)
    else:
        model = GSNN(**gsnn_kwargs).to(device)

    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=args.patience, threshold=1e-3,
    )
    crit = torch.nn.MSELoss(reduction='mean')

    model, best_W, history_df, summary = train_with_fei_validation(
        args,
        model,
        train_loader,
        val_loader,
        optim,
        scheduler,
        crit,
        device,
        accessible_out_ix,
        x_val_fei,
        y_val_fei,
        edge_ctx,
        data,
    )
    _save_fei_artifacts(args.out, args.sample, history_df, summary, best_W)

    torch.save(model, f'{args.out}/pretrained_model_{args.sample}.pt')
    torch.save(torch.tensor([dxdt_scale]), f'{args.out}/dxdt_scale_{args.sample}.pt')
    print(f'Saved pretrained model to {args.out}/pretrained_model_{args.sample}.pt')
