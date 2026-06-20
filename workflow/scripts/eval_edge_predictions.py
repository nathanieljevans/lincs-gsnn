'''
Evaluate edge prediction scores against held-out val/test partitions.

Shared by MEI and baseline methods. Reads ``split`` from ``removed_edges.csv``
when present; otherwise treats all removed edges as test positives (legacy).

When ``--full_partition`` is set, additional positive edges are loaded from
OmniPath resources (Transcriptional, CollecTRI, PathwayExtra, etc.) configured
via CLI flags. Train/val/test edges are excluded from the full partition.
'''

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lincs_gsnn.utils.eval import eval_edge_inference_any


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate edge prediction scores')
    parser.add_argument('--predictions_csv', type=str, required=True)
    parser.add_argument('--bionet', type=str, required=True,
                        help='Directory containing bionetwork.pt')
    parser.add_argument('--removed_edges', type=str, required=True,
                        help='Path to removed_edges.csv')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--method_name', type=str, required=True)
    parser.add_argument('--score_col', type=str, default='score')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--full_partition', action='store_true', default=False,
                        help='Enable full-partition evaluation from OmniPath resources')
    parser.add_argument('--full_include_transcriptional', action='store_true', default=False)
    parser.add_argument('--full_include_collecTRI', action='store_true', default=False)
    parser.add_argument('--full_include_dorothea', action='store_true', default=False)
    parser.add_argument('--full_dorothea_levels', type=str, default='ABCD')
    parser.add_argument('--full_include_omnipath', action='store_true', default=False)
    parser.add_argument('--full_include_pathway_extra', action='store_true', default=False)
    parser.add_argument('--full_include_kinase_extra', action='store_true', default=False)
    parser.add_argument('--full_include_ligrec_extra', action='store_true', default=False)
    parser.add_argument('--full_include_tf_mirna', action='store_true', default=False)
    return parser.parse_args()


def _fmt_metrics(metrics: dict) -> str:
    if not metrics:
        return 'n=0'
    parts = [f"n={metrics.get('n')}"]
    for key in ('auroc', 'aupr', 'mrr', 'top1_acc', 'top10_acc', 'top100_acc'):
        val = metrics.get(key)
        if val is not None:
            parts.append(f'{key}={val:.4f}' if isinstance(val, float) else f'{key}={val}')
    return '  '.join(parts)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def _load_train_edges(data):
    node_names = np.array(data.node_names_dict['function'])
    src = data.edge_index_dict['function', 'to', 'function'][0]
    dst = data.edge_index_dict['function', 'to', 'function'][1]
    return pd.DataFrame({
        'source': node_names[src.cpu().numpy()],
        'target': node_names[dst.cpu().numpy()],
        'train_edge': True,
    })


def _load_holdout_edges(path):
    holdout = pd.read_csv(path, low_memory=False)
    holdout = holdout.rename(columns={'src_name': 'source', 'dst_name': 'target'})
    if 'split' in holdout.columns:
        holdout = holdout.assign(
            val_edge=holdout['split'].eq('val'),
            test_edge=holdout['split'].eq('test'),
        )
    else:
        holdout = holdout.assign(val_edge=False, test_edge=True)
    return holdout[['source', 'target', 'val_edge', 'test_edge']]


def _edges_from_op(df, src_prefix, tgt_prefix):
    '''Standardize an OmniPath interaction table to (source, target) func names.'''
    out = df[['source_genesymbol', 'target_genesymbol']].copy()
    out = out.assign(
        source=lambda x: src_prefix + x.source_genesymbol.astype(str),
        target=lambda x: tgt_prefix + x.target_genesymbol.astype(str),
    )
    return out[['source', 'target']]


def load_full_partition_edges(args):
    '''Fetch configured OmniPath resources and return deduplicated (source, target) edges.'''
    import omnipath as op  # lazy: only required when --full_partition is set

    parts = []
    if args.full_include_transcriptional:
        _log(args.verbose, '  full partition: loading Transcriptional...')
        df = op.interactions.Transcriptional().get(genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'RNA__'))

    if args.full_include_collecTRI:
        _log(args.verbose, '  full partition: loading CollecTRI...')
        df = op.interactions.CollecTRI().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'RNA__'))

    if args.full_include_dorothea:
        _log(args.verbose, '  full partition: loading DoRothEA...')
        levels = list(str(args.full_dorothea_levels))
        df = op.interactions.Dorothea().get(
            organism='human', dorothea_levels=levels, genesymbol=True,
        )
        parts.append(_edges_from_op(df, 'PROTEIN__', 'RNA__'))

    if args.full_include_omnipath:
        _log(args.verbose, '  full partition: loading OmniPath...')
        df = op.interactions.OmniPath().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'PROTEIN__'))

    if args.full_include_pathway_extra:
        _log(args.verbose, '  full partition: loading PathwayExtra...')
        df = op.interactions.PathwayExtra().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'PROTEIN__'))

    if args.full_include_kinase_extra:
        _log(args.verbose, '  full partition: loading KinaseExtra...')
        df = op.interactions.KinaseExtra().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'PROTEIN__'))

    if args.full_include_ligrec_extra:
        _log(args.verbose, '  full partition: loading LigRecExtra...')
        df = op.interactions.LigRecExtra().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(df, 'PROTEIN__', 'PROTEIN__'))

    if args.full_include_tf_mirna:
        _log(args.verbose, '  full partition: loading TF-miRNA and miRNA...')
        tf_mirna = op.interactions.TFmiRNA().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(tf_mirna, 'PROTEIN__', 'RNA__'))
        mirna = op.interactions.miRNA().get(organism='human', genesymbol=True)
        parts.append(_edges_from_op(mirna, 'RNA__', 'RNA__'))

    if not parts:
        return pd.DataFrame(columns=['source', 'target'])

    full = pd.concat(parts, axis=0, ignore_index=True)
    full = full.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    return full.assign(full_edge=True)


def build_eval_table(predictions, data, removed_edges_path, score_col='score'):
    preds = predictions.rename(columns={
        'src_func': 'source',
        'dst_func': 'target',
    }).copy()
    if score_col not in preds.columns and 'corr' in preds.columns:
        preds = preds.rename(columns={'corr': score_col})

    train_edges = _load_train_edges(data)
    holdout = _load_holdout_edges(removed_edges_path)

    res = preds[['source', 'target', score_col]].copy()
    res = res.merge(train_edges, on=['source', 'target'], how='left')
    res = res.merge(holdout, on=['source', 'target'], how='left')
    for col in ('train_edge', 'val_edge', 'test_edge'):
        res[col] = res[col].eq(True)
    return res


def _apply_full_partition(res, args):
    '''Merge full-partition truth edges and exclude train/val/test from full positives.'''
    full_edges = load_full_partition_edges(args)
    _log(args.verbose, f'  full partition: {len(full_edges)} unique truth edges loaded')
    res = res.merge(full_edges, on=['source', 'target'], how='left')
    res['full_edge'] = res['full_edge'].fillna(False).astype(bool)
    res['full_edge'] = res['full_edge'] & ~(res.train_edge | res.val_edge | res.test_edge)
    res['negative_edge'] = ~(res.train_edge | res.val_edge | res.test_edge | res.full_edge)
    return res


def _plot_score_distribution(res, metric, out_path, method_name):
    scores = res[metric].astype(float)
    pos_val = res.loc[res['val_edge'], metric].astype(float)
    pos_test = res.loc[res['test_edge'], metric].astype(float)
    pos_full = (
        res.loc[res['full_edge'], metric].astype(float)
        if 'full_edge' in res.columns else pd.Series(dtype=float)
    )
    neg = res.loc[res['negative_edge'], metric].astype(float) if 'negative_edge' in res.columns else scores

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return
    bins = np.linspace(finite.min(), finite.max(), 50)

    plt.figure(figsize=(8, 4))
    if len(pos_val):
        plt.hist(pos_val, bins=bins, alpha=0.5, label='val', density=True, color='orange')
    if len(pos_test):
        plt.hist(pos_test, bins=bins, alpha=0.5, label='test', density=True, color='red')
    if len(pos_full):
        plt.hist(pos_full, bins=bins, alpha=0.5, label='full', density=True, color='purple')
    if len(neg):
        plt.hist(neg, bins=bins, alpha=0.5, label='neg', density=True, color='blue')
    plt.xlabel(metric)
    plt.ylabel('density')
    plt.title(f'{method_name} score distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print('-' * 80, flush=True)
    print(args, flush=True)
    print('-' * 80, flush=True)

    _log(args.verbose, f'Loading bionetwork from {args.bionet}')
    data = torch.load(os.path.join(args.bionet, 'bionetwork.pt'), weights_only=False)
    n_func = len(data.node_names_dict['function'])
    _log(args.verbose, f'  function nodes: {n_func}')

    _log(args.verbose, f'Loading predictions from {args.predictions_csv}')
    preds = pd.read_csv(args.predictions_csv, low_memory=False)
    _log(args.verbose, f'  predictions: {len(preds)} rows')

    _log(args.verbose, f'Loading holdout edges from {args.removed_edges}')
    holdout = _load_holdout_edges(args.removed_edges)
    _log(
        args.verbose,
        f'  holdout: {len(holdout)} edges '
        f'(val={int(holdout["val_edge"].sum())}, test={int(holdout["test_edge"].sum())})',
    )

    _log(args.verbose, 'Building evaluation table...')
    res = build_eval_table(preds, data, args.removed_edges, score_col=args.score_col)

    if args.full_partition:
        _log(args.verbose, 'Applying full-partition truth edges...')
        res = _apply_full_partition(res, args)
    else:
        res = res.assign(
            negative_edge=lambda x: ~(x.train_edge | x.val_edge | x.test_edge),
        )

    n_finite = int(res[args.score_col].notna().sum())
    partition_msg = (
        '  partitions: '
        f'train={int(res["train_edge"].sum())} '
        f'val={int(res["val_edge"].sum())} '
        f'test={int(res["test_edge"].sum())} '
    )
    if args.full_partition:
        partition_msg += f'full={int(res["full_edge"].sum())} '
    partition_msg += (
        f'neg={int(res["negative_edge"].sum())} '
        f'finite_scores={n_finite}/{len(res)}'
    )
    _log(args.verbose, partition_msg)

    _log(args.verbose, f'Evaluating {args.method_name} (score_col={args.score_col})...')
    results = eval_edge_inference_any(res.dropna(subset=[args.score_col]), data, args.score_col)

    merged_path = os.path.join(args.out_dir, f'{args.method_name}_merged_edges.csv')
    res.to_csv(merged_path, index=False)
    _log(args.verbose, f'Wrote merged edges to {merged_path}')

    json_path = os.path.join(args.out_dir, f'{args.method_name}_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    plot_path = os.path.join(args.out_dir, f'{args.method_name}_score_distribution.png')
    _plot_score_distribution(res, args.score_col, plot_path, args.method_name)
    _log(args.verbose, f'Wrote score distribution plot to {plot_path}')

    print(f'Wrote evaluation results to {json_path}', flush=True)
    print(f'Overall metrics ({args.method_name}):', flush=True)
    splits = ('train', 'val', 'test', 'full', 'neg') if args.full_partition else ('train', 'val', 'test', 'neg')
    for split in splits:
        print(f'  {split}: {_fmt_metrics(results["overall"].get(split, {}))}', flush=True)

    if args.verbose and results.get('by_edge_type'):
        print('By edge type (test split):', flush=True)
        for edge_type, metrics in sorted(results['by_edge_type'].items()):
            test_m = metrics.get('test', {})
            if test_m.get('n', 0):
                print(f'  {edge_type}: {_fmt_metrics(test_m)}', flush=True)


if __name__ == '__main__':
    main()
