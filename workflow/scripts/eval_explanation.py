'''
Aggregate explanation-quality and training-validation metrics for one
contrastive_explanation config entry.

Writes ``<out_dir>/<example>.pt`` (full Python dict) and ``<example>.json``
(JSON-safe view with config snapshot).
'''

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from lincs_gsnn.explain.eval import (
    agg_edge_scores,
    agg_node_scores,
    data2nx,
    edge_ranking_comparison,
    eval_node_activity,
    eval_traj_diff,
    node_ranking_comparison,
    path_ranking_comparison,
    primary_regulator_comparison,
)
from lincs_gsnn.proc.graph import map_function_node, remap_eval_spec


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate explanation outputs for one workflow example',
    )
    parser.add_argument('--root_gsnn', type=str, required=True,
                        help='Run output root (<runs>/<run_id>)')
    parser.add_argument('--root_traj', type=str, required=True,
                        help='Trajectory preds root (lincs-traj output)')
    parser.add_argument('--example', type=str, required=True,
                        help='contrastive_explanations config key')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Contrastive explanation output folder name')
    parser.add_argument('--explanation_target', type=str, required=True,
                        choices=('edge', 'node'))
    parser.add_argument('--target_gene', type=str, required=True)
    parser.add_argument('--pert_id', type=str, required=True)
    parser.add_argument('--cell_line_1', type=str, required=True)
    parser.add_argument('--cell_line_2', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory for eval artifacts (<run_id>/eval)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Absolute path to the workflow config YAML')
    parser.add_argument('--eval_spec_json', type=str, required=True,
                        help='JSON-serialised eval: block for this example')
    parser.add_argument('--pretrain_dir', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='',
                        help='Odeint train dir; empty when train.enabled=false')
    parser.add_argument('--use_train', action='store_true',
                        help='Include odeint train val metrics when set')
    parser.add_argument('--model_id', type=str, default='',
                        help='Model id for eval_node_activity (default: first model_*)')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def _non_contrastive_dir(contrastive_output_dir: str, cell_line: str) -> str:
    base = contrastive_output_dir.replace(
        'contrastive_explanation_', 'non_contrastive_explanation_', 1,
    )
    return f'{base}__{cell_line}'


def _load_yaml(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _build_config_snapshot(config_path: str, example: str) -> dict:
    cfg = _load_yaml(config_path)
    example_cfg = dict(cfg.get('contrastive_explanations', {}).get(example, {}))
    return {
        'run_id': cfg.get('run_id'),
        'config_path': os.path.abspath(config_path),
        'example': example,
        'example_config': example_cfg,
        'pretrain': cfg.get('pretrain', {}),
        'train': cfg.get('train', {}),
        'node_activity': cfg.get('node_activity', {}),
        'hypernetwork': cfg.get('hypernetwork', {}),
        'make_bio_network': cfg.get('make_bio_network', {}),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _df_from_obj(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, 'data'):
        return obj.data
    return None


def _discover_models(pretrain_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(pretrain_dir, 'val_metrics_pretrain_model_*.json')))
    return [os.path.basename(p).replace('val_metrics_pretrain_', '').replace('.json', '')
            for p in paths]


def _discover_samples(pretrain_dir: str) -> list[str]:
    """Backward-compatible alias for :func:`_discover_models`."""
    return _discover_models(pretrain_dir)


def _load_val_metrics(metrics_dir: str, prefix: str, model_id: str) -> dict | None:
    path = os.path.join(metrics_dir, f'{prefix}_{model_id}.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _aggregate_val_metrics(metrics_dir: str, prefix: str, models: list[str]) -> dict:
    per_model = {}
    for model_id in models:
        metrics = _load_val_metrics(metrics_dir, prefix, model_id)
        if metrics is not None:
            per_model[model_id] = metrics

    def _mean(key: str) -> float | None:
        vals = [m[key] for m in per_model.values() if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        'per_model': per_model,
        'per_sample': per_model,  # backward-compatible alias
        'mean_best_val_nll': _mean('best_val_nll'),
        'mean_best_val_mse': _mean('best_val_mse'),
        'mean_best_val_r2': _mean('best_val_r2'),
        'mean_final_val_nll': _mean('final_val_nll'),
        'mean_final_val_mse': _mean('final_val_mse'),
        'mean_final_val_r2': _mean('final_val_r2'),
    }


def _count_models_in_agg_csv(csv_path: str) -> int:
    header = pd.read_csv(csv_path, nrows=0)
    model_ids = set()
    for col in header.columns:
        for prefix in ('gsnn_score_', 'ig_score_', 'occlusion_score_'):
            if col.startswith(prefix):
                model_id = col[len(prefix):]
                if model_id.startswith('model_'):
                    model_ids.add(model_id)
    return len(model_ids)


def _count_samples_in_agg_csv(csv_path: str) -> int:
    """Backward-compatible alias for :func:`_count_models_in_agg_csv`."""
    return _count_models_in_agg_csv(csv_path)


def _score_distribution(df: pd.DataFrame) -> dict:
    score_cols = [c for c in ('mean_gsnn_score', 'mean_ig_score', 'mean_oc_score')
                  if c in df.columns]
    out = {}
    for col in score_cols:
        s = df[col].astype(float)
        out[col] = {
            'mean': float(s.mean()),
            'std': float(s.std(ddof=0)),
            'min': float(s.min()),
            'max': float(s.max()),
            'q05': float(s.quantile(0.05)),
            'q50': float(s.quantile(0.50)),
            'q95': float(s.quantile(0.95)),
        }
    return out


def _primary_regulator_mrr(primary_regulators: list[dict]) -> dict:
    if not primary_regulators:
        return {}
    out = {}
    for method in ('gsnn_rank', 'ig_rank', 'oc_rank'):
        ranks = [r[method] for r in primary_regulators if r.get(method)]
        out[method.replace('_rank', '')] = float(np.mean([1.0 / rk for rk in ranks])) if ranks else None
    return out


def _summary_mrr_edge(
    primary_regulators: list[dict],
    edge_mrr: pd.DataFrame | None,
    path_mrr: pd.DataFrame | None,
) -> dict:
    summary = {}
    pr_mrr = _primary_regulator_mrr(primary_regulators)
    if pr_mrr:
        summary['primary_regulator'] = pr_mrr
    if edge_mrr is not None and not edge_mrr.empty:
        summary['edge'] = edge_mrr.iloc[0].to_dict()
    if path_mrr is not None and not path_mrr.empty:
        summary['path'] = path_mrr.iloc[0].to_dict()
    return summary


def _summary_mrr_node(node_mrr: pd.DataFrame | None) -> dict:
    if node_mrr is None or node_mrr.empty:
        return {}
    return {'node': node_mrr.iloc[0].to_dict()}


def _eval_edge_source(
    agg_csv_path: str,
    G,
    eval_spec: dict,
    target_gene: str,
    pert_id: str,
    verbose: bool,
) -> dict:
    edge_df = agg_edge_scores(agg_csv_path, fill_value=np.nan)
    expected_direction = eval_spec.get('expected_direction', 'negative')
    max_path_length = int(eval_spec.get('max_path_length', 6))

    primary_regulators = []
    for row in eval_spec.get('primary_regulators', []) or []:
        reg = primary_regulator_comparison(
            edge_df=edge_df,
            target_node=row['target_node'],
            expected_regulator=row['regulator'],
            expected_direction=expected_direction,
            plot=False,
            save_dir=None,
        )
        primary_regulators.append(reg)

    expected_edges = pd.DataFrame(eval_spec.get('expected_edges', []) or [])
    edge_mrr, edge_ranks = edge_ranking_comparison(
        edge_df=edge_df,
        G=G,
        target_node=f'GENE__{target_gene}',
        source_node=f'DRUG__{pert_id}',
        expected_edges=expected_edges,
        expected_direction=expected_direction,
        verbose=verbose,
    )

    expected_paths = eval_spec.get('expected_paths', []) or []
    path_mrr, path_ranks = path_ranking_comparison(
        path_df=edge_df,
        G=G,
        target_node=f'GENE__{target_gene}',
        source_node=f'DRUG__{pert_id}',
        expected_paths=expected_paths,
        expected_direction=expected_direction,
        verbose=verbose,
        max_path_length=max_path_length,
    )

    return {
        'n_samples': _count_samples_in_agg_csv(agg_csv_path),
        'score_distribution': _score_distribution(edge_df),
        'primary_regulators': primary_regulators,
        'primary_regulator_mrr': _primary_regulator_mrr(primary_regulators),
        'edge_ranking_mrr': edge_mrr,
        'edge_ranking_ranks': _df_from_obj(edge_ranks),
        'path_ranking_mrr': path_mrr,
        'path_ranking_ranks': path_ranks,
        'summary_mrr': _summary_mrr_edge(primary_regulators, edge_mrr, path_mrr),
    }


def _eval_node_source(
    agg_csv_path: str,
    G,
    eval_spec: dict,
    target_gene: str,
    pert_id: str,
    verbose: bool,
    node_map: dict | None = None,
) -> dict:
    node_df = agg_node_scores(agg_csv_path, fill_value=np.nan)
    expected_direction = eval_spec.get('expected_direction', 'negative')
    expected_nodes = eval_spec.get('expected_nodes', []) or []

    node_mrr, node_ranks = node_ranking_comparison(
        node_df=node_df,
        G=G,
        target_node=map_function_node(f'RNA__{target_gene}', node_map),
        source_node=f'DRUG__{pert_id}',
        expected_nodes=expected_nodes,
        expected_direction=expected_direction,
        verbose=verbose,
    )

    return {
        'n_samples': _count_samples_in_agg_csv(agg_csv_path),
        'score_distribution': _score_distribution(node_df),
        'node_ranking_mrr': node_mrr,
        'node_ranking_ranks': _df_from_obj(node_ranks),
        'summary_mrr': _summary_mrr_node(node_mrr),
    }


def _eval_source(
    label: str,
    explanation_dir: str,
    explanation_target: str,
    G,
    eval_spec: dict,
    target_gene: str,
    pert_id: str,
    verbose: bool,
    node_map: dict | None = None,
) -> dict:
    agg_csv = os.path.join(explanation_dir, 'aggregated_cres.csv')
    if not os.path.isfile(agg_csv):
        raise FileNotFoundError(f'Missing aggregated cres for {label}: {agg_csv}')

    if explanation_target == 'edge':
        result = _eval_edge_source(
            agg_csv, G, eval_spec, target_gene, pert_id, verbose,
        )
    else:
        result = _eval_node_source(
            agg_csv, G, eval_spec, target_gene, pert_id, verbose, node_map,
        )

    result['aggregated_cres_path'] = agg_csv
    result['explanation_dir'] = explanation_dir
    return result


def main():
    args = get_args()
    eval_spec = json.loads(args.eval_spec_json)
    os.makedirs(args.out_dir, exist_ok=True)

    print('--' * 40)
    print(f'Evaluating example: {args.example}')
    print(f'Explanation target: {args.explanation_target}')
    print('--' * 40)

    config_snapshot = _build_config_snapshot(args.config_path, args.example)
    models = _discover_models(args.pretrain_dir)
    if not models:
        print(f'WARNING: no pretrain val metrics found under {args.pretrain_dir}')

    pretrain_metrics = _aggregate_val_metrics(
        args.pretrain_dir, 'val_metrics_pretrain', models,
    )
    train_metrics = None
    if args.use_train and args.train_dir:
        train_metrics = _aggregate_val_metrics(
            args.train_dir, 'val_metrics_train', models,
        )

    bionet_path = os.path.join(args.root_gsnn, 'bionetwork', 'bionetwork.pt')
    data = torch.load(bionet_path, weights_only=False)
    node_map = getattr(data, 'function_node_map', None)
    if node_map:
        eval_spec = remap_eval_spec(eval_spec, node_map)
    G = data2nx(data)

    contrastive_dir = os.path.join(args.root_gsnn, args.output_dir)
    nc_dir_1 = os.path.join(
        args.root_gsnn,
        _non_contrastive_dir(args.output_dir, args.cell_line_1),
    )
    nc_dir_2 = os.path.join(
        args.root_gsnn,
        _non_contrastive_dir(args.output_dir, args.cell_line_2),
    )

    explanation_eval = {
        'contrastive': _eval_source(
            'contrastive', contrastive_dir, args.explanation_target,
            G, eval_spec, args.target_gene, args.pert_id, args.verbose, node_map,
        ),
        f'non_contrastive_{args.cell_line_1}': _eval_source(
            f'non_contrastive_{args.cell_line_1}', nc_dir_1,
            args.explanation_target, G, eval_spec,
            args.target_gene, args.pert_id, args.verbose, node_map,
        ),
        f'non_contrastive_{args.cell_line_2}': _eval_source(
            f'non_contrastive_{args.cell_line_2}', nc_dir_2,
            args.explanation_target, G, eval_spec,
            args.target_gene, args.pert_id, args.verbose, node_map,
        ),
    }

    eval_traj_diff_result = None
    if args.output_dir.startswith('contrastive_explanation_'):
        auc1, auc2, auc_diff, pvalue = eval_traj_diff(contrastive_dir)
        eval_traj_diff_result = {
            'auc1': float(np.mean(auc1)),
            'auc2': float(np.mean(auc2)),
            'auc_diff': float(np.mean(auc_diff)),
            'pvalue': float(pvalue),
            'cell_line_1': args.cell_line_1,
            'cell_line_2': args.cell_line_2,
        }

    model_id = args.model_id or (models[0] if models else 'model_0')
    na_model_path = None
    if args.use_train and args.train_dir:
        na_model_path = os.path.join(args.train_dir, f'trained_model_{model_id}.pt')
    na_mean_var, na_corrs = eval_node_activity(
        root_gsnn=args.root_gsnn,
        root_traj=args.root_traj,
        model_id=model_id,
        plot=False,
        save_dir=None,
        model_path=na_model_path,
    )
    node_activity = None
    if na_mean_var is not None:
        node_activity = {
            'model_id': model_id,
            'sample_id': model_id,  # backward-compatible alias
            'mean_score_var': float(na_mean_var),
            'corrs': na_corrs,
        }

    result = {
        'example': args.example,
        'config': config_snapshot,
        'pretrain_metrics': pretrain_metrics,
        'train_metrics': train_metrics,
        'eval_traj_diff': eval_traj_diff_result,
        'explanation_eval': explanation_eval,
        'node_activity': node_activity,
    }

    pt_path = os.path.join(args.out_dir, f'{args.example}.pt')
    json_path = os.path.join(args.out_dir, f'{args.example}.json')

    torch.save(result, pt_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(result), f, indent=2)

    print(f'Saved eval results to {pt_path} and {json_path}')


if __name__ == '__main__':
    main()
