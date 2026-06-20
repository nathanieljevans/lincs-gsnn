'''
Merge per-example eval artifacts into a single run-level report.

Writes ``eval_results.pt``, ``eval_results.json``, a one-row wide
``eval_results.csv``, and a ``config.yaml`` snapshot under ``--eval_dir``.
'''

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser(
        description='Aggregate per-example eval results into one run-level report',
    )
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='Directory containing per-example .pt files')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Workflow config YAML used for this run')
    parser.add_argument('--examples', type=str, required=True,
                        help='Comma-separated contrastive_explanations keys')
    parser.add_argument('--out_name', type=str, default='eval_results',
                        help='Basename for output files (default: eval_results)')
    return parser.parse_args()


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


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return True
        return True
    if isinstance(value, (np.floating, np.integer)):
        return True
    return False


def _scalar_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _sanitize_key(value: Any) -> str:
    text = str(value)
    for old, new in ((' -> ', '_to_'), ('/', '_'), (' ', '_')):
        text = text.replace(old, new)
    return text


def _record_id(record: dict, index: int) -> str:
    for key in ('expected_regulator', 'path_short', 'node', 'target'):
        if record.get(key) is not None:
            return _sanitize_key(record[key])
    source = record.get('source')
    if source is not None:
        return _sanitize_key(source)
    return str(index)


def _is_scalar_dict(value: dict) -> bool:
    return all(_is_scalar(v) for v in value.values())


def _identity_keys_for_record(record: dict) -> frozenset[str]:
    for key in ('expected_regulator', 'path_short', 'node', 'target'):
        if record.get(key) is not None:
            return frozenset({key})
    if record.get('source') is not None:
        return frozenset({'source'})
    return frozenset()


def _flatten_list_of_dicts(
    items: list | tuple,
    prefix: str,
    out: dict[str, Any],
    skip_keys: frozenset[str],
) -> None:
    dict_items = [item for item in items if isinstance(item, dict)]
    if not dict_items:
        return

    if len(dict_items) == 1 and _is_scalar_dict(dict_items[0]):
        out.update(_flatten_scalars(dict_items[0], prefix, skip_keys))
        return

    for index, item in enumerate(dict_items):
        item_prefix = f'{prefix}__{_record_id(item, index)}'
        item_skip = skip_keys | _identity_keys_for_record(item)
        out.update(_flatten_scalars(item, item_prefix, item_skip))


def _flatten_scalars(
    obj: Any,
    prefix: str = '',
    skip_keys: frozenset[str] = frozenset({'per_sample'}),
) -> dict[str, Any]:
    '''Recursively flatten nested dicts into scalar columns for a wide CSV row.'''
    out: dict[str, Any] = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in skip_keys:
                continue
            key_str = str(key)
            new_prefix = f'{prefix}__{key_str}' if prefix else key_str
            if isinstance(value, dict):
                out.update(_flatten_scalars(value, new_prefix, skip_keys))
            elif _is_scalar(value):
                out[new_prefix] = _scalar_value(value)
            elif isinstance(value, (list, tuple)):
                if value and all(isinstance(item, dict) for item in value):
                    _flatten_list_of_dicts(value, new_prefix, out, skip_keys)
                elif value and all(_is_scalar(item) for item in value):
                    out[new_prefix] = json.dumps(_json_safe(value))
                else:
                    out[new_prefix] = json.dumps(_json_safe(value))
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                continue
            else:
                continue
        return out

    if _is_scalar(obj):
        if prefix:
            out[prefix] = _scalar_value(obj)
        return out

    if isinstance(obj, (list, tuple)) and prefix:
        out[prefix] = json.dumps(_json_safe(obj))
    return out


def _flatten_aggregate_for_csv(aggregate: dict) -> dict[str, Any]:
    '''Build one wide CSV row with plan column naming for per-example metrics.'''
    flat: dict[str, Any] = {}

    if aggregate.get('run_id') is not None:
        flat['run_id'] = aggregate['run_id']

    for key, prefix in (
        ('config', 'config'),
        ('pretrain_metrics', 'pretrain'),
        ('train_metrics', 'train'),
        ('node_activity', 'node_activity'),
    ):
        if key in aggregate and aggregate[key] is not None:
            flat.update(_flatten_scalars(aggregate[key], prefix=prefix))

    for example, example_data in (aggregate.get('examples') or {}).items():
        example_prefix = str(example)
        if example_data.get('eval_traj_diff') is not None:
            flat.update(
                _flatten_scalars(example_data['eval_traj_diff'], prefix=f'{example_prefix}__eval_traj_diff')
            )
        if example_data.get('explanation_target') is not None:
            flat[f'{example_prefix}__explanation_target'] = example_data['explanation_target']
        for source, source_data in (example_data.get('explanation_eval') or {}).items():
            flat.update(_flatten_scalars(source_data, prefix=f'{example_prefix}__{source}'))

    return flat


def _load_example_block(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing per-example eval artifact: {path}')
    return torch.load(path, weights_only=False)


def _build_aggregate(config_path: str, eval_dir: str, examples: list[str]) -> dict:
    if not examples:
        raise ValueError('No examples provided for aggregate_eval')

    with open(config_path, encoding='utf-8') as f:
        full_config = yaml.safe_load(f)

    per_example = {}
    for example in examples:
        pt_path = os.path.join(eval_dir, f'{example}.pt')
        per_example[example] = _load_example_block(pt_path)

    first = per_example[examples[0]]
    run_id = full_config.get('run_id') or first.get('config', {}).get('run_id')

    examples_out = {}
    for example, result in per_example.items():
        example_cfg = (result.get('config') or {}).get('example_config') or {}
        examples_out[example] = {
            'example_config': example_cfg,
            'explanation_target': example_cfg.get('explanation_target'),
            'eval_traj_diff': result.get('eval_traj_diff'),
            'explanation_eval': _json_safe(result.get('explanation_eval')),
        }

    return {
        'run_id': run_id,
        'config': full_config,
        'pretrain_metrics': first.get('pretrain_metrics'),
        'train_metrics': first.get('train_metrics'),
        'node_activity': _json_safe(first.get('node_activity')),
        'examples': examples_out,
    }


def main():
    args = get_args()
    examples = [x.strip() for x in args.examples.split(',') if x.strip()]
    os.makedirs(args.eval_dir, exist_ok=True)

    print('--' * 40)
    print(f'Aggregating eval results for {len(examples)} examples')
    print(f'Eval dir: {args.eval_dir}')
    print('--' * 40)

    aggregate = _build_aggregate(args.config_path, args.eval_dir, examples)

    pt_path = os.path.join(args.eval_dir, f'{args.out_name}.pt')
    json_path = os.path.join(args.eval_dir, f'{args.out_name}.json')
    csv_path = os.path.join(args.eval_dir, f'{args.out_name}.csv')
    config_out = os.path.join(args.eval_dir, 'config.yaml')

    torch.save(aggregate, pt_path)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(aggregate), f, indent=2)

    flat = _flatten_aggregate_for_csv(_json_safe(aggregate))
    pd.DataFrame([flat]).to_csv(csv_path, index=False)

    shutil.copy2(args.config_path, config_out)

    print(f'Saved {pt_path}')
    print(f'Saved {json_path}')
    print(f'Saved {csv_path} ({len(flat)} columns, 1 row)')
    print(f'Saved {config_out}')


if __name__ == '__main__':
    main()
