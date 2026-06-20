'''
Merge per-method evaluation_results.json files into a side-by-side CSV.
'''

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate method evaluation JSON files')
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()


def _flatten_method(method, payload):
    rows = []
    overall = payload.get('overall', {})
    for split in ('train', 'val', 'test', 'full', 'neg'):
        m = overall.get(split, {})
        rows.append({
            'method': method,
            'partition': split,
            'scope': 'overall',
            **{k: m.get(k) for k in ('n', 'auroc', 'aupr', 'mrr', 'top1_acc', 'top10_acc', 'top100_acc')},
        })
    for edge_type, parts in payload.get('by_edge_type', {}).items():
        for split in ('train', 'val', 'test', 'full', 'neg'):
            m = parts.get(split, {})
            rows.append({
                'method': method,
                'partition': split,
                'scope': edge_type,
                **{k: m.get(k) for k in ('n', 'auroc', 'aupr', 'mrr', 'top1_acc', 'top10_acc', 'top100_acc')},
            })
    return rows


def main():
    args = get_args()
    pattern = os.path.join(args.eval_dir, '*_evaluation_results.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No evaluation JSON files in {args.eval_dir}')

    rows = []
    for path in files:
        method = os.path.basename(path).replace('_evaluation_results.json', '')
        with open(path) as f:
            payload = json.load(f)
        rows.extend(_flatten_method(method, payload))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote comparison table ({len(df)} rows) to {args.out}')


if __name__ == '__main__':
    main()
