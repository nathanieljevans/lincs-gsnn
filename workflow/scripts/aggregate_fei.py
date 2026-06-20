'''
Aggregate per-sample FEI inference CSVs into a single consensus table.
'''

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate per-sample FEI edge scores')
    parser.add_argument('--infer_edges_dir', type=str, required=True,
                        help='Directory containing sample_*/inferred_edges_test.csv')
    parser.add_argument('--out', type=str, required=True,
                        help='Output CSV path (e.g. infer_edges/aggregated_predictions.csv)')
    parser.add_argument('--score_col', type=str, default='corr',
                        help='Primary score column to aggregate')
    return parser.parse_args()


def main():
    args = get_args()
    pattern = os.path.join(args.infer_edges_dir, '*/inferred_edges_test.csv')
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f'No inferred_edges_test.csv files found under {args.infer_edges_dir}')

    print(f'Found {len(csv_files)} sample files')
    dfs = []
    for csv_file in csv_files:
        sample = os.path.basename(os.path.dirname(csv_file))
        df = pd.read_csv(csv_file, low_memory=False)
        df['sample'] = sample
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    group_cols = ['src_func', 'dst_func']
    numeric_cols = [c for c in ['corr', 'p_value', 'q_value'] if c in combined.columns]
    agg = combined.groupby(group_cols, as_index=False)[numeric_cols].mean()

    agg = agg.rename(columns={
        'src_func': 'source',
        'dst_func': 'target',
        'corr': 'score',
    })
    if 'score' not in agg.columns and args.score_col in agg.columns:
        agg = agg.rename(columns={args.score_col: 'score'})

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f'Wrote aggregated predictions ({len(agg)} edges) to {args.out}')


if __name__ == '__main__':
    main()
