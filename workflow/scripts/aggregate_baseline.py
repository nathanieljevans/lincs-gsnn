'''
Generic per-sample baseline aggregator: mean score per (source, target).
'''

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate per-sample baseline predictions')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing sample_*/predictions.csv')
    parser.add_argument('--out', type=str, required=True,
                        help='Output aggregated predictions.csv path')
    parser.add_argument('--score_col', type=str, default='score')
    return parser.parse_args()


def main():
    args = get_args()
    pattern = os.path.join(args.input_dir, '*/predictions.csv')
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f'No predictions.csv files found under {args.input_dir}')

    dfs = []
    for csv_file in csv_files:
        sample = os.path.basename(os.path.dirname(csv_file))
        df = pd.read_csv(csv_file, low_memory=False)
        df['sample'] = sample
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    score_col = args.score_col
    if score_col not in combined.columns:
        raise KeyError(f'Score column {score_col!r} not in predictions: {combined.columns.tolist()}')

    agg = (
        combined.groupby(['source', 'target'], as_index=False)[score_col]
        .mean()
        .rename(columns={score_col: 'score'})
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f'Wrote {len(agg)} aggregated predictions to {args.out}')


if __name__ == '__main__':
    main()
