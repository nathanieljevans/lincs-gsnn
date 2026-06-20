'''
ARACNe-AP baseline for one trajectory sample (wrapper around lincs_gsnn.utils.aracne).
'''

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from lincs_gsnn.utils.aracne import build_candidate_edges, run_sample_pipeline


def get_args():
    parser = argparse.ArgumentParser(description='ARACNe baseline for one sample')
    parser.add_argument('--sample_id', type=str, required=True,
                        help='Sample directory name (e.g. sample_0)')
    parser.add_argument('--traj_dir', type=str, required=True,
                        help='lincs-traj output root (contains predict_grid/)')
    parser.add_argument('--bionet', type=str, required=True)
    parser.add_argument('--aracne_root', type=str, required=True)
    parser.add_argument('--aracne_jar', type=str, required=True)
    parser.add_argument('--time_stride', type=int, default=20)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--exclude_pert_ids', type=str, default='BRD-K54997624',
                        help='Comma-separated pert_ids to exclude from obs meta')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    sample_i = int(args.sample_id.split('_')[-1])
    bionet = Path(args.bionet)
    traj_dir = Path(args.traj_dir)
    preds_dir = traj_dir / 'predict_grid'

    data = __import__('torch').load(bionet / 'bionetwork.pt', weights_only=False)
    gene_names = pd.read_csv(preds_dir / 'gene_names.csv')['gene_names'].astype(str).tolist()
    obs_meta = pd.read_csv(preds_dir / 'pred_meta.csv')
    if args.exclude_pert_ids:
        exclude = [p.strip() for p in args.exclude_pert_ids.split(',') if p.strip()]
        obs_meta = obs_meta[~obs_meta['pert_id'].isin(exclude)]

    res = build_candidate_edges(data, gene_names, bionet)
    regulators = pd.Series(sorted(res.source_gene.dropna().unique()))

    scores = run_sample_pipeline(
        sample_i=sample_i,
        traj_dir=traj_dir,
        obs_meta=obs_meta,
        gene_names=gene_names,
        regulators=regulators,
        aracne_root=args.aracne_root,
        aracne_jar=args.aracne_jar,
        res=res,
        stride=args.time_stride,
    )

    out_df = res[['source', 'target']].copy()
    out_df['score'] = scores.values
    out_path = os.path.join(args.out, 'predictions.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Wrote ARACNe predictions for {args.sample_id} to {out_path}')


if __name__ == '__main__':
    main()
