'''
Expression-correlation baseline for one trajectory sample.
'''

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Correlation baseline for one sample')
    parser.add_argument('--sample_id', type=str, required=True)
    parser.add_argument('--traj_dir', type=str, required=True)
    parser.add_argument('--bionet', type=str, required=True)
    parser.add_argument('--correlation', type=str, default='spearman', choices=['pearson', 'spearman'])
    parser.add_argument('--dose', type=float, default=10.0)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--exclude_pert_ids', type=str, default='BRD-K54997624')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    bionet_path = os.path.join(args.bionet, 'bionetwork.pt')
    data = torch.load(bionet_path, weights_only=False)
    preds_dir = os.path.join(args.traj_dir, 'predict_grid')
    gene_names = pd.read_csv(os.path.join(preds_dir, 'gene_names.csv'))['gene_names'].astype(str).tolist()

    obs_meta = pd.read_csv(os.path.join(preds_dir, 'pred_meta.csv'))
    if args.exclude_pert_ids:
        exclude = [p.strip() for p in args.exclude_pert_ids.split(',') if p.strip()]
        obs_meta = obs_meta[~obs_meta['pert_id'].isin(exclude)]
    obs_meta = obs_meta[obs_meta['dose'] == args.dose]

    candidates = pd.DataFrame({
        'source': ['PROTEIN__' + g1 for g1 in gene_names for g2 in gene_names],
        'target': ['RNA__' + g2 for g1 in gene_names for g2 in gene_names],
    })
    candidates = candidates.assign(
        source_gene=lambda x: x.source.str.split('__', expand=True)[1],
        target_gene=lambda x: x.target.str.split('__', expand=True)[1],
    )
    candidates = candidates[candidates.source_gene != candidates.target_gene]

    gene2idx = {g: i for i, g in enumerate(gene_names)}
    src_idx = candidates.source_gene.map(gene2idx).astype(int).values
    tgt_idx = candidates.target_gene.map(gene2idx).astype(int).values

    sample_dir = os.path.join(preds_dir, args.sample_id, 'obs')
    xs = []
    for _, row in obs_meta.iterrows():
        xs.append(torch.load(os.path.join(sample_dir, row.file_name), weights_only=False))
    xs = torch.stack(xs, dim=0)
    xs = xs - xs[:, [0], :]
    xs = xs.reshape(-1, xs.shape[2]).numpy()

    r_mat = pd.DataFrame(xs).corr(method=args.correlation).values
    scores = r_mat[src_idx, tgt_idx]

    out_df = candidates[['source', 'target']].copy()
    out_df['score'] = scores
    out_path = os.path.join(args.out, 'predictions.csv')
    out_df.to_csv(out_path, index=False)
    print(f'Wrote correlation predictions for {args.sample_id} to {out_path}')


if __name__ == '__main__':
    main()
