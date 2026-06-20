'''
Convert a dense FEI score matrix (fei_W_{sample}.pt) to long-format edge CSV.

Output schema matches MEI inferred_edges_test.csv so downstream aggregate/eval
scripts are unchanged.
'''

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Convert dense FEI W matrix to inferred_edges_test.csv',
    )
    parser.add_argument('--root_gsnn', type=str, required=True,
                        help='Run output root (<runs>/<run_id>)')
    parser.add_argument('--sample_id', type=str, required=True,
                        help='Sample directory name (e.g. sample_0)')
    parser.add_argument('--fei_w', type=str, required=True,
                        help='Path to fei_W_{sample}.pt')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for this sample')
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    print('-' * 80, flush=True)
    print(args, flush=True)
    print('-' * 80, flush=True)

    bionet_dir = os.path.join(args.root_gsnn, 'bionetwork')
    data = torch.load(os.path.join(bionet_dir, 'bionetwork.pt'), weights_only=False)

    payload = torch.load(args.fei_w, weights_only=False)
    if isinstance(payload, dict) and 'W' in payload:
        W = payload['W']
    else:
        W = payload
    W = np.asarray(W, dtype=np.float64)

    node_names = np.array(data.node_names_dict['function'])
    n = len(node_names)
    if W.shape != (n, n):
        raise ValueError(f'Expected W shape ({n}, {n}), got {W.shape}')

    src_idx, dst_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    mask = src_idx != dst_idx
    src_names = node_names[src_idx[mask]]
    dst_names = node_names[dst_idx[mask]]
    scores = W[mask]

    df = pd.DataFrame({
        'src_func': src_names,
        'dst_func': dst_names,
        'corr': scores,
        'sample_id': args.sample_id,
    })

    prot_rna_mask = (
        df['src_func'].str.startswith('PROTEIN__')
        & df['dst_func'].str.startswith(('PROTEIN__', 'RNA__'))
    )
    df = df.loc[prot_rna_mask].reset_index(drop=True)

    out_csv = os.path.join(args.out, 'inferred_edges_test.csv')
    df.to_csv(out_csv, index=False)
    print(f'Wrote {len(df)} edge scores to {out_csv}', flush=True)


if __name__ == '__main__':
    main()
