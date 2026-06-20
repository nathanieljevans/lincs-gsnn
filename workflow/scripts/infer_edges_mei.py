'''
Per-sample Magnitude Edge Inference (MEI) via activation/gradient magnitude
correlation (gsnn.optim.MagnitudeEdgeInferer).

Requires model.checkpoint=False at inference time; the script disables
checkpointing on the loaded model in-memory only (saved weights unchanged).
'''

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from gsnn.optim.MagnitudeEdgeInferer import MagnitudeEdgeInferer
from torch.utils.data import DataLoader

from lincs_gsnn.data.DXDTDataset import DXDTDataset
from lincs_gsnn.data.dxdt_meta import filter_min_dose, subsample as subsample_dxdt_meta


def get_args():
    parser = argparse.ArgumentParser(description='Magnitude Edge Inference (MEI) for one trajectory sample')

    parser.add_argument('--root_gsnn', type=str, required=True,
                        help='Run output root (<runs>/<run_id>)')
    parser.add_argument('--root_traj', type=str, required=True,
                        help='Trajectory preds root (lincs-traj output)')
    parser.add_argument('--sample_id', type=str, required=True,
                        help='Sample directory name (e.g. sample_0)')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing pretrained_model_{sample}.pt or trained_model_{sample}.pt')
    parser.add_argument('--model_name', type=str, default='pretrained_model',
                        help='Model filename prefix without sample suffix (pretrained_model or trained_model)')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for this sample')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--reduction', type=str, default='l1', choices=['l1', 'l2'])
    parser.add_argument('--score', type=str, default='partial', choices=['corr', 'partial'])
    parser.add_argument('--layer_agg', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--ridge', type=float, default=1e-8)
    parser.add_argument('--use_pre_norm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--subsample',
        type=float,
        default=1.0,
        help='Fraction of dxdt_meta rows to use (1.0 = all, 0.1 = 10%%)',
    )
    parser.add_argument(
        '--subsample_seed',
        type=int,
        default=0,
        help='Random seed for --subsample',
    )
    parser.add_argument(
        '--min_dose_um',
        type=float,
        default=None,
        help='Keep dxdt_meta rows with dose >= this value (µM); omit for no filter',
    )
    return parser.parse_args()


def load_data(args):
    bionet_dir = os.path.join(args.root_gsnn, 'bionetwork')
    data = torch.load(os.path.join(bionet_dir, 'bionetwork.pt'), weights_only=False)

    model_path = os.path.join(args.model_dir, f'{args.model_name}_{args.sample_id}.pt')
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # MagnitudeEdgeInferer requires checkpoint=False during backward.
    if getattr(model, 'checkpoint', False):
        model.checkpoint = False

    scale_path = os.path.join(args.model_dir.replace('train', 'pretrain'), f'dxdt_scale_{args.sample_id}.pt')
    if not os.path.isfile(scale_path):
        scale_path = os.path.join(args.model_dir, f'dxdt_scale_{args.sample_id}.pt')
    dxdt_scale = torch.load(scale_path, weights_only=False).item()

    preds_dir = os.path.join(args.root_traj, 'predict_grid')
    x_names = pd.read_csv(os.path.join(preds_dir, 'gene_names.csv'))['gene_names'].values.astype(str)
    dxdt_meta = pd.read_csv(os.path.join(preds_dir, 'dxdt_meta.csv'))

    valid_drugs = [x.split('DRUG__')[1] for x in data.node_names_dict['input'] if 'DRUG__' in x]
    dxdt_meta = dxdt_meta[dxdt_meta['pert_id'].isin(valid_drugs)]

    return data, model, dxdt_scale, x_names, dxdt_meta


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    print('-' * 80, flush=True)
    print(args, flush=True)
    print('-' * 80, flush=True)

    data, model, dxdt_scale, x_names, dxdt_meta = load_data(args)

    if args.min_dose_um is not None:
        n_before = len(dxdt_meta)
        dxdt_meta = filter_min_dose(dxdt_meta, args.min_dose_um)
        if args.verbose:
            print(
                f'MEI min_dose_um filter: >= {args.min_dose_um} '
                f'rows={len(dxdt_meta)}/{n_before}',
                flush=True,
            )

    n_meta_before = len(dxdt_meta)
    dxdt_meta = subsample_dxdt_meta(dxdt_meta, args.subsample, seed=args.subsample_seed)
    if args.verbose:
        print(
            f'MEI subsample: frac={args.subsample} seed={args.subsample_seed} '
            f'rows={len(dxdt_meta)}/{n_meta_before}',
            flush=True,
        )

    sample_obs_dir = os.path.join(args.root_traj, 'predict_grid', args.sample_id, 'obs')
    sample_dxdt_dir = os.path.join(args.root_traj, 'predict_grid', args.sample_id, 'dxdt')

    dataset = DXDTDataset(
        dxdt_meta,
        input_names=data.node_names_dict['input'],
        output_names=data.node_names_dict['output'],
        src_names=x_names,
        obs_dir=sample_obs_dir,
        dxdt_dir=sample_dxdt_dir,
        scale=dxdt_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    inferer = MagnitudeEdgeInferer(
        model,
        data,
        reduction=args.reduction,
        use_pre_norm=args.use_pre_norm,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batches = len(loader)
    print(
        f'MEI fit starting: sample={args.sample_id} device={device} '
        f'dataset={len(dataset)} batches={n_batches}',
        flush=True,
    )
    n_samples = inferer.fit(loader, device=device, verbose=args.verbose)
    print(f'MEI fit complete: n={n_samples}', flush=True)

    print('MEI evaluate starting...', flush=True)
    res = inferer.evaluate(
        layer_agg=args.layer_agg,
        score=args.score,
        ridge=args.ridge,
    )
    res = res.assign(sample_id=args.sample_id)

    out_csv = os.path.join(args.out, 'inferred_edges_test.csv')
    res.to_csv(out_csv, index=False)
    print(f'Wrote {len(res)} edge scores to {out_csv}', flush=True)


if __name__ == '__main__':
    main()
