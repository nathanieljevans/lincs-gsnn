'''
Aggregate contrastive explanation results from multiple samples.

This script:
1. Merges cres DataFrames from each sample (outer merge with sample suffix)
2. Aggregates out_dict from each sample into a single dictionary

'''

import argparse
import pandas as pd
import torch
import os
import glob


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate contrastive explanation results from multiple samples')

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing sample subdirectories with results')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for aggregated results')
    parser.add_argument('--name', type=str, default='aggregated',
                       help='Name prefix for output files (default: aggregated)')

    args = parser.parse_args()
    return args


def find_sample_dirs(input_dir):
    """Find all sample directories in the input directory."""
    sample_dirs = sorted(glob.glob(os.path.join(input_dir, "sample_*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]
    return sample_dirs


def load_cres(sample_dir, sample_id):
    """Load contrastive results CSV for a sample."""
    csv_path = os.path.join(sample_dir, f'contrastive_results_{sample_id}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return None


def load_out_dict(sample_dir, sample_id):
    """Load out_dict (contrastive_results .pt file) for a sample."""
    pt_path = os.path.join(sample_dir, f'contrastive_results_{sample_id}.pt')
    if os.path.exists(pt_path):
        return torch.load(pt_path, weights_only=False)
    return None


def merge_cres_dataframes(cres_dict):
    """
    Merge contrastive results DataFrames from multiple samples.
    Uses outer merge to include all edges across samples.
    Score columns are suffixed with sample_id.
    """
    if not cres_dict:
        return None

    sample_ids = sorted(cres_dict.keys())
    
    # Start with the first sample
    first_sample = sample_ids[0]
    merged = cres_dict[first_sample].copy()
    
    # Determine merge columns based on whether it's edge or node explanation
    if 'source' in merged.columns and 'target' in merged.columns:
        merge_cols = ['source', 'target']
    elif 'node' in merged.columns:
        merge_cols = ['node']
    else:
        raise ValueError("Cannot determine merge columns - expected 'source'/'target' or 'node'")
    
    # Rename score columns with sample suffix
    score_cols = [c for c in merged.columns if c not in merge_cols]
    rename_dict = {c: f'{c}_{first_sample}' for c in score_cols}
    merged = merged.rename(columns=rename_dict)
    
    # Merge remaining samples
    for sample_id in sample_ids[1:]:
        df = cres_dict[sample_id].copy()
        
        # Rename score columns with sample suffix
        score_cols = [c for c in df.columns if c not in merge_cols]
        rename_dict = {c: f'{c}_{sample_id}' for c in score_cols}
        df = df.rename(columns=rename_dict)
        
        # Outer merge
        merged = merged.merge(df, on=merge_cols, how='outer')
    
    return merged


def aggregate_out_dicts(out_dict_dict):
    """Aggregate out_dicts from multiple samples into a single dictionary."""
    return out_dict_dict


if __name__ == '__main__':
    
    print()
    args = get_args()
    print('--'*40)
    print('Aggregating contrastive explanation results')
    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.out}')
    print('--'*40)

    # Find sample directories
    sample_dirs = find_sample_dirs(args.input_dir)
    print(f'Found {len(sample_dirs)} sample directories')

    if len(sample_dirs) == 0:
        raise ValueError(f'No sample directories found in {args.input_dir}')

    # Load results from each sample
    cres_dict = {}
    out_dict_dict = {}

    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        print(f'Loading results for {sample_id}...')
        
        # Load cres
        cres = load_cres(sample_dir, sample_id)
        if cres is not None:
            cres_dict[sample_id] = cres
            print(f'  - Loaded cres with {len(cres)} rows')
        else:
            print(f'  - WARNING: No cres found for {sample_id}')
        
        # Load out_dict
        out_dict = load_out_dict(sample_dir, sample_id)
        if out_dict is not None:
            out_dict_dict[sample_id] = out_dict
            print(f'  - Loaded out_dict')
        else:
            print(f'  - WARNING: No out_dict found for {sample_id}')

    # Merge cres DataFrames
    print('--'*40)
    print('Merging cres DataFrames...')
    merged_cres = merge_cres_dataframes(cres_dict)
    
    if merged_cres is not None:
        print(f'Merged cres shape: {merged_cres.shape}')
        
        # Save merged cres
        os.makedirs(args.out, exist_ok=True)
        merged_cres_path = os.path.join(args.out, f'{args.name}_cres.csv')
        merged_cres.to_csv(merged_cres_path, index=False)
        print(f'Saved merged cres to {merged_cres_path}')
        
        # Also save as parquet for efficiency
        merged_cres_parquet = os.path.join(args.out, f'{args.name}_cres.parquet')
        merged_cres.to_parquet(merged_cres_parquet, index=False)
        print(f'Saved merged cres to {merged_cres_parquet}')
    else:
        print('WARNING: No cres DataFrames to merge')

    # Aggregate out_dicts
    print('--'*40)
    print('Aggregating out_dicts...')
    aggregated_out_dict = aggregate_out_dicts(out_dict_dict)
    
    if aggregated_out_dict:
        aggregated_out_dict_path = os.path.join(args.out, f'{args.name}_out_dict.pt')
        torch.save(aggregated_out_dict, aggregated_out_dict_path)
        print(f'Saved aggregated out_dict with {len(aggregated_out_dict)} samples to {aggregated_out_dict_path}')
    else:
        print('WARNING: No out_dicts to aggregate')

    print('--'*40)
    print('Aggregation complete!')
    print()

