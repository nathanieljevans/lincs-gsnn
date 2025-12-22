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
import gc


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate contrastive explanation results from multiple samples')

    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing sample subdirectories with results')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for aggregated results')
    parser.add_argument('--name', type=str, default='aggregated',
                       help='Name prefix for output files (default: aggregated)')
    parser.add_argument('--keep_long', action='store_true',
                       help='Keep the intermediate long format CSV file')

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


def concatenate_cres_long_format(sample_dirs, output_path):
    """
    Concatenate contrastive results in LONG format (memory efficient).
    Instead of wide merge, adds a 'sample_id' column to each DataFrame.
    Writes directly to CSV in chunks to minimize memory usage.
    
    Returns (total_rows, merge_cols) where merge_cols are the key columns.
    """
    first_write = True
    total_rows = 0
    merge_cols = None
    
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        cres = load_cres(sample_dir, sample_id)
        
        if cres is None:
            print(f'  - WARNING: No cres found for {sample_id}')
            continue
        
        # Determine merge columns on first load
        if merge_cols is None:
            if 'source' in cres.columns and 'target' in cres.columns:
                merge_cols = ['source', 'target']
            elif 'node' in cres.columns:
                merge_cols = ['node']
            else:
                raise ValueError("Cannot determine merge columns - expected 'source'/'target' or 'node'")
        
        # Add sample_id column
        cres['sample_id'] = sample_id
        
        # Write to CSV (append mode after first write)
        if first_write:
            cres.to_csv(output_path, index=False, mode='w')
            first_write = False
        else:
            cres.to_csv(output_path, index=False, mode='a', header=False)
        
        total_rows += len(cres)
        print(f'  - Wrote {len(cres)} rows for {sample_id}')
        
        del cres
        gc.collect()
    
    return total_rows, merge_cols


def pivot_long_to_wide(long_csv_path, wide_csv_path, merge_cols):
    """
    Convert long format CSV to wide format using pivot_table.
    Reads the concatenated long CSV and pivots score columns by sample_id.
    Uses pivot_table (not pivot) to handle duplicate entries by taking the mean.
    More memory efficient than repeated outer merges.
    """
    print(f'  - Reading long format CSV...')
    df_long = pd.read_csv(long_csv_path)
    print(f'  - Long format shape: {df_long.shape}')
    
    # Check for duplicates
    group_cols = merge_cols + ['sample_id']
    n_dupes = df_long.duplicated(subset=group_cols).sum()
    if n_dupes > 0:
        print(f'  - Found {n_dupes} duplicate entries, will aggregate by mean')
    
    # Identify score columns (everything except merge_cols and sample_id)
    score_cols = [c for c in df_long.columns if c not in merge_cols + ['sample_id']]
    print(f'  - Score columns to pivot: {score_cols}')
    
    # Pivot each score column using pivot_table (handles duplicates via aggfunc)
    pivoted_dfs = []
    for score_col in score_cols:
        pivot_df = df_long.pivot_table(
            index=merge_cols, 
            columns='sample_id', 
            values=score_col,
            aggfunc='mean'  # Aggregate duplicates by taking mean
        )
        # Flatten column names: score_col + sample_id
        pivot_df.columns = [f'{score_col}_{sample_id}' for sample_id in pivot_df.columns]
        pivoted_dfs.append(pivot_df)
        del pivot_df
        gc.collect()
    
    # Combine all pivoted score columns
    if pivoted_dfs:
        df_wide = pd.concat(pivoted_dfs, axis=1).reset_index()
        del pivoted_dfs
        gc.collect()
        
        print(f'  - Wide format shape: {df_wide.shape}')
        df_wide.to_csv(wide_csv_path, index=False)
        print(f'  - Saved wide format to {wide_csv_path}')
        
        del df_wide
        gc.collect()
        return True
    
    return False


def merge_cres_dataframes_incremental(sample_dirs):
    """
    Merge contrastive results DataFrames incrementally from sample directories.
    Uses outer merge to include all edges across samples.
    Score columns are suffixed with sample_id.
    Frees memory after processing each sample to reduce peak memory usage.
    
    WARNING: This can use a lot of memory with many samples. 
    Consider using concatenate_cres_long_format() instead.
    """
    merged = None
    merge_cols = None
    
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        cres = load_cres(sample_dir, sample_id)
        
        if cres is None:
            print(f'  - WARNING: No cres found for {sample_id}')
            continue
            
        print(f'  - Loaded cres for {sample_id} with {len(cres)} rows')
        
        # Determine merge columns on first load
        if merge_cols is None:
            if 'source' in cres.columns and 'target' in cres.columns:
                merge_cols = ['source', 'target']
            elif 'node' in cres.columns:
                merge_cols = ['node']
            else:
                raise ValueError("Cannot determine merge columns - expected 'source'/'target' or 'node'")
        
        # Rename score columns with sample suffix
        score_cols = [c for c in cres.columns if c not in merge_cols]
        cres = cres.rename(columns={c: f'{c}_{sample_id}' for c in score_cols})
        
        if merged is None:
            merged = cres
        else:
            merged = merged.merge(cres, on=merge_cols, how='outer')
            del cres
            gc.collect()
    
    return merged


def aggregate_out_dicts(sample_dirs, output_path):
    """
    Aggregate out_dicts from all samples into a single dictionary.
    Loads each .pt file one at a time and adds to the aggregated dict.
    Result: out_dict[sample_id] = <contents of that sample's .pt file>
    """
    aggregated = {}
    
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir)
        out_dict = load_out_dict(sample_dir, sample_id)
        
        if out_dict is not None:
            aggregated[sample_id] = out_dict
            print(f'  - Loaded out_dict for {sample_id}')
        else:
            print(f'  - WARNING: No out_dict found for {sample_id}')
        
        gc.collect()
    
    # Save aggregated result
    if aggregated:
        torch.save(aggregated, output_path)
        print(f'Saved aggregated out_dict with {len(aggregated)} samples to {output_path}')
    
    return len(aggregated)


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

    os.makedirs(args.out, exist_ok=True)

    # Aggregate cres DataFrames using two-step approach:
    # 1. Stream to long format (memory efficient)
    # 2. Pivot to wide format
    print('--'*40)
    wide_cres_path = os.path.join(args.out, f'{args.name}_cres.csv')
    long_cres_path = os.path.join(args.out, f'{args.name}_cres_long.csv')
    
    # Step 1: Stream concatenate to long format
    print('Step 1: Concatenating cres DataFrames to LONG format (streaming)...')
    total_rows, merge_cols = concatenate_cres_long_format(sample_dirs, long_cres_path)
    
    if total_rows > 0 and merge_cols is not None:
        print(f'  - Total rows in long format: {total_rows}')
        
        # Step 2: Pivot long to wide
        print('--'*40)
        print('Step 2: Pivoting LONG to WIDE format...')
        success = pivot_long_to_wide(long_cres_path, wide_cres_path, merge_cols)
        
        if success:
            # Remove intermediate long file unless requested to keep
            if not args.keep_long:
                os.remove(long_cres_path)
                print(f'  - Removed intermediate long format file')
            else:
                print(f'  - Kept long format file: {long_cres_path}')
        else:
            print('WARNING: Failed to pivot to wide format')
    else:
        print('WARNING: No cres DataFrames to aggregate')

    # Aggregate out_dicts from all samples
    print('--'*40)
    print('Aggregating out_dicts...')
    out_dict_path = os.path.join(args.out, f'{args.name}_out_dict.pt')
    n_aggregated = aggregate_out_dicts(sample_dirs, out_dict_path)
    
    if n_aggregated == 0:
        print('WARNING: No out_dicts found')

    print('--'*40)
    print('Aggregation complete!')
    print()

