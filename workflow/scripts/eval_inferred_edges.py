'''
This script does the following: 

- loads the inferred output edges for each sample (i.e., each csv file in the infer_edges directory) 
- aggregates the inferred output edges across samples using various schemes (mean rank, mean, prob(rank > X), etc. - depends on arguments)
- loads the true output edges that were held out in the `make_bio_network.py` script (i.e., the `removed_edges.csv` file in the bio_network directory) 
- merges the true output edges with the inferred output edges to make a single dataframe with column `hold_out_known_edge` (True if the edge is in the true output edges, False otherwise)
- saves the dataframe to a csv file (in inference_outputs directory)
- evaluates the performance of the inferred edges using various metrics (e.g., MRR, AUROC, top@k) 
- prints evaluations results to console and to file (in inference_outputs directory)

'''

import argparse
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate inferred output edges performance')
    
    parser.add_argument('--infer_edges_dir', type=str, required=True,
                       help='Directory containing inferred edges for each sample')
    parser.add_argument('--bio_network_dir', type=str, required=True,
                       help='Directory containing bionetwork files and removed edges')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--agg_method', type=str, default='mean',
                       choices=['mean', 'median', 'max', 'prob_improved'],
                       help='Method for aggregating scores across samples')
    parser.add_argument('--target_metric', type=str, default='r2_gain',
                       choices=['r2_gain', 'snr', 'p_value', 'r_gain', 'mse_gain', 'r2', 'r', 'mse'],
                       help='Target metric to use for ranking/aggregation (p_value uses inverse ranking)')

    
    return parser.parse_args()


def load_inferred_edges(infer_edges_dir, agg_method='mean', target_metric='r2_gain'):
    """
    Load and aggregate inferred output edges across all samples.
    
    Parameters:
    -----------
    infer_edges_dir : str
        Directory containing subdirectories for each sample with inferred_output_edges_test.csv files
    agg_method : str
        Aggregation method: 'mean' or 'median'
    target_metric : str
        Target metric to use for aggregation ('r2_gain', 'snr', 'p_value')
        
    Returns:
    --------
    pd.DataFrame
        Aggregated results with columns [func_node, output_node, aggregated_score, ...]
    """
    print(f"Loading inferred edges from {infer_edges_dir}")
    
    # Find all sample directories and their CSV files
    csv_files = glob.glob(os.path.join(infer_edges_dir, "*/inferred_output_edges_test.csv"))
    
    if not csv_files:
        raise ValueError(f"No inferred edge CSV files found in {infer_edges_dir}")
    
    print(f"Found {len(csv_files)} sample files")
    
    # Load all CSV files
    dfs = []
    for i, csv_file in enumerate(csv_files):
        print(f'loading sample csv: {i+1}/{len(csv_files)}', end='\r')
        sample_name = os.path.basename(os.path.dirname(csv_file))
        df = pd.read_csv(csv_file, low_memory=False)
        df['sample'] = sample_name
        dfs.append(df)

    # Combine all samples
    combined_df = pd.concat(dfs, ignore_index=True)

    # remove edges that are present in the graph
    combined_df = combined_df[lambda x: ~x['has_edge']]

    print(f"Loaded {len(combined_df)} total edge predictions across {len(csv_files)} samples")
    
    agg_df = combined_df.groupby(['func_node', 'output_node'])[target_metric].mean().reset_index()
    agg_df = agg_df.rename(columns={target_metric: 'mean_score'})

    # remove the OUTPUT__ portion of the output_node
    agg_df = agg_df.assign(output_gene = [x.split('__')[1] for x in agg_df['output_node']])
    
    return agg_df


def load_true_edges(bio_network_dir):
    """
    Load the true output edges that were removed during network construction.
    
    Parameters:
    -----------
    bio_network_dir : str
        Directory containing removed_edges.csv file
        
    Returns:
    --------
    pd.DataFrame
        True edges with columns [func_node, output_node, ...]
    """
    removed_edges_file = os.path.join(bio_network_dir, 'removed_edges.csv')
    
    if not os.path.exists(removed_edges_file):
        raise FileNotFoundError(f"True edges file not found: {removed_edges_file}")
    
    true_edges = pd.read_csv(removed_edges_file, low_memory=False)
    print(f"Loaded {len(true_edges)} true (removed) edges from {removed_edges_file}")

    # Standardize column names to match inferred edges
    true_edges = true_edges.rename(columns={'src_name': 'func_node', 'dst_name': 'output_node_raw'})

    # need to remove the RNA__ portion of the dst_name 
    true_edges = true_edges.assign(output_gene = [x.split('__')[1] for x in true_edges['output_node_raw']])
    
    return true_edges[['func_node', 'output_gene', 'output_node_raw']].drop_duplicates()


def merge_edges(inferred_edges, true_edges):
    """
    Merge inferred and true edges, adding hold_out_known_edge column.
    
    Parameters:
    -----------
    inferred_edges : pd.DataFrame
        Aggregated inferred edges
    true_edges : pd.DataFrame
        True edges that were removed
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with hold_out_known_edge column
    """

    true_edges = true_edges.assign(hold_out_known_edge = True)

    inferred_edges = inferred_edges.merge(true_edges, on=['func_node', 'output_gene'], how='left') 
    inferred_edges['hold_out_known_edge'] = inferred_edges['hold_out_known_edge'].fillna(False)

    assert inferred_edges['hold_out_known_edge'].sum() == true_edges.shape[0], f"Number of true edges {true_edges.shape[0]} does not match number of inferred edges {inferred_edges['hold_out_known_edge'].sum()}"

    return inferred_edges


def evaluate_performance(merged_df, target_metric='r2', out_dir='.'):
    """
    Evaluate performance using various metrics.
    
    Parameters:
    -----------
    merged_df : pd.DataFrame
        Merged dataframe with hold_out_known_edge and aggregated_score columns
    target_metric : str
        The target metric that was used for aggregation (affects score interpretation)
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    print("Evaluating performance...")
    
    # Prepare data
    y_true = merged_df['hold_out_known_edge'].values.astype(int)
    y_scores = merged_df['mean_score'].values

    if target_metric in ['r2_gain', 'mse_gain']:
        scale = -1
    else:
        scale = 1
    
    overall_auroc = roc_auc_score(y_true, y_scores * scale) 
    overall_ap = average_precision_score(y_true, y_scores * scale)  


    _min = np.min(y_scores * scale)
    _max = np.max(y_scores * scale)
    _bins = np.linspace(_min, _max, 50)
    plt.figure() 
    plt.hist(merged_df[lambda x: x.hold_out_known_edge].mean_score.values * scale, bins=_bins, color='r', alpha=0.5, label='Positives', density=True)
    plt.hist(merged_df[lambda x: ~x.hold_out_known_edge].mean_score.values * scale, bins=_bins, color='b', alpha=0.5, label='Negatives', density=True)
    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.savefig(f'{out_dir}/score_distribution.png')
    plt.close()


    ranks = [] 
    for i, row in merged_df[lambda x: x.hold_out_known_edge].reset_index().iterrows():
        print(f'evaluating within output gene rank: {i+1}/{len(merged_df[lambda x: x.hold_out_known_edge])}', end='\r')
        negatives = merged_df[lambda x: ~x.hold_out_known_edge & (x.output_node == row.output_node)].mean_score.values * scale 
        positives = row.mean_score * scale 
        rank = (negatives >= positives).sum() + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    mrr = np.mean(1.0 / ranks)
    top_1 = np.mean(ranks <= 1)
    top_3 = np.mean(ranks <= 3)
    top_10 = np.mean(ranks <= 10)
    top_100 = np.mean(ranks <= 100)
    top_250 = np.mean(ranks <= 250) 

    results = {
        'overall_auroc': overall_auroc,
        'overall_ap': overall_ap,
        'mrr': mrr,
        'top_1': top_1,
        'top_3': top_3,
        'top_10': top_10,
        'top_100': top_100,
        'top_250': top_250
    }

    return results


def print_and_save_results(results, out_dir):
    """
    Print results to console and save to file.
    
    Parameters:
    -----------
    results : dict
        Evaluation results
    out_dir : str
        Output directory
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    
    # Print top@k metrics
    print(f"\nResults:")
    for metric, value in results.items():
        print(f"\t->{metric}: {value:.4f}")
    
    print("="*60)
    
    # Save to file
    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, 'evaluation_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_file}")


def main():
    args = get_args()
    
    print("Starting evaluation of inferred edges...")
    print(f"Infer edges directory: {args.infer_edges_dir}")
    print(f"Bio network directory: {args.bio_network_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Aggregation method: {args.agg_method}")
    print(f"Target metric: {args.target_metric}")
    
    # Load and aggregate inferred edges
    inferred_edges = load_inferred_edges(
        args.infer_edges_dir, 
        agg_method=args.agg_method, 
        target_metric=args.target_metric
    )
    
    # Load true edges
    true_edges = load_true_edges(args.bio_network_dir)
    
    # Merge edges
    merged_edges = merge_edges(inferred_edges, true_edges)
    
    # Save merged results
    os.makedirs(args.out_dir, exist_ok=True)
    merged_file = os.path.join(args.out_dir, 'merged_edges_evaluation.csv')
    merged_edges.to_csv(merged_file, index=False)
    print(f"Saved merged edges to {merged_file}")
    
    # Evaluate performance
    results = evaluate_performance(merged_edges, target_metric=args.target_metric, out_dir=args.out_dir)
    
    # Print and save results
    print_and_save_results(results, args.out_dir)


if __name__ == '__main__':
    main()