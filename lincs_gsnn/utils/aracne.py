import shutil
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch


def build_candidate_edges(data, gene_names, net_dir):
    """Build the PROTEIN->RNA candidate edge DataFrame with train/test/negative labels.

    Filters to non-self, candidate-only edges whose source and target are
    in the function graph.  Isolate-target handling is deferred to
    ``eval_edge_inference``.
    """
    net_dir = Path(net_dir)
    node_names = np.array(data.node_names_dict['function'])

    res = pd.DataFrame({
        'source': node_names[data.edge_index_dict['function', 'to', 'function'][0]],
        'target': node_names[data.edge_index_dict['function', 'to', 'function'][1]],
    }).assign(train_edge=True)

    candidates = pd.DataFrame({
        'source': ['PROTEIN__' + g1 for g1 in gene_names for g2 in gene_names],
        'target': ['RNA__' + g2 for g1 in gene_names for g2 in gene_names],
    }).assign(candidate=True)

    test_edges = pd.read_csv(net_dir / 'removed_edges.csv')
    test_edges = (
        test_edges[['src_name', 'dst_name']]
        .rename(columns={'src_name': 'source', 'dst_name': 'target'})
        .assign(test_edge=True)
    )

    res = res.merge(candidates, on=['source', 'target'], how='outer')
    res = res.merge(test_edges, on=['source', 'target'], how='outer')

    for col in ['train_edge', 'candidate', 'test_edge']:
        res[col] = res[col].fillna(False).astype(bool)

    res = res.assign(negative=lambda x: ~x.test_edge & ~x.train_edge)
    res = res.assign(source_gene=lambda x: x.source.str.split('__', expand=True)[1])
    res = res.assign(target_gene=lambda x: x.target.str.split('__', expand=True)[1])

    gene2idx = pd.DataFrame({'gene': gene_names, 'gene_idx': range(len(gene_names))})
    res = res.merge(
        gene2idx.rename(columns={'gene_idx': 'source_gene_idx', 'gene': 'source_gene'}),
        on='source_gene', how='left',
    )
    res = res.merge(
        gene2idx.rename(columns={'gene_idx': 'target_gene_idx', 'gene': 'target_gene'}),
        on='target_gene', how='left',
    )

    res = res[res.source.isin(data.node_names_dict['function']) & res.target.isin(data.node_names_dict['function'])]
    res = res[lambda x: x.source.str.contains('PROTEIN__') & x.target.str.contains('RNA__')]
    res = res[lambda x: x.candidate]
    res = res[lambda x: x.source_gene != x.target_gene]

    return res.reset_index(drop=True)


def load_expression_matrix(sample_dir, obs_meta, gene_names, stride=1,
                           subtract_baseline=False):
    """Load expression trajectories as a (genes x samples) DataFrame.

    Parameters
    ----------
    stride : int
        Keep every ``stride``-th timepoint (1 = all timepoints).
    subtract_baseline : bool
        If True, subtract the t=0 value from every timepoint (and drop t=0).
        Disabled by default because ARACNe relies on cross-sample variance
        in raw expression; baseline subtraction removes that signal.
    """
    sample_dir = Path(sample_dir)
    xs = []
    for _, row in obs_meta.iterrows():
        xs.append(torch.load(sample_dir / row.file_name, weights_only=False))

    xs = torch.stack(xs, dim=0)              # (N_obs, N_time, N_genes)
    if subtract_baseline:
        xs = xs - xs[:, [0], :]
        xs = xs[:, 1::stride, :]             # drop t=0 (all zeros), then stride
    else:
        xs = xs[:, ::stride, :]

    n_obs, n_time, n_genes = xs.shape
    xs_flat = xs.reshape(-1, n_genes).detach().cpu().numpy().astype(np.float32, copy=False)

    sample_ids = [
        f'{row.file_name}__t{t:02d}'
        for row in obs_meta.itertuples(index=False)
        for t in range(n_time)
    ]

    return pd.DataFrame(xs_flat.T, index=gene_names, columns=sample_ids)


def write_aracne_inputs(expr_df, regulators, input_dir, output_dir):
    """Write expression matrix, TF list, and dummy MI threshold for ARACNe-AP.

    Cleans and recreates ``output_dir`` so stale bootstrap/threshold files
    don't contaminate results.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    expr_path = input_dir / 'matrix.txt'
    tf_path = input_dir / 'tfs.txt'

    expr_out = expr_df.copy().fillna(0.0)
    expr_out.insert(0, 'gene', expr_out.index)
    expr_out.to_csv(expr_path, sep='\t', index=False)

    regulators.to_csv(tf_path, index=False, header=False)

    n_samples = expr_df.shape[1]
    with open(output_dir / f'miThreshold_p1E0_samples{n_samples}.txt', 'w') as f:
        f.write('0.0')

    return expr_path, tf_path


def run_aracne(aracne_jar, expr_path, output_dir, tf_path, seed=1):
    """Run ARACNe-AP in nobootstrap mode via subprocess."""
    cmd = [
        'java', '-Xmx16G', '-jar', str(aracne_jar),
        '-e', str(expr_path),
        '-o', str(output_dir),
        '--tfs', str(tf_path),
        '--pvalue', '1',
        '--seed', str(seed),
        '--nobootstrap',
        '--nobonferroni',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ARACNe failed (exit {result.returncode}):\n{result.stderr}')
    return result


def load_aracne_network(output_dir):
    """Load ``nobootstrap_network.txt`` and return symmetrised directed edge scores.

    Returns a DataFrame with columns ``[source_gene, target_gene, score]``.
    """
    network_path = Path(output_dir) / 'nobootstrap_network.txt'
    if not network_path.exists():
        raise FileNotFoundError(f'Expected {network_path}')

    aracne_net = pd.read_csv(
        network_path, sep=r'\s+', skiprows=1, header=None,
        names=['gene_a', 'gene_b', 'mi', 'pvalue'],
    )

    fwd = aracne_net.rename(columns={'gene_a': 'source_gene', 'gene_b': 'target_gene', 'mi': 'score'})
    rev = aracne_net.rename(columns={'gene_b': 'source_gene', 'gene_a': 'target_gene', 'mi': 'score'})
    aracne_edges = pd.concat([fwd, rev], ignore_index=True).drop_duplicates(subset=['source_gene', 'target_gene'])

    return aracne_edges[['source_gene', 'target_gene', 'score']]


def score_candidates(res, aracne_edges):
    """Merge ARACNe MI scores onto candidate edges, filling undetected with 0."""
    res_cand = res.merge(
        aracne_edges[['source_gene', 'target_gene', 'score']],
        on=['source_gene', 'target_gene'],
        how='left',
    )
    res_cand['score'] = res_cand['score'].fillna(0.0)
    return res_cand


def run_sample_pipeline(sample_i, traj_dir, obs_meta, gene_names, regulators,
                        aracne_root, aracne_jar, res, stride=1,
                        subtract_baseline=False):
    """End-to-end ARACNe pipeline for a single trajectory sample.

    Returns a Series of MI scores aligned with ``res.index``.
    """
    traj_dir = Path(traj_dir)
    aracne_root = Path(aracne_root)

    sample_dir = traj_dir / 'predict_grid' / f'sample_{sample_i}' / 'obs'
    input_dir = aracne_root / 'inputs' / f'sample_{sample_i}'
    output_dir = aracne_root / 'output' / f'sample_{sample_i}'

    expr_df = load_expression_matrix(sample_dir, obs_meta, gene_names,
                                     stride=stride,
                                     subtract_baseline=subtract_baseline)
    expr_path, tf_path = write_aracne_inputs(expr_df, regulators, input_dir, output_dir)
    run_aracne(aracne_jar, expr_path, output_dir, tf_path, seed=1)
    aracne_edges = load_aracne_network(output_dir)
    scored = score_candidates(res, aracne_edges)

    return scored['score']
