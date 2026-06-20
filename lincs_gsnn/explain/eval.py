
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sbn 
import json
import os
import warnings
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from lincs_gsnn.explain.topology import topology_baseline_score
from lincs_gsnn.explain.paths import path_score
import networkx as nx
from lincs_gsnn.explain.utils import predict_node_activity
from lincs_gsnn.proc.model_paths import gsnn_model_path
import torch
from scipy.stats import ttest_rel


################################################################################################################
################################################################################################################
################################################################################################################


def eval_traj_diff(path):
    """Compute per-sample trajectory AUC change for two cell lines and test
    whether cell line 1 has a more negative response than cell line 2.

    Returns
    -------
    auc1, auc2 : np.ndarray, shape (n_samples,)
        Trapezoidal AUC of (xt - xt[t=0]) for each cell line.
    auc_diff : np.ndarray, shape (n_samples,)
        ``auc1 - auc2``.
    pvalue : float
        One-sided Welch t-test p-value for H1: mean(auc1) < mean(auc2).
    """

    out_dict = torch.load(f'{path}/aggregated_out_dict.pt', weights_only=False)

    xt1 = []
    xt2 = []
    for model_id, model_dict in out_dict.items():
        target_ix = model_dict['target_gene_output_ix']
        xt1.append(model_dict['xt_hat_1'][:, target_ix])
        xt2.append(model_dict['xt_hat_2'][:, target_ix])

    xt1 = np.stack(xt1, axis=0)  # shape: (n_samples, n_timesteps)
    xt2 = np.stack(xt2, axis=0)  # shape: (n_samples, n_timesteps)

    # keep dim so broadcasting against (n_samples, n_timesteps) works correctly
    xt1_t0 = xt1[:, [0]]  # shape: (n_samples, 1)
    xt2_t0 = xt2[:, [0]]  # shape: (n_samples, 1)

    auc1 = np.trapz(xt1 - xt1_t0, axis=1)  # AUC of change from t=0 per sample
    auc2 = np.trapz(xt2 - xt2_t0, axis=1)
    auc_diff = auc1 - auc2  # shape: (n_samples,)

    # H1: mean(auc1) < mean(auc2) i.e. cell_line_1 has a more negative response
    ttest = ttest_rel(auc1, auc2, alternative='less')

    return auc1, auc2, auc_diff, ttest.pvalue



################################################################################################################
################################################################################################################
################################################################################################################

def eval_node_activity(root_gsnn, root_traj, model_id, plot=False, save_dir=None, model_path=None): 
    """Evaluate node activity scores.

    Returns ``(None, None)`` when the loaded GSNN has no node-activity module.
    """
    data = torch.load(os.path.join(root_gsnn, 'bionetwork/bionetwork.pt'), weights_only=False)
    resolved_model_path = gsnn_model_path(root_gsnn, model_id, model_path=model_path)
    model = torch.load(resolved_model_path, weights_only=False, map_location='cpu')
    na_module = getattr(model, 'node_activity_model', None)
    if not getattr(model, 'node_activity', False) or na_module is None:
        warnings.warn(
            f"Skipping eval_node_activity for {model_id!r}: "
            "pretrained model has no node_activity module.",
            stacklevel=2,
        )
        return None, None

    na_path = os.path.join(root_gsnn, 'bionetwork/node_activity.pt')
    if not os.path.isfile(na_path):
        warnings.warn(
            f"Skipping eval_node_activity for {model_id!r}: "
            f"node activity artifact not found at {na_path!r}.",
            stacklevel=2,
        )
        return None, None

    del model

    cells = [x.split('__')[1] for x in data.node_names_dict['input'] if 'LINE__' in x]
    nadf =[]
    for cell in cells: 
        try: 
            nadf.append( predict_node_activity(root_gsnn=root_gsnn, root_traj=root_traj, model_id=model_id, cell=cell).assign(cell_line=cell) ) 
        except Exception as KeyError:
            print(f'missing cell iname ({cell}) in node activity artifact')
            continue
    nadf = pd.concat(nadf)

    activity_score_names = [
        c for c in nadf.columns if c.startswith('node_activity_score')
    ]
    activity_feat_names = [
        c for c in nadf.columns
        if c not in ('node_name', 'mode', 'cell_line')
        and not c.startswith('node_activity_score')
    ]

    mean_score_vars = nadf[activity_score_names].var(0).mean(0) # is there any variation in the node activity scores?

    corr_cols = list(activity_score_names) + list(activity_feat_names)
    corrs = nadf[corr_cols].corr(method='spearman', numeric_only=True)
    corrs = corrs.loc[activity_score_names]


    if plot:
        n_scores = len(activity_score_names)
        n_feats = len(activity_feat_names)
        ncells = nadf['cell_line'].nunique()
        n_unique_feats = nadf[activity_feat_names].drop_duplicates().shape[0]

        plot_fig = None
        if n_unique_feats == ncells:
            heatmap_df = (
                nadf.groupby('cell_line', sort=True)[activity_score_names]
                .mean()
            )
            row_order = leaves_list(
                linkage(
                    pdist(heatmap_df.values, metric='euclidean'),
                    method='average',
                )
            )
            heatmap_df = heatmap_df.iloc[row_order]
            plot_fig, ax = plt.subplots(
                figsize=(n_scores + 2, ncells * 0.35 + 1.5),
            )
            sbn.heatmap(
                heatmap_df,
                ax=ax,
                cmap='viridis',
                cbar_kws={'label': 'mean node activity score', 'shrink': 0.8},
            )
            ax.set_xticklabels([str(i) for i in range(n_scores)])
            ax.set_xlabel('node activity score')
            ax.set_ylabel('cell line')
            ax.set_title('Mean node activity gate by cell line')
        else:
            # Per-node features (e.g. expr/mut): one scatter per score/feature pair.
            fig, axes = plt.subplots(
                n_scores,
                n_feats,
                figsize=(2.5 * n_feats, 2.5 * n_scores),
                squeeze=False,
            )
            rng = np.random.default_rng(0)
            for i, score_col in enumerate(activity_score_names):
                for j, feat_col in enumerate(activity_feat_names):
                    ax = axes[i, j]
                    x = nadf[feat_col].to_numpy(dtype=float)
                    y = nadf[score_col].to_numpy(dtype=float)
                    if nadf[feat_col].nunique() <= 10:
                        x = x + rng.uniform(-0.04, 0.04, size=len(x))
                    ax.scatter(x, y, s=4, alpha=0.3)
                    if i == n_scores - 1:
                        ax.set_xlabel(feat_col)
                    if j == 0:
                        ax.set_ylabel(score_col)
            plot_fig = fig
            plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plot_fig.savefig(
                os.path.join(
                    save_dir,
                    f'node_activity_scores__plot_{model_id}.png',
                ),
                dpi=300,
                bbox_inches='tight',
            )
        else:
            plt.show()
        plt.close()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        pd.DataFrame({'mean_score_var': [mean_score_vars]}).to_csv(
            os.path.join(save_dir, f'node_activity_scores__mean_score_variation.csv'),
            index=False,
        )
        corrs.to_csv(os.path.join(save_dir, f'node_activity_scores__correlations.csv'))

    return mean_score_vars, corrs

################################################################################################################
################################################################################################################
################################################################################################################

def _replicate_score_cols(columns, score_prefix: str) -> list[str]:
    """Return wide-format replicate score columns for one scorer.

    Aggregated explanation CSVs suffix replicates as either
    ``{score_prefix}_model_<id>`` (current) or ``{score_prefix}_sample_<id>``
    (legacy).
    """
    cols = []
    head = f'{score_prefix}_'
    for col in columns:
        if not col.startswith(head):
            continue
        replicate_id = col[len(head):]
        if replicate_id.startswith('model_') or replicate_id.startswith('sample_'):
            cols.append(col)
    return cols


def agg_edge_scores(path, fill_value=np.nan):
    """Aggregate per-replicate edge scores into row-wise means.

    Reads an edge-level score CSV and averages GSNN, integrated-gradient, and
    occlusion replicate columns separately, skipping missing values.
    """

    edge_cell1 = pd.read_csv(path)

    gsnn_score_samples = _replicate_score_cols(edge_cell1.columns, 'gsnn_score')
    ig_score_samples = _replicate_score_cols(edge_cell1.columns, 'ig_score')
    oc_score_samples = _replicate_score_cols(edge_cell1.columns, 'occlusion_score')

    edge_cell1_gsnn = edge_cell1[['source', 'target'] + gsnn_score_samples].fillna(fill_value)
    edge_cell1_gsnn = edge_cell1_gsnn.set_index(['source', 'target'])
    edge_cell1_gsnn = edge_cell1_gsnn.assign(mean_gsnn_score = edge_cell1_gsnn.mean(axis=1, skipna=True).values)
    edge_cell1_gsnn = edge_cell1_gsnn.assign(std_gsnn_score = edge_cell1_gsnn.std(axis=1, skipna=True).values)
    edge_cell1_gsnn = edge_cell1_gsnn[['mean_gsnn_score', 'std_gsnn_score']]

    edge_cell1_ig = edge_cell1[['source', 'target'] + ig_score_samples].fillna(fill_value)
    edge_cell1_ig = edge_cell1_ig.set_index(['source', 'target'])
    edge_cell1_ig = edge_cell1_ig.assign(mean_ig_score = edge_cell1_ig.mean(axis=1, skipna=True).values)
    edge_cell1_ig = edge_cell1_ig.assign(std_ig_score = edge_cell1_ig.std(axis=1, skipna=True).values)
    edge_cell1_ig = edge_cell1_ig[['mean_ig_score', 'std_ig_score']]

    edge_cell1_oc  = edge_cell1[['source', 'target'] + oc_score_samples].fillna(fill_value)
    edge_cell1_oc = edge_cell1_oc.set_index(['source', 'target'])
    edge_cell1_oc = edge_cell1_oc.assign(mean_oc_score = edge_cell1_oc.mean(axis=1, skipna=True).values)
    edge_cell1_oc = edge_cell1_oc.assign(std_oc_score = edge_cell1_oc.std(axis=1, skipna=True).values)
    edge_cell1_oc = edge_cell1_oc[['mean_oc_score', 'std_oc_score']]

    edge_cell1_agg = edge_cell1_gsnn.merge(edge_cell1_ig, on=['source', 'target'], how='outer').merge(edge_cell1_oc, on=['source', 'target'], how='outer')
    edge_cell1_agg = edge_cell1_agg.assign(abs_ig_score = lambda x: x.mean_ig_score.abs(),
                                        abs_oc_score = lambda x: x.mean_oc_score.abs())

    edge_cell1_agg = edge_cell1_agg.reset_index()

    edge_cell1_agg = edge_cell1_agg.fillna(0)

    return edge_cell1_agg

################################################################################################################
################################################################################################################
################################################################################################################

def agg_node_scores(path, fill_value=np.nan):
    """Aggregate per-replicate node scores into row-wise means.

    Reads a node-level score CSV and averages GSNN, integrated-gradient, and
    occlusion replicate columns separately, skipping missing values.
    """
    node_cell1 = pd.read_csv(path)

    gsnn_score_samples = _replicate_score_cols(node_cell1.columns, 'gsnn_score')
    ig_score_samples = _replicate_score_cols(node_cell1.columns, 'ig_score')
    oc_score_samples = _replicate_score_cols(node_cell1.columns, 'occlusion_score')

    node_cell1_gsnn = node_cell1[['node'] + gsnn_score_samples].fillna(fill_value).set_index('node')
    node_cell1_gsnn = node_cell1_gsnn.assign(mean_gsnn_score=node_cell1_gsnn.mean(axis=1, skipna=True))[['mean_gsnn_score']]

    node_cell1_ig = node_cell1[['node'] + ig_score_samples].fillna(fill_value).set_index('node')
    node_cell1_ig = node_cell1_ig.assign(mean_ig_score=node_cell1_ig.mean(axis=1, skipna=True))[['mean_ig_score']]

    node_cell1_oc = node_cell1[['node'] + oc_score_samples].fillna(fill_value).set_index('node')
    node_cell1_oc = node_cell1_oc.assign(mean_oc_score=node_cell1_oc.mean(axis=1, skipna=True))[['mean_oc_score']]

    node_cell1_agg = (
        node_cell1_gsnn.merge(node_cell1_ig, on='node', how='outer')
        .merge(node_cell1_oc, on='node', how='outer')
        .reset_index()
    )

    # any remaining missing values should be filled with 0
    node_cell1_agg = node_cell1_agg.fillna(0)

    return node_cell1_agg

################################################################################################################
################################################################################################################
################################################################################################################

def data2nx(data): 
    '''
    Convert a pyg data object to a networkx directed graph.
    '''
    node_names_dict = data.node_names_dict
    edge_index_dict = data.edge_index_dict 

    edges_i2f = pd.DataFrame({'source': np.array(node_names_dict['input'])[edge_index_dict['input', 'to', 'function'][0]], 'target': np.array(node_names_dict['function'])[edge_index_dict['input', 'to', 'function'][1]]})
    edges_f2o = pd.DataFrame({'source': np.array(node_names_dict['function'])[edge_index_dict['function', 'to', 'output'][0]], 'target': np.array(node_names_dict['output'])[edge_index_dict['function', 'to', 'output'][1]]})
    edges_f2f = pd.DataFrame({'source': np.array(node_names_dict['function'])[edge_index_dict['function', 'to', 'function'][0]], 'target': np.array(node_names_dict['function'])[edge_index_dict['function', 'to', 'function'][1]]})
    
    edges = pd.concat([edges_i2f, edges_f2o, edges_f2f])
    G = nx.from_pandas_edgelist(edges, source='source', target='target', create_using=nx.DiGraph())
    return G

################################################################################################################
################################################################################################################
################################################################################################################

def _loc_scalar(df, key, col):
    """Return a single scalar from a DataFrame ``.loc`` lookup."""
    val = df.loc[key, col]
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val


def _edge_baselines(edge_df, G, source_node, target_node, verbose=True, rank_kwargs = {'method': 'max'}, expected_direction = 'negative'): 

    if verbose: print('generating random walk baseline...')
    baseline_rw = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='random_walk',  level='edge',
    ).to_frame().rename(columns={'topology_score': 'rw_score'}).reset_index()

    if verbose: print('generating pagerank baseline...')
    baseline_pr = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='pagerank',  level='edge',
    ).to_frame().rename(columns={'topology_score': 'pr_score'}).reset_index()

    if verbose: print('generating rw-betweenness baseline...')
    baseline_rwb = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='rw_betweenness',  level='edge',
    ).to_frame().rename(columns={'topology_score': 'rwb_score'}).reset_index()

    if verbose: print('generating betweenness centrality baseline...')
    baseline_bc = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='betweenness_centrality',  level='edge',
    ).to_frame().rename(columns={'topology_score': 'bc_score'}).reset_index()

    dfbe = edge_df[['source', 'target', 'mean_gsnn_score', 'mean_ig_score', 'mean_oc_score']]
    dfbe = dfbe.merge(baseline_rw, on=['source', 'target'], how='left')
    dfbe = dfbe.merge(baseline_pr, on=['source', 'target'], how='left')
    dfbe = dfbe.merge(baseline_rwb, on=['source', 'target'], how='left')
    dfbe = dfbe.merge(baseline_bc, on=['source', 'target'], how='left').copy() # fragmentation 

    dfbe = dfbe.assign(gsnn_rank = lambda x: x.mean_gsnn_score.rank(**rank_kwargs, ascending=False))
    dfbe = dfbe.assign(ig_rank = lambda x: x.mean_ig_score.rank(**rank_kwargs, ascending=expected_direction == 'negative'))
    dfbe = dfbe.assign(oc_rank = lambda x: x.mean_oc_score.rank(**rank_kwargs, ascending=expected_direction == 'negative'))
    dfbe = dfbe.assign(rw_rank = lambda x: x.rw_score.rank(**rank_kwargs, ascending=False))
    dfbe = dfbe.assign(pr_rank = lambda x: x.pr_score.rank(**rank_kwargs, ascending=False))
    dfbe = dfbe.assign(rwb_rank = lambda x: x.rwb_score.rank(**rank_kwargs, ascending=False))
    dfbe = dfbe.assign(bc_rank = lambda x: x.bc_score.rank(**rank_kwargs, ascending=False))

    return dfbe 


################################################################################################################
################################################################################################################
################################################################################################################

def path_ranking_comparison(path_df, G, target_node, source_node, rank_kwargs = {'method': 'max'}, expected_paths = None, 
                            expected_direction = 'negative', verbose=True, max_path_length = 5,
                            final_scores = ['mean_gsnn_score_product', 
                                            'mean_ig_score_sum', 
                                            'mean_oc_score_sum', 
                                            'rw_score_sum', 
                                            'pr_score_sum', 
                                            'rwb_score_sum', 
                                            'bc_score_sum']): 
                            
    """Rank paths based on GSNN, integrated-gradient, and occlusion scores.

    Compares GSNN, integrated-gradient (IG), and occlusion (OC) path-importance
    scores for paths whose ``target`` column equals ``target_node``. Each score method
    is ranked separately. GSNN always uses descending rank (higher score is better).
    IG and OC use ascending rank when ``expected_direction`` is ``'negative'`` and
    descending rank otherwise.
    """

    dfbe = _edge_baselines(path_df, G, source_node, target_node, verbose, rank_kwargs, expected_direction)

    # need to add edge attributes to G 
    if verbose: print('adding edge attributes to G...')
    if dfbe.index.names != ['source', 'target']:
        dfbe_ = dfbe.set_index(['source', 'target'])
    else:
        dfbe_ = dfbe

    transform = lambda x: x # deprecated 
    for edge in G.edges():
        G.edges[edge]['mean_gsnn_score'] = transform(_loc_scalar(dfbe_, edge, 'mean_gsnn_score'))
        G.edges[edge]['mean_ig_score'] = transform(_loc_scalar(dfbe_, edge, 'mean_ig_score'))
        G.edges[edge]['mean_oc_score'] = transform(_loc_scalar(dfbe_, edge, 'mean_oc_score'))
        G.edges[edge]['rw_score'] = transform(_loc_scalar(dfbe_, edge, 'rw_score'))
        G.edges[edge]['pr_score'] = transform(_loc_scalar(dfbe_, edge, 'pr_score'))
        G.edges[edge]['rwb_score'] = transform(_loc_scalar(dfbe_, edge, 'rwb_score'))
        G.edges[edge]['bc_score'] = transform(_loc_scalar(dfbe_, edge, 'bc_score'))

    if verbose: print('scoring paths (product) ...')
    path_df_prod = path_score(G, source_node=source_node, target_node=target_node, 
                        cutoff=max_path_length, method='product', rank_method='max', signed=False)

    if verbose: print('scoring paths (sum) ...')
    path_df_sum = path_score(G, source_node=source_node, target_node=target_node, 
                        cutoff=max_path_length, method='sum', rank_method='max', signed=True)

    if verbose: print('scoring paths (mean) ...')
    path_df_mean = path_score(G, source_node=source_node, target_node=target_node, 
                        cutoff=max_path_length, method='mean', rank_method='max', signed=True)

    path_df = path_df_prod.merge(path_df_sum, on=['path_short', 'path_length'], how='left')
    path_df = path_df.merge(path_df_mean, on=['path_short', 'path_length'], how='left')

    path_df = path_df.assign(**{
        f'{c}_rank': (lambda x, c=c: x[c].rank(**rank_kwargs, ascending=(expected_direction == 'negative') if c in ['mean_ig_score_sum', 'mean_oc_score_sum'] else False))
        for c in final_scores
    })

    path_df_expected = path_df[lambda x: x.path_short.isin(expected_paths)][['path_short'] + [c + '_rank' for c in final_scores]]

    mrr_df = pd.DataFrame({c: [np.mean(1 / path_df_expected[f'{c}_rank'].values)] for c in final_scores})

    mrr_df.style.hide().format({c: '{:.3f}' for c in final_scores})

    return mrr_df, path_df_expected


################################################################################################################
################################################################################################################
################################################################################################################

def edge_ranking_comparison(edge_df, G, target_node, source_node, rank_kwargs = {'method': 'max'}, expected_edges = None, expected_direction = 'negative', verbose=True): 
    """Rank edges based on GSNN, integrated-gradient, and occlusion scores.

    Compares GSNN, integrated-gradient (IG), and occlusion (OC) edge-importance
    scores for edges whose ``target`` column equals ``target_node``. Each score method
    is ranked separately. GSNN always uses descending rank (higher score is better).
    IG and OC use ascending rank when ``expected_direction`` is ``'negative'`` and
    descending rank otherwise.
    """

    dfbe = _edge_baselines(edge_df, G, source_node, target_node, verbose, rank_kwargs, expected_direction)

    dfbe_involved = dfbe.merge(expected_edges, on=['source', 'target'], how='inner')

    mrr_gsnn = np.mean(1 / dfbe_involved.gsnn_rank.values)
    mrr_ig = np.mean(1 / dfbe_involved.ig_rank.values)
    mrr_oc = np.mean(1 / dfbe_involved.oc_rank.values)
    mrr_rw = np.mean(1 / dfbe_involved.rw_rank.values)
    mrr_pr = np.mean(1 / dfbe_involved.pr_rank.values)
    mrr_rwb = np.mean(1 / dfbe_involved.rwb_rank.values)
    mrr_bc = np.mean(1 / dfbe_involved.bc_rank.values)

    mrr_df_edge = pd.DataFrame({'gsnn': [mrr_gsnn], 'ig': [mrr_ig], 'oc': [mrr_oc], 'rw': [mrr_rw], 'pr': [mrr_pr], 'rwb': [mrr_rwb], 'bc': [mrr_bc]})

    dfbe_involved = dfbe_involved[['source', 'target', 'gsnn_rank', 'ig_rank', 'oc_rank', 'rw_rank', 'pr_rank', 'rwb_rank', 'bc_rank']].style.hide().format({'gsnn_rank': '{:.0f}', 'ig_rank': '{:.0f}', 'oc_rank': '{:.0f}', 'rw_rank': '{:.0f}', 'pr_rank': '{:.0f}', 'rwb_rank': '{:.0f}', 'bc_rank': '{:.0f}'})

    return mrr_df_edge, dfbe_involved




################################################################################################################
################################################################################################################
################################################################################################################

def node_ranking_comparison(node_df, G, target_node, source_node, rank_kwargs = {'method': 'max'}, expected_nodes = None, expected_direction = 'negative', verbose=True): 
    """Rank nodes based on GSNN, integrated-gradient, and occlusion scores.

    Compares GSNN, integrated-gradient (IG), and occlusion (OC) node-importance
    scores for nodes whose ``node`` column equals ``target_node``. Each score method
    is ranked separately. GSNN always uses descending rank (higher score is better).
    IG and OC use ascending rank when ``expected_direction`` is ``'negative'`` and
    descending rank otherwise.
    """
    if verbose: print('generating random walk baseline...')
    baseline_rw = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='random_walk',  level='node',
    ).to_frame().rename(columns={'topology_score': 'rw_score'})

    if verbose: print('generating pagerank baseline...')
    baseline_pr = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='pagerank',  level='node',
    ).to_frame().rename(columns={'topology_score': 'pr_score'})

    if verbose: print('generating rw-betweenness baseline...')
    baseline_rwb = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='rw_betweenness',  level='node',
    ).to_frame().rename(columns={'topology_score': 'rwb_score'})

    if verbose: print('generating betweenness centrality baseline...')
    baseline_bc = topology_baseline_score(
        G, source_node=source_node, target_node=target_node,
        method='betweenness_centrality',  level='node',
    ).to_frame().rename(columns={'topology_score': 'bc_score'})

    cres_ = node_df.reset_index()[['node', 'mean_gsnn_score', 'mean_ig_score', 'mean_oc_score']]
    dfb = cres_.merge(baseline_rw, on='node', how='left')
    dfb = dfb.merge(baseline_pr, on='node', how='left')
    dfb = dfb.merge(baseline_rwb, on='node', how='left')
    dfb = dfb.merge(baseline_bc, on='node', how='left')

    dfb = dfb.assign(gsnn_rank = lambda x: x.mean_gsnn_score.rank(**rank_kwargs, ascending=False))

    dfb = dfb.assign(ig_rank = lambda x: x.mean_ig_score.rank(**rank_kwargs, ascending=expected_direction == 'negative'))
    dfb = dfb.assign(oc_rank = lambda x: x.mean_oc_score.rank(**rank_kwargs, ascending=expected_direction == 'negative'))

    dfb = dfb.assign(rw_rank = lambda x: x.rw_score.rank(**rank_kwargs, ascending=False))
    dfb = dfb.assign(pr_rank = lambda x: x.pr_score.rank(**rank_kwargs, ascending=False))
    dfb = dfb.assign(rwb_rank = lambda x: x.rwb_score.rank(**rank_kwargs, ascending=False))
    dfb = dfb.assign(bc_rank = lambda x: x.bc_score.rank(**rank_kwargs, ascending=False))

    dfb_involved = dfb[lambda x: x.node.isin(expected_nodes)][['node', 'gsnn_rank', 'ig_rank', 'oc_rank', 'rw_rank', 'pr_rank', 'rwb_rank', 'bc_rank']]

    mrr_gsnn = np.mean(1 / dfb_involved.gsnn_rank.values)
    mrr_ig = np.mean(1 / dfb_involved.ig_rank.values)
    mrr_oc = np.mean(1 / dfb_involved.oc_rank.values)
    mrr_rw = np.mean(1 / dfb_involved.rw_rank.values)
    mrr_pr = np.mean(1 / dfb_involved.pr_rank.values)
    mrr_rwb = np.mean(1 / dfb_involved.rwb_rank.values)
    mrr_bc = np.mean(1 / dfb_involved.bc_rank.values)

    mrr_df_node = pd.DataFrame({'gsnn': [mrr_gsnn], 'ig': [mrr_ig], 'oc': [mrr_oc], 'rw': [mrr_rw], 'pr': [mrr_pr], 'rwb': [mrr_rwb], 'bc': [mrr_bc]})
    return mrr_df_node, dfb_involved

################################################################################################################
################################################################################################################
################################################################################################################

def primary_regulator_comparison(edge_df: pd.DataFrame, target_node: str, expected_regulator: str, expected_direction = 'negative', plot = False, save_dir = None) -> dict:
    """Rank an expected regulator among all incoming edges to a target node.

    Compares GSNN, integrated-gradient (IG), and occlusion (OC) edge-importance
    scores for edges whose ``target`` equals ``target_node``. Each score method
    is ranked separately across regulators. GSNN always uses descending rank
    (higher score is better). IG and OC use ascending rank when
    ``expected_direction`` is ``'negative'`` and descending rank otherwise.

    Parameters
    ----------
    edge_df : pandas.DataFrame
        Edge-level scores with ``source`` and ``target`` columns (or a
        MultiIndex named ``source`` and ``target``). Must contain
        ``mean_gsnn_score``, ``mean_ig_score``, and ``mean_oc_score``.
    target_node : str
        Target node name (e.g. ``'RNA__DUSP6'``) whose incoming regulators are
        ranked.
    expected_regulator : str
        Source node (e.g. ``'PROTEIN__ETS1'``) whose rank is reported.
    expected_direction : {'negative', 'positive'}, default 'negative'
        Expected sign of regulation. For IG and OC, ``'negative'`` uses
        ascending rank (more negative score is better) and ``'positive'`` uses
        descending rank (more positive score is better).
    plot : bool, default False
        If True, draw a grouped bar chart of regulator scores by method.
    save_dir : str or None, default None
        Directory for output when ``plot=True``. If set, saves a ``.png`` figure
        and a ``.json`` summary; otherwise calls ``plt.show()``.

    Returns
    -------
    dict
        Keys: ``gsnn_rank``, ``ig_rank``, ``oc_rank`` (1-based ranks of
        ``expected_regulator``), ``num_regulators`` (number of unique sources
        targeting ``target_node``), plus echoed ``target_node``,
        ``expected_regulator``, and ``expected_direction``.
    """
    # if index is source, target, then nothing 
    if edge_df.index.names != ['source', 'target']:
        edge_df = edge_df.set_index(['source', 'target']) 

    cres_long = edge_df[['mean_gsnn_score', 'mean_ig_score', 'mean_oc_score']].stack().reset_index().rename({'level_2':'score_method', 0:'score'}, axis=1)
    reg_edges = cres_long[lambda x: x.target == target_node]
    reg_edges = reg_edges.copy() 

    reg_edges['rank_ascending'] = reg_edges.groupby('score_method')['score'].rank(method='max', ascending=True).astype(int) # lower score is better
    reg_edges['rank_descending'] = reg_edges.groupby('score_method')['score'].rank(method='max', ascending=False).astype(int) # higher score is better

    num_regulators = reg_edges.source.nunique()

    out = {'gsnn_rank': reg_edges[lambda x: x.score_method == 'mean_gsnn_score'][lambda x: x.source == expected_regulator]['rank_descending'].values[0], # higher score is always better
           'ig_rank': reg_edges[lambda x: x.score_method == 'mean_ig_score'][lambda x: x.source == expected_regulator]['rank_ascending' if expected_direction == 'negative' else 'rank_descending'].values[0],
           'oc_rank': reg_edges[lambda x: x.score_method == 'mean_oc_score'][lambda x: x.source == expected_regulator]['rank_ascending' if expected_direction == 'negative' else 'rank_descending'].values[0], 
           'num_regulators': num_regulators, 
           'target_node': target_node,
           'expected_regulator': expected_regulator,
           'expected_direction': expected_direction}

    if plot:

        ig_oc_normalized = reg_edges.groupby('score_method')['score'].transform(
            lambda x: x / (x.std() + 1e-6) 
        )
        # gsnn scores are already between 0 and 1, so keep the raw score for plotting
        reg_edges['normalized_score'] = reg_edges['score'].where(
            reg_edges['score_method'] == 'mean_gsnn_score',
            ig_oc_normalized,
        )

        plt.figure(figsize=(3*num_regulators, 4))
        sbn.barplot(data=reg_edges, x='source', y='normalized_score', hue='score_method', palette='Set1')
        plt.ylabel('Normalized score')
        plt.xlabel(f'{target_node} regulators')
        plt.title(f'Primary Regulator Comparison for {target_node} and {expected_regulator}')
        plt.legend(title='Score Method')
        plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/primary_regulator_comparison_{target_node}_{expected_regulator}_{expected_direction}.png')
            plt.close()

            with open(f'{save_dir}/primary_regulator_comparison_{target_node}_{expected_regulator}_{expected_direction}.json', 'w') as f:
                json.dump(out, f)

        else: 
            plt.show() 

    return out