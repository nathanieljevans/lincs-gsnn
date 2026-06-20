"""
Edge inference evaluation for the GSNN link prediction task.

Candidate edges are restricted to non-isolate, non-self PROTEIN -> RNA pairs.
Each edge is classified into one of four mutually exclusive groups: train,
test (held-out during training), test_tft (present in OmniPath Transcriptional
but absent from train/test), or negative (none of the above).

Ranking uses *filtered* *target* ranking: for every edge, its score is compared only
to negatives sharing the same target node. Metrics reported per group include
filtered MRR, top-k accuracy (k=1, 10, 100), AUROC, and AUPR. AUROC/AUPR
are computed against the negative set for each positive group.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
import omnipath as op

from lincs_gsnn.proc.graph import (
    gene_symbol_from_node,
    map_function_node,
    protein_to_rna_edge_mask,
)


def eval_edge_inference(res, data, metric, function_node_map=None):
    '''Evaluate edge inference across train, test, test_tft, and test_tft_isolate subsets.

    Filters ``res`` to non-self PROTEIN -> RNA edges, identifies isolate targets
    (zero in/out-degree in the training graph), and pulls additional TF-target
    edges from OmniPath Transcriptional as the ``test_tft`` set.  Edges in
    ``test_tft`` on isolate targets form the ``test_tft_isolate`` subset.

    AUROC/AUPR for train, test, and test_tft are computed on non-isolate targets
    only.  test_tft_isolate AUROC/AUPR are computed on isolate targets only.
    Rank-based metrics (MRR, top-k) are computed on all targets for every group.

    Parameters
    ----------
    res : pd.DataFrame
        Candidate edges with columns: source, target, train_edge, test_edge,
        and the score column named by ``metric``.  May optionally contain
        val_edge / valid_edge to exclude validation edges from negatives.
    data : Data
        Graph data object with ``node_names_dict``, ``edge_index_dict``.
    metric : str
        Name of the numeric score column in ``res`` (higher = more likely edge).
    function_node_map : dict, optional
        ``old_name -> new_name`` from graph simplification (``data.function_node_map``).
        When omitted, uses ``getattr(data, 'function_node_map', None)``.

    Returns
    -------
    dict[str, dict]
        Nested dict keyed by subset (train, test, test_tft, test_tft_isolate,
        neg) each containing auroc, aupr, mrr, top1_acc, top10_acc,
        top100_acc, and n (number of edges in that subset).
    '''

    if function_node_map is None:
        function_node_map = getattr(data, "function_node_map", None)

    ################### Filtering ###################

    # The test set nodes were sampled specifically from nodes that had more than 2 edges 
    # As is, the model performance (model_r and model_r2) will be much lower in isolate targets than non-isolate targets.
    # isolates vs not isolates and will potentially bias the performance metrics 
    # to be safe, we should only consider edges that are not isolates 
    node_names = np.array(data.node_names_dict['function']) 
    src_names = node_names[data.edge_index_dict['function', 'to', 'function'][0]]
    dst_names = node_names[data.edge_index_dict['function', 'to', 'function'][1]] 
    train_edges = pd.DataFrame({'source': src_names, 'target': dst_names})

    G = nx.from_pandas_edgelist(train_edges[['source', 'target']], source='source', target='target', create_using=nx.DiGraph())
    for g in data.node_names_dict['output']:
        rna_node = map_function_node('RNA__' + g.split('__', 1)[1], function_node_map)
        G.add_node(rna_node)
    isolates = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]

    # NOT REMOVING - BC now we can evaluate with tft edges
    # TRAIN, TEST, TEST_TFT will not be evaluated on non-isolate targets only 
    # TEST_TFT_ISOLATE will be evaluated on isolate targets ONLY 
    #res = res[~res['target'].isin(isolates)]

    # let's assign which targets are isolates and use tft to evaluate performance on these edges 
    res = res.assign(isolate_target = lambda x: x.target.isin(isolates))



    ########################################################### 

    # Protein --> RNA edges only (legacy prefixes or simplified supernode names).
    res = res[protein_to_rna_edge_mask(res, function_node_map=function_node_map)]

    ########################################################### 

    # drop self edges 
    res = res.assign(source_genesymbol=lambda x: x.source.map(gene_symbol_from_node))
    res = res.assign(target_genesymbol=lambda x: x.target.map(gene_symbol_from_node))
    res = res[lambda x: x.source_genesymbol != x.target_genesymbol]

    ########################## Omnipath Extra TF Targets ########################## 
    # larger set of low quality TF target edges (OEI uses dorothea levels A and B)
    # this dataset uses Dorothea ABCD, CollecTRI and TF_mrna datasets
    # See: https://omnipath.readthedocs.io/en/latest/api/omnipath.interactions.Transcriptional.html#omnipath.interactions.Transcriptional 


    tft = op.interactions.Transcriptional().get(genesymbol=True)[['source_genesymbol', 'target_genesymbol']]
    tft = tft.assign(
        source=lambda df: df['source_genesymbol'].map(
            lambda g: map_function_node(f'PROTEIN__{g}', function_node_map)
        ),
        target=lambda df: df['target_genesymbol'].map(
            lambda g: map_function_node(f'RNA__{g}', function_node_map)
        ),
    )
    tft = tft[['source', 'target']]
    tft = tft.drop_duplicates()
    tft = tft.assign(in_tft = True)  

    res = res.merge(tft, on=['source', 'target'], how='left')
    res['in_tft'] = res['in_tft'].fillna(False).astype(bool)

    if ('val_edge' in res.columns):
        res = res.assign(test_tft = lambda x: ~(x.test_edge | x.train_edge | x.val_edge) & x.in_tft)
    elif ('valid_edge' in res.columns):
        res = res.assign(test_tft = lambda x: ~(x.test_edge | x.train_edge | x.valid_edge) & x.in_tft)
    else: 
        res = res.assign(test_tft = lambda x: ~(x.test_edge | x.train_edge) & x.in_tft)

    ##############################################################

    # assign negative edges
    if ('val_edge' in res.columns):
        res = res.assign(negative_edge = lambda x: ~(x.train_edge | x.test_edge | x.val_edge | x.test_tft))
    elif ('valid_edge' in res.columns):
        res = res.assign(negative_edge = lambda x: ~(x.train_edge | x.test_edge | x.valid_edge | x.test_tft))
    else:
        res = res.assign(negative_edge = lambda x: ~(x.train_edge | x.test_edge | x.test_tft))

    ###########################################################  

    # assign ranks 
    ranks, max_ranks = filtered_rank(res, metric)
    res = res.assign(rank = ranks, max_rank = max_ranks)

    ###########################################################  

    # calculate MRR 
    mrr_train = np.mean(1 / res[lambda x: x.train_edge]['rank'].values)
    mrr_test = np.mean(1 / res[lambda x: x.test_edge]['rank'].values)
    mrr_test_tft = np.mean(1 / res[lambda x: x.test_tft]['rank'].values)
    mrr_test_tft_isolate = np.mean(1 / res[lambda x: x.test_tft & x.isolate_target]['rank'].values)
    mrr_neg = np.mean(1 / res[lambda x: x.negative_edge]['rank'].values)
    mrr_test_has_expr = np.mean(1 / res[lambda x: x.test_edge & x.has_expr]['rank'].values)

    ###########################################################  

    # calculate top 1 accuracy 
    top1_acc_train = np.mean(res[lambda x: x.train_edge]['rank'] <= 1)
    top1_acc_test = np.mean(res[lambda x: x.test_edge]['rank'] <= 1)
    top1_acc_test_tft = np.mean(res[lambda x: x.test_tft]['rank'] <= 1)
    top1_acc_test_tft_isolate = np.mean(res[lambda x: x.test_tft & x.isolate_target]['rank'] <= 1)
    top1_acc_neg = np.mean(res[lambda x: x.negative_edge]['rank'] <= 1)
    top1_acc_test_has_expr = np.mean(res[lambda x: x.test_edge & x.has_expr]['rank'] <= 1)

    ###########################################################  

    # calculate top 10 accuracy 
    top10_acc_train = np.mean(res[lambda x: x.train_edge]['rank'] <= 10)
    top10_acc_test = np.mean(res[lambda x: x.test_edge]['rank'] <= 10)
    top10_acc_test_tft = np.mean(res[lambda x: x.test_tft]['rank'] <= 10)
    top10_acc_test_tft_isolate = np.mean(res[lambda x: x.test_tft & x.isolate_target]['rank'] <= 10)
    top10_acc_neg = np.mean(res[lambda x: x.negative_edge]['rank'] <= 10)
    top10_acc_test_has_expr = np.mean(res[lambda x: x.test_edge & x.has_expr]['rank'] <= 10)

    ###########################################################  
    
    # calculate top 100 accuracy 
    top100_acc_train = np.mean(res[lambda x: x.train_edge]['rank'] <= 100)
    top100_acc_test = np.mean(res[lambda x: x.test_edge]['rank'] <= 100)
    top100_acc_test_tft = np.mean(res[lambda x: x.test_tft]['rank'] <= 100)
    top100_acc_test_tft_isolate = np.mean(res[lambda x: x.test_tft & x.isolate_target]['rank'] <= 100)
    top100_acc_neg = np.mean(res[lambda x: x.negative_edge]['rank'] <= 100)
    top100_acc_test_has_expr = np.mean(res[lambda x: x.test_edge & x.has_expr]['rank'] <= 100)

    ###########################################################  
    
    
    # calculate AUROC 
    # NOTE: TRAIN, TEST, TEST_TFT will not be evaluated on isolate targets
    train_res = res[lambda x: (x.train_edge | x.negative_edge) & ~x.isolate_target]
    train_auroc = roc_auc_score(train_res.train_edge.values.astype(int), train_res[metric].values.astype(float))

    test_res = res[lambda x: (x.test_edge | x.negative_edge) & ~x.isolate_target]
    test_auroc = roc_auc_score(test_res.test_edge.values.astype(int), test_res[metric].values.astype(float))

    test_tft_res = res[lambda x: (x.test_tft | x.negative_edge) & ~x.isolate_target]
    if test_tft_res.test_tft.sum() > 0:
        test_tft_auroc = roc_auc_score(test_tft_res.test_tft.values.astype(int), test_tft_res[metric].values.astype(float))
    else:
        test_tft_auroc = np.nan

    # NOTE: TEST_TFT_ISOLATE will be evaluated on isolate targets ONLY 
    test_tft_isolate_res = res[lambda x: (x.test_tft | x.negative_edge) & x.isolate_target]
    if test_tft_isolate_res.test_tft.sum() > 0:
        test_tft_isolate_auroc = roc_auc_score(test_tft_isolate_res.test_tft.values.astype(int), test_tft_isolate_res[metric].values.astype(float))
    else:
        test_tft_isolate_auroc = np.nan

    test_has_expr_res = res[lambda x: (x.test_edge | x.negative_edge) & x.has_expr]
    if test_has_expr_res.test_edge.sum() > 0:
        test_has_expr_auroc = roc_auc_score(test_has_expr_res.test_edge.values.astype(int), test_has_expr_res[metric].values.astype(float))
    else:
        test_has_expr_auroc = np.nan

    ###########################################################   

    # calculate AUPR  

    train_aupr = average_precision_score(train_res.train_edge.values.astype(int), train_res[metric].values.astype(float))
    test_aupr = average_precision_score(test_res.test_edge.values.astype(int), test_res[metric].values.astype(float))
    if test_tft_res.test_tft.sum() > 0:
        test_tft_aupr = average_precision_score(test_tft_res.test_tft.values.astype(int), test_tft_res[metric].values.astype(float))
    else:
        test_tft_aupr = np.nan

    if test_tft_isolate_res.test_tft.sum() > 0:
        test_tft_isolate_aupr = average_precision_score(test_tft_isolate_res.test_tft.values.astype(int), test_tft_isolate_res[metric].values.astype(float))
    else:
        test_tft_isolate_aupr = np.nan
    
    if test_has_expr_res.test_edge.sum() > 0:
        test_has_expr_aupr = average_precision_score(test_has_expr_res.test_edge.values.astype(int), test_has_expr_res[metric].values.astype(float))
    else:
        test_has_expr_aupr = np.nan

    ###########################################################  

    n_test = res.test_edge.sum()
    n_test_tft = res.test_tft.sum()
    n_train = res.train_edge.sum()
    n_neg = res.negative_edge.sum()
    n_test_tft_isolate = (res.test_tft & res.isolate_target).sum()
    n_test_has_expr = (res.test_edge & res.has_expr).sum()

    ###########################################################  

    return {'train': {'auroc': train_auroc, 'aupr': train_aupr, 'mrr': mrr_train, 'top1_acc': top1_acc_train, 'top10_acc': top10_acc_train, 'top100_acc': top100_acc_train, 'n': n_train},
            'test': {'auroc': test_auroc, 'aupr': test_aupr, 'mrr': mrr_test, 'top1_acc': top1_acc_test, 'top10_acc': top10_acc_test, 'top100_acc': top100_acc_test, 'n': n_test},
            'test_tft': {'auroc': test_tft_auroc, 'aupr': test_tft_aupr, 'mrr': mrr_test_tft, 'top1_acc': top1_acc_test_tft, 'top10_acc': top10_acc_test_tft, 'top100_acc': top100_acc_test_tft, 'n': n_test_tft},
            'test_tft_isolate': {'auroc': test_tft_isolate_auroc, 'aupr': test_tft_isolate_aupr, 'mrr': mrr_test_tft_isolate, 'top1_acc': top1_acc_test_tft_isolate, 'top10_acc': top10_acc_test_tft_isolate, 'top100_acc': top100_acc_test_tft_isolate, 'n': n_test_tft_isolate},
            'test_has_expr': {'auroc': test_has_expr_auroc, 'aupr': test_has_expr_aupr, 'mrr': mrr_test_has_expr, 'top1_acc': top1_acc_test_has_expr, 'top10_acc': top10_acc_test_has_expr, 'top100_acc': top100_acc_test_has_expr, 'n': n_test_has_expr},
            'neg': {'auroc': None, 'aupr': None, 'mrr': mrr_neg, 'top1_acc': top1_acc_neg, 'top10_acc': top10_acc_neg, 'top100_acc': top100_acc_neg, 'n': n_neg}}
    
    
    
    
    




def filtered_rank(res, metric, neg_col='negative_edge'):
    """
    Filtered ranking within each ``target`` (link prediction evaluation).

    For each row, compare its ``metric`` score only to edges flagged as
    negatives via ``neg_col``.  Positive edges (train, test, dorothea, etc.)
    are excluded from the reference pool so they are not ranked against
    each other.

    For a row with score ``v``, the returned count is

        |{ negative edges in target : v <= neg_score }|

    i.e. the number of negatives whose score is at least ``v`` (larger ``metric``
    means better).

    Parameters
    ----------
    res : pd.DataFrame
        Must include columns ``target``, ``neg_col``, and the column named by
        ``metric``.
    metric : str
        Numeric score column.
    neg_col : str
        Boolean column identifying negative edges (default ``'negative_edge'``).

    Returns
    -------
    ranks : np.ndarray
        Integer counts, length ``len(res)``, order aligned with ``res.index``.
    max_ranks : np.ndarray
        Number of negative edges in that row's ``target`` (same length).
    """
    ranks = pd.Series(index=res.index, dtype=np.int64)
    max_ranks = pd.Series(index=res.index, dtype=np.int64)

    for _, g in res.groupby("target", sort=False):
        neg_mask = g[neg_col].fillna(False).astype(bool)
        neg_vals = g.loc[neg_mask, metric].to_numpy(dtype=np.float64, copy=False)
        m = neg_vals.size
        neg_sorted = np.sort(neg_vals) if m else neg_vals

        v = g[metric].to_numpy(dtype=np.float64, copy=False)
        max_ranks.loc[g.index] = m

        if m == 0:
            ranks.loc[g.index] = 0
            continue

        local_ranks = np.full(len(g), m, dtype=np.int64)
        ok = ~np.isnan(v)
        if ok.any():
            pos = np.searchsorted(neg_sorted, v[ok], side="left")
            local_ranks[ok] = m - pos
        ranks.loc[g.index] = local_ranks

    return ranks.to_numpy() + 1, max_ranks.to_numpy() + 1


def _subset_metrics(res, metric, pos_col, neg_col='negative_edge'):
    '''Compute rank and AUROC metrics for one positive subset vs negatives.'''
    pos_mask = res[pos_col].fillna(False).astype(bool)
    n_pos = int(pos_mask.sum())

    out = {
        'n': n_pos,
        'auroc': None,
        'aupr': None,
        'mrr': None,
        'top1_acc': None,
        'top10_acc': None,
        'top100_acc': None,
    }
    if n_pos == 0:
        return out

    ranks, _ = filtered_rank(res, metric, neg_col=neg_col)
    res = res.assign(rank=ranks)
    pos_ranks = res.loc[pos_mask, 'rank'].astype(float)

    out['mrr'] = float(np.mean(1.0 / pos_ranks))
    out['top1_acc'] = float(np.mean(pos_ranks <= 1))
    out['top10_acc'] = float(np.mean(pos_ranks <= 10))
    out['top100_acc'] = float(np.mean(pos_ranks <= 100))

    eval_res = res[(pos_mask | res[neg_col].fillna(False).astype(bool))].copy()
    if eval_res[neg_col].sum() > 0 and eval_res[pos_col].sum() > 0:
        y = eval_res[pos_col].fillna(False).astype(int).values
        scores = eval_res[metric].astype(float).values
        if len(np.unique(y)) > 1:
            out['auroc'] = float(roc_auc_score(y, scores))
            out['aupr'] = float(average_precision_score(y, scores))
    return out


def eval_edge_inference_any(res, data, metric):
    '''Evaluate edge inference on any function->function edge type.

    Expects ``res`` with columns ``source``, ``target``, ``train_edge``,
    ``val_edge``, ``test_edge``, and the score column named by ``metric``.
    Optionally accepts ``full_edge`` for an additional positive partition.
    Negatives are edges that are not train, val, test, or full.

    Returns a nested dict with keys ``overall`` (train/val/test/full/neg) and
    ``by_edge_type`` (per src_type->dst_type breakdown).
    '''
    res = res.copy()
    has_full = 'full_edge' in res.columns
    bool_cols = ('train_edge', 'val_edge', 'test_edge')
    if has_full:
        bool_cols = bool_cols + ('full_edge',)
    for col in bool_cols:
        if col not in res.columns:
            res[col] = False
        res[col] = res[col].fillna(False).astype(bool)

    res = res.assign(
        source_type=lambda x: x.source.str.split('__', expand=True)[0],
        target_type=lambda x: x.target.str.split('__', expand=True)[0],
    )
    res = res.assign(
        source_genesymbol=lambda x: x.source.str.split('__', expand=True)[1],
        target_genesymbol=lambda x: x.target.str.split('__', expand=True)[1],
    )
    res = res[lambda x: x.source_genesymbol != x.target_genesymbol]

    if has_full:
        res = res.assign(
            negative_edge=lambda x: ~(x.train_edge | x.val_edge | x.test_edge | x.full_edge),
        )
    else:
        res = res.assign(
            negative_edge=lambda x: ~(x.train_edge | x.val_edge | x.test_edge),
        )

    overall = {
        'train': _subset_metrics(res, metric, 'train_edge'),
        'val': _subset_metrics(res, metric, 'val_edge'),
        'test': _subset_metrics(res, metric, 'test_edge'),
        'neg': _subset_metrics(res, metric, 'negative_edge', neg_col='negative_edge'),
    }
    if has_full:
        overall['full'] = _subset_metrics(res, metric, 'full_edge')

    by_edge_type = {}
    for (src_t, dst_t), g in res.groupby(['source_type', 'target_type'], sort=False):
        key = f'{src_t}->{dst_t}'
        by_edge_type[key] = {
            'train': _subset_metrics(g, metric, 'train_edge'),
            'val': _subset_metrics(g, metric, 'val_edge'),
            'test': _subset_metrics(g, metric, 'test_edge'),
            'neg': _subset_metrics(g, metric, 'negative_edge', neg_col='negative_edge'),
        }
        if has_full:
            by_edge_type[key]['full'] = _subset_metrics(g, metric, 'full_edge')

    return {'overall': overall, 'by_edge_type': by_edge_type}