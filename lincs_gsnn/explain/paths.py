"""Per-path aggregation of edge-level explainer / topology scores.

Given a directed bionetwork whose edges carry one or more numeric
scores (e.g. Contrastive GSNN inclusion probabilities, Integrated
Gradients, Occlusion, random-walk fluxes, betweenness centralities),
this module enumerates every simple source -> target path up to a
length cutoff and reduces the per-edge scores along each path to a
single per-path number using a chosen aggregation rule.

This is the building block for path-level explanation comparisons
(e.g. ranking known canonical paths against decoys via MRR).
"""

import networkx as nx
import numpy as np
import pandas as pd


def path_score(
    G: nx.DiGraph,
    source_node,
    target_node,
    cutoff: int = 5,
    score_cols=None,
    method: str = "product",
    signed: bool = False,
    rank_method: str = 'max',
    drop_source: bool = True,
    drop_target: bool = True,
    shorten: bool = True, 

) -> pd.DataFrame:
    """Aggregate per-edge scores along every simple ``s -> t`` path.

    Enumerates all simple paths from ``source_node`` to ``target_node``
    in ``G`` whose length (number of edges) is at most ``cutoff``,
    then for each edge attribute named in ``score_cols`` collapses the
    scores along the path using ``method``.

    Aggregation choices
    -------------------
    * ``'product'`` -- multiplicative path score, ``prod_e score_e``.
      For probability-like scores, equals the joint inclusion
      probability under an edge-independence assumption (rigorous for
      the contrastive-GSNN posterior, a ranking heuristic for the
      others). Heavily biased toward short paths because every factor
      lies in ``[0, 1]``.
    * ``'sum'``     -- additive path score, ``sum_e score_e``. The
      natural aggregation for additive attributions (IG / Occlusion),
      since those satisfy a per-input-feature completeness axiom.
      Biased toward long paths.
    * ``'mean'``    -- per-edge mean, ``sum_e score_e / n_edges``. A
      length-normalised version of ``'sum'``: lets you compare paths
      of different lengths on a common per-edge scale.

    Sign handling
    -------------
    By default (``signed=False``) every edge score is passed through
    ``np.abs`` before aggregation. This is the right behaviour for
    signed attributions (IG, OC) where two large-magnitude *negative*
    edges would otherwise multiply to a large *positive* product or
    cancel additively. Set ``signed=True`` for scores that are
    intrinsically non-negative (probabilities in ``[0, 1]``,
    centralities, fluxes) -- keeping the raw values means an
    accidental negative will surface rather than being masked.

    Caveats
    -------
    * The product of probability-like scores is only a true joint
      probability under an edge-independence assumption; for
      non-probability scores (RW / PR / RWB / BC fluxes, IG, OC) the
      product is a ranking heuristic, not a probability.
    * ``'product'`` with edge scores in ``[0, 1]`` shrinks
      geometrically with path length, so longer paths are
      systematically downweighted relative to shorter ones. Use
      ``'mean'`` to factor path length out of the comparison.
    * ``nx.all_simple_paths`` enumerates every simple path up to
      ``cutoff``; the count grows combinatorially with ``cutoff`` and
      the graph's branching factor, so keep ``cutoff`` modest
      (typically ``<= 6``) on dense bionetworks.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph with per-edge numeric attributes.
    source_node, target_node : node id
        Endpoints of the paths to enumerate. Must both exist in ``G``.
    cutoff : int, default 5
        Maximum path length (in *edges*, not nodes) passed through to
        ``nx.all_simple_paths``.
    score_cols : list[str], optional
        Names of edge attributes to aggregate. If ``None``, every
        attribute present on the first edge of ``G`` is used.
    method : {'product', 'sum', 'mean'}, default 'product'
        Aggregation rule along the path. See "Aggregation choices"
        above.
    signed : bool, default False
        If ``False`` (the default), ``np.abs`` is applied to each edge
        score before aggregation. If ``True``, raw signed values are
        used.
    rank_method : {'max', 'min', 'first', 'last', 'dense', 'average', 'ordinal'}, default 'max'
        Method to use for ranking path scores. See ``pandas.Series.rank`` for details.

    Returns
    -------
    pd.DataFrame
        One row per simple path, with columns:

        ``path``
            List of node ids forming the path (length = ``path_length``).
        ``path_length``
            Number of *nodes* on the path (so it has
            ``path_length - 1`` edges).
        ``path_short``
            ``' -> '.join(path[1:-1])`` -- the intermediate nodes only,
            for compact display in tables/plots.
        ``{col}_{method}`` (one per ``score_cols`` entry)
            Aggregated path score under ``method``.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``{'product', 'sum', 'mean'}``,
        or if ``score_cols`` cannot be inferred from ``G``.
    """
    if method not in {"product", "sum", "mean"}:
        raise ValueError(
            f"Unknown method {method!r}. Must be one of "
            "'product', 'sum', 'mean'."
        )

    # Infer score columns from the first edge if the caller didn't
    # supply them. Only fall back to inference when score_cols is
    # actually None -- explicit empty lists are a caller bug, not an
    # implicit "all-columns" request.
    if score_cols is None:
        first_attrs = next(
            (attrs for _, _, attrs in G.edges(data=True)), None
        )
        if not first_attrs:
            raise ValueError(
                "score_cols=None but G has no edges with attributes; "
                "cannot infer which columns to aggregate."
            )
        score_cols = list(first_attrs.keys())

    all_paths = list(
        nx.all_simple_paths(
            G, source=source_node, target=target_node, cutoff=cutoff,
        )
    )

    init_val = 1.0 if method == "product" else 0.0
    records = []
    for p in all_paths:
        n_edges = len(p) - 1
        agg = {col: init_val for col in score_cols}
        for i in range(n_edges):
            attrs = G[p[i]][p[i + 1]]
            for col in score_cols:
                v = attrs[col]
                if not signed:
                    v = abs(v)
                if method == "product":
                    agg[col] *= v
                else:  # 'sum' or 'mean' -- accumulate then maybe normalise
                    agg[col] += v

        if method == "mean" and n_edges > 0:
            agg = {col: v / n_edges for col, v in agg.items()}

        plen = len(p) 

        if drop_source:
            p = p[1:]
        if drop_target:
            p = p[:-1]

        if shorten:
            p = [x.split('__')[1] for x in p]

        records.append(
            {
                "path": p,
                "path_length": plen,
                **{f"{col}_{method}": agg[col] for col in score_cols},
            }
        )

    path_df = pd.DataFrame(records)
    if not path_df.empty:
        path_df = path_df.assign(
            path_short=[" -> ".join(p) for p in path_df["path"]]
        )
    else:
        path_df["path_short"] = pd.Series(dtype=str)

    # add ranks by path score 
    for score_col in score_cols:
        path_df = path_df.assign(**{f'{score_col}_rank': lambda x: x[f'{score_col}_{method}'].rank(method=rank_method, ascending=False)})
    
    return path_df
