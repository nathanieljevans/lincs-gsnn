"""Topology-based adjustment of GSNN node and edge importance scores.

Explainers (Contrastive GSNN, Integrated Gradients, Occlusion, ...)
score every node / edge in the bionetwork. Some of those scores are
high simply because the node or edge sits on many short paths between
the perturbation (drug) and the read-out (target gene) -- not because
the model has learned anything specific about it. This module computes
the score that would be expected from network *topology alone* and
subtracts it, leaving a residual that isolates the model's *learned*
preference.

Two topology metrics are provided:

    * ``pagerank``    -- (personalized) PageRank stationary mass.
    * ``random_walk`` -- Monte-Carlo expected node visits / edge
      traversals from short walks.

Both are *flux* measures: for nodes they are expected occupancies,
for edges they are expected traversals (forward) and the corresponding
reverse-graph traversals (backward, re-keyed to the original
orientation). Unlike a pure reachability indicator, a flux value can
exceed 1 for a node or edge that is visited multiple times in a single
walk.

The composite source -> target score is the product of forward and
backward flux through the node/edge (an estimate of total flow under
a random walker that starts at the source and ends at the target), and
the adjusted score is interpretable as:

    topology_adjusted_score = observed_score - expected_score_from_topology

    > 0  : the model values this node/edge MORE than its topological
           position would predict (learned preference).
    ~ 0  : the model uses this node/edge about as much as topology
           suggests it should (uninformative).
    < 0  : the model down-weights this node/edge despite easy
           reachability.
"""

import random
from collections import Counter, deque
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

__N_MC__ = 100_000


TopologyFeatures = Tuple[np.ndarray, Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]
"""Return type for the standalone edge-level topology helpers.

Tuple of:
  * ``X_edge`` -- ``(n_edges, 2)`` per-edge features
    ``[log_edge_flux_src(u, v), log_edge_flux_tgt(u, v)]``.
  * ``log_edge_flux_src`` -- dict ``(u, v) -> log( edge flux of u->v
    under the forward walk anchored at the source )``.
  * ``log_edge_flux_tgt`` -- dict ``(u, v) -> log( edge flux of u->v
    under the reverse walk anchored at the target, i.e. the reverse
    walk on G^T traversing (v, u) re-keyed back to original
    orientation )``.

For ``random_walk``, "edge flux" is the expected number of traversals
of the edge per walk. For ``pagerank``, it is ``pi(u) * P(u->v)`` (the
mass that flows along the edge under the personalized-PR stationary
distribution), which for an unweighted random walk reduces to
``pi_src(u) / out_deg(u)`` on the forward side and
``pi_tgt(v) / in_deg(v)`` on the reverse side.
"""


def _as_edge_frame(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    """Return a flat dataframe with `source_col`/`target_col` as columns.

    Accepts either a frame where source/target are columns, or one where
    they live in a (Multi)Index (as in `cres_agg` from the explainer
    notebooks).
    """
    if source_col in df.columns and target_col in df.columns:
        return df
    if isinstance(df.index, pd.MultiIndex) and {source_col, target_col}.issubset(df.index.names):
        return df.reset_index()
    raise ValueError(
        f"DataFrame must contain '{source_col}' and '{target_col}' as columns "
        f"or as MultiIndex levels."
    )


def _safe_log_with_floor(
    values: np.ndarray, eps: float, nan_for_zeros: bool = False
) -> np.ndarray:
    """Element-wise log with handling for non-positive entries.

    If ``nan_for_zeros`` is True, zero/negative entries are returned as
    ``NaN`` so the caller can filter them out (this matches the
    "drop-unreachable-nodes" behaviour used in the original explainer
    notebook). Otherwise they are replaced with
    ``max(eps, min(positive values) / 100)`` so all returned values are
    finite and just below the smallest reachable value.
    """
    if nan_for_zeros:
        positive_floor = max(eps, 0.0)
        return np.where(
            values > 0, np.log(np.maximum(values, positive_floor)), np.nan
        )
    positive = values[values > 0]
    if positive.size:
        floor = max(eps, float(positive.min()) / 100.0)
    else:
        floor = eps
    return np.log(np.where(values > 0, values, floor))


def _log_dict_with_floor(
    values: Dict[str, float], eps: float, nan_for_zeros: bool = False
) -> Dict[str, float]:
    """Same floor logic as :func:`_safe_log_with_floor`, but for a dict."""
    arr = np.fromiter(values.values(), dtype=float, count=len(values))
    log_arr = _safe_log_with_floor(arr, eps, nan_for_zeros=nan_for_zeros)
    return dict(zip(values.keys(), log_arr))


def _validate_anchors(G: nx.DiGraph, source_node: Optional[str], target_node: Optional[str]) -> None:
    if source_node is not None and source_node not in G:
        raise ValueError(f"source_node {source_node!r} is not present in the graph.")
    if target_node is not None and target_node not in G:
        raise ValueError(f"target_node {target_node!r} is not present in the graph.")


def pagerank_node_features(
    G: nx.DiGraph,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    damping: float = 0.85,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-node flux features from (personalized) PageRank.

    Computes two dicts (one per node in ``G``):

        log_pi_src(n) = log( pi_src(n) )
        log_pi_tgt(n) = log( pi_tgt(n) )

    where ``pi_src`` is PageRank on the forward graph and ``pi_tgt`` is
    PageRank on the *reversed* graph. When ``source_node`` and/or
    ``target_node`` are provided, the corresponding PageRank is
    *personalized* on that node, i.e. it is the stationary distribution
    of a random walk that restarts at the anchor with probability
    ``1 - damping``. Both ``pi_src`` and ``pi_tgt`` are flux measures
    (per-step occupancy probabilities), so they double as
    drop-in replacements for the random-walk node fluxes computed by
    :func:`random_walk_node_features`.

    The two features answer different questions:
      * ``log_pi_src(n)`` -- how easy it is to reach ``n`` from the source.
      * ``log_pi_tgt(n)`` -- how easy it is to reach the target from ``n``.

    Entries with ``pi = 0`` (nodes unreachable from the anchor under
    personalized PageRank) get a finite log-floor just below the
    smallest reachable value, so they remain comparable in the
    regression without dominating the fit.

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as the transition
        weight in PageRank (passed through as ``weight=edge_weight`` to
        :func:`networkx.pagerank`). When ``None`` (the default) the
        random walk is uniform over outgoing neighbours, matching the
        previous behaviour. When set, transitions out of each node are
        proportional to that attribute on both the forward graph and
        its reverse.
    """
    _validate_anchors(G, source_node, target_node)
    src_pers = {source_node: 1.0} if source_node is not None else None
    tgt_pers = {target_node: 1.0} if target_node is not None else None
    pi_src = nx.pagerank(
        G, alpha=damping, personalization=src_pers, weight=edge_weight,
    )
    pi_tgt = nx.pagerank(
        G.reverse(copy=False), alpha=damping, personalization=tgt_pers,
        weight=edge_weight,
    )
    return (
        _log_dict_with_floor(pi_src, eps, nan_for_zeros=nan_for_zeros),
        _log_dict_with_floor(pi_tgt, eps, nan_for_zeros=nan_for_zeros),
    )


def pagerank_edge_features(
    G: nx.DiGraph,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    damping: float = 0.85,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Per-edge flux features from (personalized) PageRank.

    Computes two dicts (one per directed edge ``(u, v)`` in ``G``):

        log_edge_flux_src(u, v) = log( pi_src(u) * P(u -> v) )
        log_edge_flux_tgt(u, v) = log( pi_tgt(v) * P_rev(v -> u) )

    These are the PageRank analogues of the random-walk edge fluxes:
    under the random-walk model whose stationary distribution is
    personalized PageRank, the mass that flows along edge ``(u, v)``
    per step is ``pi(u) * P(u -> v)`` in the forward walk, and
    analogously ``pi_tgt(v) * P_rev(v -> u)`` for the reverse walk on
    ``G^T``.

    When ``edge_weight`` is ``None`` (the default), transitions are
    uniform over outgoing neighbours, so
    ``P(u -> v) = 1 / out_deg(u)`` and
    ``P_rev(v -> u) = 1 / in_deg(v)``. When ``edge_weight`` names an
    edge attribute, transitions are proportional to that attribute,
    so ``P(u -> v) = w(u, v) / W_out(u)`` (with
    ``W_out(u) = sum_j w(u, j)``) and
    ``P_rev(v -> u) = w(u, v) / W_in(v)`` (with
    ``W_in(v) = sum_j w(j, v)``), matching the transition kernel that
    weighted ``nx.pagerank`` uses internally.

    Edges incident to a node with (weighted) degree zero in the
    relevant direction have flux ``0``; with ``nan_for_zeros=True``
    they get ``NaN`` (so the OLS fit in :func:`adjust_for_topology`
    filters them out), otherwise they are floored to just below the
    smallest observed positive flux.

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as the transition
        weight (forwarded as ``weight=edge_weight`` to
        :func:`networkx.pagerank`, and used in the per-edge flux
        formula). When ``None`` the unweighted/uniform behaviour is
        recovered.
    """
    _validate_anchors(G, source_node, target_node)
    src_pers = {source_node: 1.0} if source_node is not None else None
    tgt_pers = {target_node: 1.0} if target_node is not None else None
    pi_src = nx.pagerank(
        G, alpha=damping, personalization=src_pers, weight=edge_weight,
    )
    pi_tgt = nx.pagerank(
        G.reverse(copy=False), alpha=damping, personalization=tgt_pers,
        weight=edge_weight,
    )

    edge_flux_src: Dict[Tuple[str, str], float] = {}
    edge_flux_tgt: Dict[Tuple[str, str], float] = {}

    if edge_weight is None:
        out_deg = dict(G.out_degree())
        in_deg = dict(G.in_degree())
        for u, v in G.edges():
            d_out = out_deg.get(u, 0)
            d_in = in_deg.get(v, 0)
            edge_flux_src[(u, v)] = (pi_src.get(u, 0.0) / d_out) if d_out > 0 else 0.0
            edge_flux_tgt[(u, v)] = (pi_tgt.get(v, 0.0) / d_in) if d_in > 0 else 0.0
    else:
        # Weighted transition kernel: P(u -> v) = w(u, v) / W_out(u),
        # P_rev(v -> u) = w(u, v) / W_in(v). Use clamped-non-negative
        # weights to match `_random_walk_flux` and to avoid divide-by-
        # zero when a row of weights sums to <= 0.
        w_out: Dict[str, float] = {}
        w_in: Dict[str, float] = {}
        for u, v, data in G.edges(data=True):
            w = max(float(data.get(edge_weight, 1.0)), 0.0)
            w_out[u] = w_out.get(u, 0.0) + w
            w_in[v] = w_in.get(v, 0.0) + w
        for u, v, data in G.edges(data=True):
            w = max(float(data.get(edge_weight, 1.0)), 0.0)
            wo = w_out.get(u, 0.0)
            wi = w_in.get(v, 0.0)
            edge_flux_src[(u, v)] = (
                pi_src.get(u, 0.0) * w / wo if wo > 0 else 0.0
            )
            edge_flux_tgt[(u, v)] = (
                pi_tgt.get(v, 0.0) * w / wi if wi > 0 else 0.0
            )

    return (
        _log_dict_with_floor(edge_flux_src, eps, nan_for_zeros=nan_for_zeros),
        _log_dict_with_floor(edge_flux_tgt, eps, nan_for_zeros=nan_for_zeros),
    )


def pagerank_edge_score(
    df: pd.DataFrame,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    damping: float = 0.85,
    eps: float = 1e-12,
    source_col: str = "source",
    target_col: str = "target",
    G: Optional[nx.DiGraph] = None,
    edge_weight: Optional[str] = None,
) -> TopologyFeatures:
    """Per-edge flux features from (personalized) PageRank.

    Thin wrapper over :func:`pagerank_edge_features` that also returns
    a per-edge feature matrix indexed by ``df``'s rows.

    Parameters
    ----------
    df : pd.DataFrame
        Edge dataframe (with ``source_col`` and ``target_col``).
    source_node, target_node : str, optional
        Anchor nodes for personalized PageRank. If ``None`` the
        un-personalized PageRank is used in that direction.
    damping : float, default 0.85
        PageRank damping factor.
    eps : float
        Hard floor used only if *every* edge has zero PageRank flux.
    source_col, target_col : str
        Column names for source and target nodes in `df`.
    G : nx.DiGraph, optional
        Pre-built graph to compute PageRank over. If ``None``, ``G`` is
        built from ``df`` via ``nx.from_pandas_edgelist``. Pass a custom
        ``G`` when the topology you want to score against is larger
        than (or differs from) the edges in ``df``.
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as the PageRank
        transition weight (forwarded to :func:`pagerank_edge_features`).
        When ``None`` (default) the unweighted/uniform behaviour is
        used.

    Returns
    -------
    TopologyFeatures
        Tuple ``(X_edge, log_edge_flux_src, log_edge_flux_tgt)``.
        ``X_edge`` has shape ``(n_edges, 2)`` with columns
        ``[log_edge_flux_src(u, v), log_edge_flux_tgt(u, v)]``, in the
        same row order as ``df``. The two dicts hold the per-edge
        log-fluxes for every edge in ``G``.
    """
    edges = _as_edge_frame(df, source_col, target_col)
    if G is None:
        G = nx.from_pandas_edgelist(
            edges, source=source_col, target=target_col, create_using=nx.DiGraph()
        )
    log_edge_src, log_edge_tgt = pagerank_edge_features(
        G, source_node=source_node, target_node=target_node, damping=damping, eps=eps,
        edge_weight=edge_weight,
    )

    srcs = edges[source_col].to_numpy()
    dsts = edges[target_col].to_numpy()
    log_src = np.fromiter(
        (log_edge_src.get((u, v), np.nan) for u, v in zip(srcs, dsts)),
        dtype=float, count=len(srcs),
    )
    log_tgt = np.fromiter(
        (log_edge_tgt.get((u, v), np.nan) for u, v in zip(srcs, dsts)),
        dtype=float, count=len(srcs),
    )
    X_edge = np.column_stack([log_src, log_tgt])
    return X_edge, log_edge_src, log_edge_tgt


def _random_walk_flux(
    G: nx.DiGraph,
    source: Optional[str] = None,
    n_steps: int = 10,
    n_restarts: int = __N_MC__,
    edge_attr: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Per-node and per-edge flux from short random walks on `G`.

    Each restart starts a walk at `source` (or, if `source` is None,
    at a uniformly-random node) and takes `n_steps` random steps along
    outgoing edges. We track:

      * ``node_flux[n]``  -- expected number of times node ``n`` is
        occupied per walk (including the starting position). Summing
        over all nodes gives ``n_steps + 1`` exactly.
      * ``edge_flux[(u, v)]`` -- expected number of times edge ``u -> v``
        is traversed per walk. Summing over all edges gives
        ``n_steps`` minus the expected number of dead-end teleports.

    Both are flux measures (counts per walk), not "ever-visited"
    probabilities, so they may exceed 1 for nodes/edges visited many
    times in a single walk.

    If `edge_attr` is given, transition probabilities at each node are
    proportional to that edge attribute; otherwise transitions are
    uniform over outgoing neighbours. Dead-ends (no outgoing edges)
    teleport back to the start; teleports are *not* counted as edge
    traversals.
    """
    rng = random.Random(seed)
    node_visits: Counter = Counter()
    edge_traversals: Counter = Counter()
    nodes = list(G.nodes())

    neighbors_cache: Dict[str, list] = {}
    probs_cache: Dict[str, list] = {}
    for node in nodes:
        out_neighbors = list(G.successors(node))
        neighbors_cache[node] = out_neighbors
        if edge_attr and out_neighbors:
            weights = [
                max(G.edges[node, nbr].get(edge_attr, 1.0), 0.0)
                for nbr in out_neighbors
            ]
            total = sum(weights)
            if total > 0:
                probs_cache[node] = [w / total for w in weights]
            else:
                probs_cache[node] = [1.0 / len(out_neighbors)] * len(out_neighbors)

    def _restart_node() -> str:
        return source if source is not None else rng.choice(nodes)

    for _ in range(n_restarts):
        current = _restart_node()
        node_visits[current] += 1
        for _ in range(n_steps):
            out_neighbors = neighbors_cache[current]
            if not out_neighbors:
                # Dead-end: teleport back to start. Don't count as an
                # edge traversal (no real edge was crossed).
                current = _restart_node()
            else:
                prev = current
                if edge_attr:
                    current = rng.choices(out_neighbors, weights=probs_cache[prev])[0]
                else:
                    current = rng.choice(out_neighbors)
                edge_traversals[(prev, current)] += 1
            node_visits[current] += 1

    inv_n = 1.0 / n_restarts
    node_flux = {node: cnt * inv_n for node, cnt in node_visits.items()}
    edge_flux = {edge: cnt * inv_n for edge, cnt in edge_traversals.items()}
    return node_flux, edge_flux


def random_walk_node_features(
    G: nx.DiGraph,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    n_steps: int = 10,
    n_restarts: int = 1_000_000,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-node flux features from short random walks.

    Computes two dicts (one per node in ``G``):

        log_flux_src(n) = log( E[# visits to n per walk from source] )
        log_flux_tgt(n) = log( E[# visits to n per walk from target on G^T] )

    estimated from ``n_restarts`` independent random walks. Reverse-graph
    walks are used to estimate ``log_flux_tgt``. These are *flux*
    measures (expected counts) rather than ever-visited probabilities,
    so values may exceed 1 for nodes traversed multiple times in a
    single walk. Unreachable nodes are floored to just below the
    smallest observed positive flux.

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as transition
        weights. When ``None`` (default), transitions are uniform over
        outgoing neighbours. When set, transitions out of each node
        are proportional to that attribute (weights are clamped to
        ``>= 0`` and rows of all-zero weight fall back to uniform).
    """
    _validate_anchors(G, source_node, target_node)
    node_flux_src, _ = _random_walk_flux(
        G, source=source_node, n_steps=n_steps, n_restarts=n_restarts,
        edge_attr=edge_weight, seed=seed,
    )
    node_flux_tgt, _ = _random_walk_flux(
        G.reverse(copy=False),
        source=target_node,
        n_steps=n_steps,
        n_restarts=n_restarts,
        edge_attr=edge_weight,
        seed=(seed + 1) if seed is not None else None,
    )
    for n in G.nodes():
        node_flux_src.setdefault(n, 0.0)
        node_flux_tgt.setdefault(n, 0.0)
    return (
        _log_dict_with_floor(node_flux_src, eps, nan_for_zeros=nan_for_zeros),
        _log_dict_with_floor(node_flux_tgt, eps, nan_for_zeros=nan_for_zeros),
    )


def random_walk_edge_features(
    G: nx.DiGraph,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    n_steps: int = 10,
    n_restarts: int = 1_000_000,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Per-edge flux features from short random walks.

    Computes two dicts (one per directed edge ``(u, v)`` in ``G``):

        log_edge_flux_src(u, v) = log( E[# traversals of u->v per walk
                                        from source on G] )
        log_edge_flux_tgt(u, v) = log( E[# traversals of v->u per walk
                                        from target on G^T] )
                                = log( E[# backward traversals of u->v
                                        per reverse walk] )

    These are estimated by counting actual edge traversals in
    ``n_restarts`` short random walks (forward on ``G`` from
    ``source_node`` and on ``G.reverse()`` from ``target_node``).
    The reverse-graph dictionary's keys are flipped from ``(v, u)``
    back to ``(u, v)`` so both dicts are indexed by the same original
    edge orientation.

    These are *flux* measures (expected counts per walk), not
    "ever-traversed" probabilities, so values may exceed 1 for edges
    crossed multiple times in a single walk. Unreachable edges are
    floored to just below the smallest observed positive flux (or to
    ``NaN`` when ``nan_for_zeros=True``).

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as transition
        weights. When ``None`` (default), transitions are uniform
        over outgoing neighbours. When set, transitions out of each
        node are proportional to that attribute (weights clamped to
        ``>= 0``, all-zero rows fall back to uniform).
    """
    _validate_anchors(G, source_node, target_node)
    _, edge_flux_src = _random_walk_flux(
        G, source=source_node, n_steps=n_steps, n_restarts=n_restarts,
        edge_attr=edge_weight, seed=seed,
    )
    _, edge_flux_tgt_rev = _random_walk_flux(
        G.reverse(copy=False),
        source=target_node,
        n_steps=n_steps,
        n_restarts=n_restarts,
        edge_attr=edge_weight,
        seed=(seed + 1) if seed is not None else None,
    )

    edge_flux_tgt: Dict[Tuple[str, str], float] = {
        (u, v): flux for (v, u), flux in edge_flux_tgt_rev.items()
    }

    for u, v in G.edges():
        edge_flux_src.setdefault((u, v), 0.0)
        edge_flux_tgt.setdefault((u, v), 0.0)

    return (
        _log_dict_with_floor(edge_flux_src, eps, nan_for_zeros=nan_for_zeros),
        _log_dict_with_floor(edge_flux_tgt, eps, nan_for_zeros=nan_for_zeros),
    )


def random_walk_edge_score(
    df: pd.DataFrame,
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    n_steps: int = 10,
    n_restarts: int = 1_000_000,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    source_col: str = "source",
    target_col: str = "target",
    G: Optional[nx.DiGraph] = None,
    edge_weight: Optional[str] = None,
) -> TopologyFeatures:
    """Per-edge flux features from short random walks.

    Thin wrapper over :func:`random_walk_edge_features` that also
    returns a per-edge feature matrix indexed by ``df``'s rows.

    Parameters
    ----------
    df : pd.DataFrame
        Edge dataframe.
    source_node, target_node : str, optional
        Anchors for the forward and reverse walks. If ``None``, walks
        start from a uniformly-random node each restart (rough analogue
        of un-personalized PageRank, useful only as a rough baseline).
    n_steps : int, default 10
        Walk length per restart. Should be on the order of the longest
        plausible source -> target path; values that are too small miss
        deep edges, values that are too large dilute the signal.
    n_restarts : int, default 10000
        Number of independent walks. Larger gives lower-variance
        edge-flux estimates at the cost of runtime.
    seed : int, optional
        RNG seed for reproducibility. The reverse-graph walk is seeded
        with ``seed + 1`` so the two walks are independent.
    eps : float
        Floor added before taking the logarithm.
    source_col, target_col : str
        Column names for source and target nodes in `df`.
    G : nx.DiGraph, optional
        Pre-built graph to walk on. If ``None``, ``G`` is built from
        ``df``. Pass a custom ``G`` when the topology you want to score
        against is larger than (or differs from) the edges in ``df``.
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as transition
        weights in the random walk (forwarded to
        :func:`random_walk_edge_features`). When ``None`` (default)
        transitions are uniform over outgoing neighbours.

    Returns
    -------
    TopologyFeatures
        Tuple ``(X_edge, log_edge_flux_src, log_edge_flux_tgt)``.
        ``X_edge`` has shape ``(n_edges, 2)`` with columns
        ``[log_edge_flux_src(u, v), log_edge_flux_tgt(u, v)]``, in the
        same row order as ``df``. Edges that no walk traversed get a
        finite log-floor just below the smallest observed positive
        flux.
    """
    edges = _as_edge_frame(df, source_col, target_col)
    if G is None:
        G = nx.from_pandas_edgelist(
            edges, source=source_col, target=target_col, create_using=nx.DiGraph()
        )
    log_edge_src, log_edge_tgt = random_walk_edge_features(
        G,
        source_node=source_node,
        target_node=target_node,
        n_steps=n_steps,
        n_restarts=n_restarts,
        seed=seed,
        eps=eps,
        edge_weight=edge_weight,
    )
    srcs = edges[source_col].to_numpy()
    dsts = edges[target_col].to_numpy()
    log_src = np.fromiter(
        (log_edge_src.get((u, v), np.nan) for u, v in zip(srcs, dsts)),
        dtype=float, count=len(srcs),
    )
    log_tgt = np.fromiter(
        (log_edge_tgt.get((u, v), np.nan) for u, v in zip(srcs, dsts)),
        dtype=float, count=len(srcs),
    )
    X_edge = np.column_stack([log_src, log_tgt])
    return X_edge, log_edge_src, log_edge_tgt


def _random_walk_betweenness_mc(
    G: nx.DiGraph,
    source: str,
    target: str,
    n_steps: int = 10,
    n_restarts: int = __N_MC__,
    edge_attr: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], int]:
    """Monte-Carlo random-walk betweenness for a single (source, target) pair.

    Runs ``n_restarts`` random walks of up to ``n_steps`` from ``source``
    on ``G``, with ``target`` treated as an absorbing state. The walk
    stops the moment it reaches ``target``; walks that hit a dead-end or
    exhaust ``n_steps`` without reaching ``target`` are *discarded*
    (no teleporting -- that would corrupt the conditional path).

    Among the kept walks (those that actually reached ``target``) we
    estimate

        node_betweenness(v)  = E[ # visits to v        | walk reaches t ]
        edge_betweenness(u,v) = E[ # traversals of u->v | walk reaches t ]

    which is the standard Monte-Carlo estimator of the source -> target
    random-walk betweenness (the empirical analogue of the closed-form
    Doob h-transform: for each kept walk, every visited node and every
    traversed edge gets one count, normalised by the number of kept
    walks).

    Notes on correctness: rejection sampling on the event ``{walk
    reaches t in <= n_steps}`` is *exactly* sampling from the
    h-transformed (target-conditioned) chain restricted to walks of
    that length, so the estimator is unbiased for the
    horizon-``n_steps`` betweenness. As ``n_steps -> infty`` (with
    ``n_restarts`` large enough to keep variance bounded) it converges
    to the true unconditional betweenness.

    Notes on variance: the effective sample size is the number of walks
    that hit ``target`` (returned as ``n_hits``). When
    ``Pr(reach t from s in n_steps)`` is small, very few walks survive
    and per-edge estimates are noisy; ``n_hits`` should be checked /
    surfaced by callers.

    Parameters
    ----------
    G : nx.DiGraph
    source, target : node ids; both required.
    n_steps : int
        Maximum walk length (truncation horizon).
    n_restarts : int
        Number of independent walks to attempt.
    edge_attr : str, optional
        If given, transition probabilities are proportional to this
        edge attribute; otherwise transitions are uniform over
        outgoing neighbours.
    seed : int, optional

    Returns
    -------
    node_betweenness : dict[node -> float]
    edge_betweenness : dict[(u, v) -> float]
    n_hits : int
        Number of walks that reached ``target`` (the effective sample
        size for the MC estimator).
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())

    neighbors_cache: Dict[str, list] = {}
    probs_cache: Dict[str, list] = {}
    for node in nodes:
        out_neighbors = list(G.successors(node))
        neighbors_cache[node] = out_neighbors
        if edge_attr and out_neighbors:
            weights = [
                max(G.edges[node, nbr].get(edge_attr, 1.0), 0.0)
                for nbr in out_neighbors
            ]
            total = sum(weights)
            if total > 0:
                probs_cache[node] = [w / total for w in weights]
            else:
                probs_cache[node] = [1.0 / len(out_neighbors)] * len(out_neighbors)

    node_count: Counter = Counter()
    edge_count: Counter = Counter()
    n_hits = 0

    if source == target:
        # Degenerate query: the walk is trivially "at target". Count
        # one visit to source per restart and zero edges. This keeps
        # the estimator well-defined and matches the n_steps=0 limit.
        node_count[source] = n_restarts
        n_hits = n_restarts
    else:
        for _ in range(n_restarts):
            path_nodes = [source]
            path_edges: list = []
            cur = source
            hit = False
            for _ in range(n_steps):
                out_neighbors = neighbors_cache[cur]
                if not out_neighbors:
                    break  # dead-end; discard this walk
                if edge_attr:
                    nxt = rng.choices(out_neighbors, weights=probs_cache[cur])[0]
                else:
                    nxt = rng.choice(out_neighbors)
                path_edges.append((cur, nxt))
                path_nodes.append(nxt)
                cur = nxt
                if cur == target:
                    hit = True
                    break
            if hit:
                n_hits += 1
                for n in path_nodes:
                    node_count[n] += 1
                for e in path_edges:
                    edge_count[e] += 1

    if n_hits == 0:
        return {}, {}, 0

    inv_hits = 1.0 / n_hits
    node_b = {n: c * inv_hits for n, c in node_count.items()}
    edge_b = {e: c * inv_hits for e, c in edge_count.items()}
    return node_b, edge_b, n_hits


def random_walk_betweenness_node_features(
    G: nx.DiGraph,
    source_node: str,
    target_node: str,
    n_steps: int = 10,
    n_restarts: int = __N_MC__,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-node Monte-Carlo random-walk betweenness for ``(s, t)``.

    Estimates ``B(v; s, t) = E[ # visits to v | walk from s reaches t ]``
    via ``_random_walk_betweenness_mc``. Unlike :func:`random_walk_node_features`
    (which returns *two* one-sided fluxes that callers multiply), this
    is already a true source -> target composite, so to fit the
    ``(log_src, log_tgt)`` contract used by :func:`topology_baseline_score`
    we return:

      * ``log_b``   -- ``log(node betweenness)`` (with the usual floor
        for never-visited nodes).
      * ``log_one`` -- ``0.0`` for every node in ``G`` (so ``exp = 1``).

    Then the downstream product ``flux_src * flux_tgt`` evaluates to
    ``betweenness * 1 = betweenness``, which is the correct composite
    score for this single bidirectional measure. The same shape also
    keeps :func:`adjust_for_topology` working: the ``log_tgt`` column
    is constant, so it contributes nothing beyond the intercept and the
    OLS effectively regresses on ``log(betweenness)`` alone.

    Both ``source_node`` and ``target_node`` are required.

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as transition
        weights in the underlying random walk. When ``None`` (default)
        transitions are uniform; when set, transitions out of each
        node are proportional to that attribute.
    """
    if source_node is None or target_node is None:
        raise ValueError(
            "random_walk_betweenness requires both source_node and target_node "
            "(betweenness is defined on a specific source -> target query)."
        )
    _validate_anchors(G, source_node, target_node)

    node_b, _edge_b, _n_hits = _random_walk_betweenness_mc(
        G,
        source=source_node,
        target=target_node,
        n_steps=n_steps,
        n_restarts=n_restarts,
        edge_attr=edge_weight,
        seed=seed,
    )
    for n in G.nodes():
        node_b.setdefault(n, 0.0)

    log_b = _log_dict_with_floor(node_b, eps, nan_for_zeros=nan_for_zeros)
    log_one = {n: 0.0 for n in G.nodes()}
    return log_b, log_one


def random_walk_betweenness_edge_features(
    G: nx.DiGraph,
    source_node: str,
    target_node: str,
    n_steps: int = 10,
    n_restarts: int = __N_MC__,
    seed: Optional[int] = None,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Per-edge Monte-Carlo random-walk betweenness for ``(s, t)``.

    Estimates
    ``B(u, v; s, t) = E[ # traversals of u->v | walk from s reaches t ]``
    via ``_random_walk_betweenness_mc``. Returns the same
    ``(log_b, log_one)`` shape as :func:`random_walk_betweenness_node_features`
    so it slots into the existing dispatch contract: the second dict
    has value ``0.0`` for every edge in ``G`` (so ``exp = 1``) and the
    downstream product yields the betweenness directly.

    Both ``source_node`` and ``target_node`` are required.

    Parameters
    ----------
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as transition
        weights in the underlying random walk. When ``None`` (default)
        transitions are uniform; when set, transitions out of each
        node are proportional to that attribute.
    """
    if source_node is None or target_node is None:
        raise ValueError(
            "random_walk_betweenness requires both source_node and target_node "
            "(betweenness is defined on a specific source -> target query)."
        )
    _validate_anchors(G, source_node, target_node)

    _node_b, edge_b, _n_hits = _random_walk_betweenness_mc(
        G,
        source=source_node,
        target=target_node,
        n_steps=n_steps,
        n_restarts=n_restarts,
        edge_attr=edge_weight,
        seed=seed,
    )
    for u, v in G.edges():
        edge_b.setdefault((u, v), 0.0)

    log_b = _log_dict_with_floor(edge_b, eps, nan_for_zeros=nan_for_zeros)
    log_one = {(u, v): 0.0 for (u, v) in G.edges()}
    return log_b, log_one


def _bfs_path_counts(
    G: nx.DiGraph, source: str
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Brandes-style BFS over an unweighted ``G`` from ``source``.

    Returns ``(sigma, dist)`` where

      * ``sigma[v]`` = number of distinct shortest paths from ``source``
        to ``v`` (stored as ``float`` because the count can grow large
        on dense graphs and ``float`` matches the downstream
        normalisation by ``sigma[target]``).
      * ``dist[v]``  = unweighted shortest-path distance from ``source``
        to ``v``.

    Only nodes reachable from ``source`` appear in the dicts.
    """
    sigma: Dict[str, float] = {source: 1.0}
    dist: Dict[str, int] = {source: 0}
    queue: deque = deque([source])
    while queue:
        u = queue.popleft()
        d_next = dist[u] + 1
        for v in G.successors(u):
            if v not in dist:
                dist[v] = d_next
                sigma[v] = sigma[u]
                queue.append(v)
            elif dist[v] == d_next:
                # Another shortest path discovered to v through u.
                sigma[v] += sigma[u]
            # dist[v] < d_next means v is closer; ignore (not on a
            # shortest path through u).
    return sigma, dist


def _shortest_path_betweenness_pair(
    G: nx.DiGraph, source: str, target: str
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
    """Per-(s, t) shortest-path betweenness via Brandes' decomposition.

    For an unweighted directed graph, the (s, t)-pair betweenness of a
    node ``v`` is

        B(v; s, t) = sigma_s(v) * sigma_t(v) / sigma_s(t)
                     if v lies on a shortest s -> t path,
                   = 0 otherwise.

    A node ``v`` lies on a shortest s -> t path iff
    ``dist_s(v) + dist_t(v) == dist_s(t)``. The edge analogue:

        B(u, v; s, t) = sigma_s(u) * sigma_t(v) / sigma_s(t)
                     if dist_s(u) + 1 + dist_t(v) == dist_s(t),
                   = 0 otherwise.

    Here ``sigma_s, dist_s`` come from a forward BFS on ``G`` from
    ``source`` and ``sigma_t, dist_t`` from a forward BFS on
    ``G.reverse()`` from ``target`` (which equivalently counts shortest
    paths *to* ``target`` in the original graph).

    Returns ``(node_b, edge_b, total_paths)``. If ``target`` is
    unreachable from ``source``, ``total_paths == 0`` and both dicts
    are empty.

    Cost: two BFS passes, ``O(|V| + |E|)``.
    """
    sigma_s, dist_s = _bfs_path_counts(G, source)
    if target not in dist_s:
        return {}, {}, 0.0
    sigma_t, dist_t = _bfs_path_counts(G.reverse(copy=False), target)

    d_st = dist_s[target]
    total = sigma_s[target]
    inv_total = 1.0 / total

    node_b: Dict[str, float] = {}
    for v in sigma_s:
        if v in dist_t and dist_s[v] + dist_t[v] == d_st:
            node_b[v] = sigma_s[v] * sigma_t[v] * inv_total

    edge_b: Dict[Tuple[str, str], float] = {}
    for u, v in G.edges():
        if (
            u in dist_s
            and v in dist_t
            and dist_s[u] + 1 + dist_t[v] == d_st
        ):
            edge_b[(u, v)] = sigma_s[u] * sigma_t[v] * inv_total

    return node_b, edge_b, total


def betweenness_centrality_node_features(
    G: nx.DiGraph,
    source_node: str,
    target_node: str,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-node shortest-path betweenness centrality for ``(s, t)``.

    Computes

        B(v; s, t) = (# shortest s -> t paths through v)
                     / (# shortest s -> t paths)

    for every node ``v`` in ``G``, via Brandes' two-BFS decomposition
    (see :func:`_shortest_path_betweenness_pair`). This is the classic
    Freeman / Brandes betweenness *restricted to a single source-target
    pair* -- the most naive topology-only baseline you can write for an
    s -> t explanation task. By construction, ``B = 0`` for any node
    not on at least one shortest s -> t path, and ``B = 1`` for both
    ``s`` and ``t`` themselves.

    Returns ``(log_b, log_one)`` to fit the dispatch contract used by
    :func:`topology_baseline_score` and :func:`adjust_for_topology`:
    the ``log_one`` half is constant ``0.0`` (so ``exp = 1``) and the
    downstream product ``flux_src * flux_tgt`` evaluates to the
    betweenness directly.

    Both ``source_node`` and ``target_node`` are required. If
    ``target_node`` is unreachable from ``source_node`` the betweenness
    is zero everywhere and ``log_b`` is floored (or NaN, depending on
    ``nan_for_zeros``).

    Notes
    -----
    * Unweighted shortest paths only -- transition costs are ignored.
    * Deterministic (no Monte-Carlo noise) and very fast: two BFS
      passes per query, ``O(|V| + |E|)``.

    The ``edge_weight`` argument is accepted only for parity with the
    other topology methods; passing a non-``None`` value raises
    ``NotImplementedError`` because the underlying Brandes BFS is
    inherently unweighted.
    """
    if edge_weight is not None:
        raise NotImplementedError(
            "betweenness_centrality does not support edge_weight; use "
            "'pagerank', 'random_walk', or 'rw_betweenness' for a weighted "
            "topology baseline."
        )
    if source_node is None or target_node is None:
        raise ValueError(
            "betweenness_centrality requires both source_node and target_node "
            "(it is defined on a specific source -> target query)."
        )
    _validate_anchors(G, source_node, target_node)

    node_b, _edge_b, _total = _shortest_path_betweenness_pair(
        G, source_node, target_node
    )
    for n in G.nodes():
        node_b.setdefault(n, 0.0)

    log_b = _log_dict_with_floor(node_b, eps, nan_for_zeros=nan_for_zeros)
    log_one = {n: 0.0 for n in G.nodes()}
    return log_b, log_one


def betweenness_centrality_edge_features(
    G: nx.DiGraph,
    source_node: str,
    target_node: str,
    eps: float = 1e-12,
    nan_for_zeros: bool = False,
    edge_weight: Optional[str] = None,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Per-edge shortest-path betweenness centrality for ``(s, t)``.

    Computes

        B(u, v; s, t) = (# shortest s -> t paths through edge (u, v))
                        / (# shortest s -> t paths)

    via Brandes' two-BFS decomposition (see
    :func:`_shortest_path_betweenness_pair`). Returns the same
    ``(log_b, log_one)`` shape as
    :func:`betweenness_centrality_node_features` so it slots into the
    existing dispatch contract.

    Both ``source_node`` and ``target_node`` are required.

    The ``edge_weight`` argument is accepted only for parity with the
    other topology methods; passing a non-``None`` value raises
    ``NotImplementedError`` because the underlying Brandes BFS is
    inherently unweighted.
    """
    if edge_weight is not None:
        raise NotImplementedError(
            "betweenness_centrality does not support edge_weight; use "
            "'pagerank', 'random_walk', or 'rw_betweenness' for a weighted "
            "topology baseline."
        )
    if source_node is None or target_node is None:
        raise ValueError(
            "betweenness_centrality requires both source_node and target_node "
            "(it is defined on a specific source -> target query)."
        )
    _validate_anchors(G, source_node, target_node)

    _node_b, edge_b, _total = _shortest_path_betweenness_pair(
        G, source_node, target_node
    )
    for u, v in G.edges():
        edge_b.setdefault((u, v), 0.0)

    log_b = _log_dict_with_floor(edge_b, eps, nan_for_zeros=nan_for_zeros)
    log_one = {(u, v): 0.0 for (u, v) in G.edges()}
    return log_b, log_one


_TOPOLOGY_NODE_METHODS: Dict[str, Callable[..., Tuple[Dict[str, float], Dict[str, float]]]] = {
    "pagerank": pagerank_node_features,
    "random_walk": random_walk_node_features,
    "rw_betweenness": random_walk_betweenness_node_features,
    "betweenness_centrality": betweenness_centrality_node_features,
}

_TOPOLOGY_EDGE_METHODS: Dict[
    str, Callable[..., Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]]
] = {
    "pagerank": pagerank_edge_features,
    "random_walk": random_walk_edge_features,
    "rw_betweenness": random_walk_betweenness_edge_features,
    "betweenness_centrality": betweenness_centrality_edge_features,
}

# Backwards-compat alias: callers that previously read this dispatch
# table assumed it was the per-node table.
_TOPOLOGY_METHODS = _TOPOLOGY_NODE_METHODS


def topology_baseline_score(
    G: nx.DiGraph,
    source_node: str,
    target_node: str,
    method: str = "pagerank",
    level: str = "node",
    edges: Optional[pd.DataFrame] = None,
    source_col: str = "source",
    target_col: str = "target",
    normalize: bool = True,
    log: bool = False,
    eps: float = 1e-12,
    edge_weight: Optional[str] = None,
    **method_kwargs,
) -> pd.Series:
    """Pure-topology baseline score for a source -> target query.

    Returns a single composite score per node (or per edge) that uses
    *only* network topology -- no model, no learned edge weights -- so
    it can be compared head-to-head against explainer scores
    (Contrastive GSNN, IG, Occlusion, ...). The intent is to answer:

        "Could topology alone, without any prediction logic, produce
         equivalent or better explanations than the model?"

    For each node ``n`` (or edge ``(u, v)``) the score combines
    forward and backward reachability into a true source -> target
    flux:

        level='node':   score(n)    = flux_src(n) * flux_tgt(n)
        level='edge':   score(u, v) = edge_flux_src(u, v)
                                      * edge_flux_tgt(u, v)

    where the per-node and per-edge fluxes come from ``method``:

      * ``'pagerank'``       -- (personalized) PageRank on ``G`` and on
        ``G.reverse()``, anchored at ``source_node`` and
        ``target_node`` respectively. Per-edge flux is the
        PageRank-implied transition flow,
        ``edge_flux_src(u, v) = pi_src(u) / out_deg(u)`` and
        ``edge_flux_tgt(u, v) = pi_tgt(v) / in_deg(v)``, i.e. the
        unweighted-random-walk flow through the edge under the
        personalized PR stationary distribution. Closed-form and
        deterministic. The composite ``flux_src * flux_tgt`` is a
        factorised (independence-assumption) proxy for source -> target
        flow.
      * ``'random_walk'``    -- Monte-Carlo short-walk fluxes
        (expected node visits / edge traversals per walk on ``G`` from
        the source and on ``G.reverse()`` from the target), controlled
        by ``n_steps`` / ``n_restarts``. Same factorised composite as
        ``'pagerank'``.
      * ``'rw_betweenness'`` -- Monte-Carlo source -> target random-walk
        *betweenness*: runs ``n_restarts`` walks from ``source_node`` of
        up to ``n_steps`` steps with ``target_node`` as an absorbing
        state, and averages node visits / edge traversals over the
        walks that actually reached the target. This is a true
        bidirectional measure (not a factorised proxy), so the
        ``flux_src`` half holds the betweenness and the ``flux_tgt``
        half is the constant ``1`` -- the composite score is the
        betweenness directly. More faithful than the factorised
        methods but Monte-Carlo noisy when the source -> target
        hitting probability is small (i.e. when few walks reach the
        target within ``n_steps``); raise ``n_restarts`` or
        ``n_steps`` if the result looks degenerate.
      * ``'betweenness_centrality'`` -- classic shortest-path
        betweenness *restricted to the single ``(source, target)``
        pair*: ``B(v) = (# shortest s -> t paths through v) /
        (# shortest s -> t paths)``, computed via Brandes' two-BFS
        decomposition (deterministic, ``O(|V| + |E|)``). Like
        ``'rw_betweenness'`` it is a true bidirectional score (the
        ``flux_tgt`` half is constant ``1``), but it cares only about
        *shortest* paths, so it is the most aggressive sparsification
        baseline and is exactly zero off the shortest-path subgraph.
        Unweighted only.

    Both forward and reverse personalizations are required, so
    ``source_node`` and ``target_node`` are both mandatory and must
    exist in ``G``.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph to score against (e.g. the bionetwork built
        from the GSNN edge list).
    source_node : str
        Anchor for forward reachability (e.g. the drug node).
        Required and must be in ``G``.
    target_node : str
        Anchor for backward reachability (e.g. the read-out gene
        node). Required and must be in ``G``.
    method : {'pagerank', 'random_walk', 'rw_betweenness', \
'betweenness_centrality'}, default 'pagerank'
        Topology metric. ``'pagerank'`` and ``'random_walk'`` produce
        a factorised flux ``flux_src * flux_tgt``;
        ``'rw_betweenness'`` produces a Monte-Carlo source -> target
        random-walk betweenness; ``'betweenness_centrality'`` produces
        the deterministic shortest-path betweenness for the single
        ``(s, t)`` pair.
    level : {'node', 'edge'}, default 'node'
        Unit of the returned score.
    edges : pd.DataFrame, optional
        Only used when ``level='edge'``. If given, scores are
        returned for the edges in this dataframe (in row order),
        which is useful when callers want the score aligned with an
        explainer output that may not cover every edge in ``G``.
        If ``None``, every edge in ``G`` is scored.
    source_col, target_col : str
        Column / index level names for source and target nodes in
        ``edges`` (only used when ``level='edge'`` and ``edges`` is
        provided).
    normalize : bool, default True
        If True, divide the raw flux by its maximum so scores live in
        ``[0, 1]`` and are directly comparable to explainer scores
        like the Contrastive GSNN inclusion probability.
    log : bool, default False
        If True, return ``log(score + eps)`` instead of the raw flux.
        Useful when the flux distribution is heavy-tailed and you
        want to rank with a more spread-out signal.
    eps : float, default 1e-12
        Floor used by ``log`` and to avoid log(0) for unreachable
        nodes/edges.
    edge_weight : str, optional
        Name of an edge attribute on ``G`` to use as the transition
        weight in the underlying topology computation. When ``None``
        (default), the topology methods behave exactly as before
        (uniform transitions / unweighted shortest paths). When set:

          * ``'pagerank'`` -- forwarded as ``weight=edge_weight`` to
            :func:`networkx.pagerank` and used to compute the per-edge
            flux ``pi(u) * w(u, v) / W_out(u)``.
          * ``'random_walk'`` and ``'rw_betweenness'`` -- transitions
            out of each node are proportional to the attribute
            (clamped to ``>= 0``; rows of all-zero weight fall back
            to uniform).
          * ``'betweenness_centrality'`` -- raises
            ``NotImplementedError`` (Brandes BFS is unweighted).

        The attribute must be present on at least one edge of ``G``;
        a ``ValueError`` is raised otherwise to catch typos.
    **method_kwargs
        Forwarded to the underlying topology method (e.g.
        ``damping=0.85`` for ``'pagerank'``, or
        ``n_steps`` / ``n_restarts`` / ``seed`` for ``'random_walk'``
        and ``'rw_betweenness'``).

    Returns
    -------
    pd.Series
        Topology-only baseline score, named ``'topology_score'``.

        * ``level='node'`` -- index is node name.
        * ``level='edge'`` -- index is a ``(source, target)``
          MultiIndex when ``edges`` is ``None``; otherwise the index
          is preserved from ``edges``.

        Higher = the node/edge sits more naturally on flux between
        ``source_node`` and ``target_node`` based on graph structure
        alone. Use ``Series.rank(...)`` or correlate with the
        explainer score column to compare against the model.

    Examples
    --------
    Per-node PageRank baseline aligned with a notebook explainer
    output ``cres_node`` (one row per node)::

        baseline = topology_baseline_score(
            G, source_node='DRUG__BRD-K12343256',
            target_node='GENE__DUSP6', method='pagerank', level='node',
        )
        cres_node['topology_baseline'] = (
            cres_node['node'].map(baseline).fillna(0.0)
        )
        # rank correlation between explanation and topology baseline
        cres_node[['mean_gsnn_score', 'topology_baseline']].corr('spearman')

    Per-edge random-walk baseline aligned with an edge-level
    explainer output ``cres_agg``::

        baseline = topology_baseline_score(
            G, source_node='DRUG__BRD-K12343256',
            target_node='GENE__DUSP6',
            method='random_walk', level='edge',
            edges=cres_agg.reset_index()[['source', 'target']],
            n_steps=10, n_restarts=10000, seed=0,
        )
    """
    if level not in {"node", "edge"}:
        raise ValueError(
            f"Unknown level {level!r}. Must be one of 'node' or 'edge'."
        )
    method_table = _TOPOLOGY_NODE_METHODS if level == "node" else _TOPOLOGY_EDGE_METHODS
    if method not in method_table:
        raise ValueError(
            f"Unknown method {method!r} at level {level!r}. "
            f"Available methods: {sorted(method_table)}"
        )
    if source_node is None or target_node is None:
        raise ValueError(
            "topology_baseline_score requires both source_node and target_node "
            "(the baseline is defined on the source -> target flux)."
        )
    _validate_anchors(G, source_node, target_node)

    if edge_weight is not None:
        # Catch typos early: networkx silently falls back to weight=1.0
        # for missing edge attributes, which would mask a misspelled
        # column name as "the topology is just uniform".
        if not any(edge_weight in d for _, _, d in G.edges(data=True)):
            raise ValueError(
                f"edge_weight={edge_weight!r} not found on any edge of G."
            )

    # Compute forward/reverse fluxes on the raw scale (we want the
    # product, not the log of either). The dispatched function returns
    # log-fluxes per node (level='node') or per edge (level='edge'),
    # which we exponentiate before combining.
    feature_fn = method_table[method]
    log_src, log_tgt = feature_fn(
        G,
        source_node=source_node,
        target_node=target_node,
        eps=method_kwargs.pop("topology_eps", 1e-12),
        nan_for_zeros=False,
        edge_weight=edge_weight,
        **method_kwargs,
    )
    flux_src = {k: float(np.exp(v)) for k, v in log_src.items()}
    flux_tgt = {k: float(np.exp(v)) for k, v in log_tgt.items()}

    if level == "node":
        nodes = list(G.nodes())
        flux = np.array(
            [flux_src.get(n, 0.0) * flux_tgt.get(n, 0.0) for n in nodes],
            dtype=float,
        )
        index = pd.Index(nodes, name="node")
    else:  # level == 'edge'
        if edges is not None:
            edge_df = _as_edge_frame(edges, source_col, target_col)
            srcs = edge_df[source_col].to_numpy()
            dsts = edge_df[target_col].to_numpy()
            if isinstance(edges.index, pd.MultiIndex) and {source_col, target_col}.issubset(
                edges.index.names
            ):
                index = edges.index
            else:
                index = pd.MultiIndex.from_arrays(
                    [srcs, dsts], names=[source_col, target_col]
                )
        else:
            edge_pairs = list(G.edges())
            srcs = np.array([u for u, _ in edge_pairs], dtype=object)
            dsts = np.array([v for _, v in edge_pairs], dtype=object)
            index = pd.MultiIndex.from_arrays(
                [srcs, dsts], names=[source_col, target_col]
            )
        flux = np.array(
            [
                flux_src.get((u, v), 0.0) * flux_tgt.get((u, v), 0.0)
                for u, v in zip(srcs, dsts)
            ],
            dtype=float,
        )

    if normalize:
        m = float(flux.max()) if flux.size else 0.0
        if m > 0:
            flux = flux / m

    if log:
        flux = np.log(np.maximum(flux, eps))

    return pd.Series(flux, index=index, name="topology_score")


def adjust_for_topology(
    df: pd.DataFrame,
    score_col: str,
    G: Optional[nx.DiGraph] = None,
    method: str = "pagerank",
    source_node: Optional[str] = None,
    target_node: Optional[str] = None,
    logit: bool = True,
    eps: float = 1e-6,
    source_col: str = "source",
    target_col: str = "target",
    node_col: str = "node",
    level: str = "edge",
    drop_unreachable: bool = True,
    verbose: bool = False,
    **method_kwargs,
) -> pd.DataFrame:
    """Adjust observed edge importance scores for network topology.

    Fits a (logit-space) linear regression of the observed score on a
    topology-derived per-edge feature, then reports the predicted score
    (``topology_score``) and the residual (``topology_adjusted_score``).

    Concretely, for the default ``pagerank`` method:

    1. Compute *two* topology features per row:

       * ``level='edge'``:
         ``x_src = log(edge_flux_src(u, v))`` and
         ``x_tgt = log(edge_flux_tgt(u, v))`` -- the forward and
         reverse fluxes through the specific edge ``(u, v)``.
       * ``level='node'``:
         ``x_src = log(flux_src(n))`` and
         ``x_tgt = log(flux_tgt(n))``.
    2. Transform observed scores ``y -> logit(y)`` (if ``logit=True``).
    3. Fit OLS ``y_t ~ a + b_src * x_src + b_tgt * x_tgt``. Forward- and
       reverse-reachability get *separate* slopes, which matters when
       one side of the path carries more signal than the other.
    4. ``topology_score = sigmoid(a + b_src * x_src + b_tgt * x_tgt)`` --
       the score the model *would* assign to a row in this topological
       position if it behaved like the average row with that flux.
    5. ``topology_adjusted_score = observed_score - topology_score``.

    Interpretation:

    * **Positive** ``topology_adjusted_score`` -> the model singles out
      this edge beyond what topology predicts (learned signal).
    * **~ Zero** -> the edge's importance is well explained by where it
      sits in the network.
    * **Negative** -> the model uses this edge less than its topological
      position would suggest.

    Parameters
    ----------
    df : pd.DataFrame
        The unit of analysis depends on ``level``:

        * ``level='edge'`` -- edge dataframe with ``source_col``,
          ``target_col``, and ``score_col``. The source/target columns
          may also be supplied via a ``(source, target)`` MultiIndex.
        * ``level='node'`` -- node dataframe with ``node_col`` and
          ``score_col`` (one row per node). ``G`` must be provided in
          this mode because the dataframe does not carry the topology.
    score_col : str
        Name of the column holding observed importance scores. Assumed
        to lie in ``(0, 1)`` when ``logit=True``.
    G : nx.DiGraph, optional
        Directed graph used to compute the topology features. If
        ``None`` and ``level='edge'``, ``G`` is built from the edges
        of ``df``. If ``None`` and ``level='node'``, a ``ValueError``
        is raised. ``G`` should contain every node referenced by
        ``df`` (extra nodes/edges are fine and let you score against
        the full bionetwork rather than just observed edges).
    method : {'pagerank', 'random_walk'}
        Topology metric to use.

        * ``'pagerank'`` -- closed-form personalized PageRank flux
          (deterministic, fast). At ``level='edge'`` the per-edge
          features come from ``edge_flux = pi(u) / out_deg(u)`` (and
          analogously on the reverse graph).
        * ``'random_walk'`` -- Monte-Carlo short-walk fluxes
          (expected node visits / edge traversals per walk;
          stochastic, controlled by ``n_steps`` / ``n_restarts``).

        The dispatch table is set up so that additional methods
        (Katz centrality, simple-path counts, etc.) can be added
        without changing the public API.
    source_node, target_node : str, optional
        Anchor nodes (e.g. drug node and target gene node). When
        provided, the topology metric uses *personalized* PageRank
        from these anchors, so the adjustment is conditioned on the
        specific source -> target query.
    logit : bool, default True
        If True, the regression is fit in logit space (appropriate for
        scores that are probabilities in ``(0, 1)``, such as the
        Contrastive GSNN inclusion probability). Set to False for
        signed scores like Integrated Gradients or Occlusion.
    eps : float
        Clipping value used to keep scores away from ``{0, 1}`` before
        the logit transform.
    source_col, target_col : str
        Column / index level names for source and target nodes (used
        only when ``level='edge'``).
    node_col : str, default 'node'
        Column name holding the node identifier (used only when
        ``level='node'``).
    level : {'edge', 'node'}, default 'edge'
        Unit of analysis for the OLS fit.

        * ``'edge'`` -- one row per edge. ``y`` is the observed edge
          score, ``X = [log_edge_flux_src(u, v), log_edge_flux_tgt(u, v)]``.
          The fit is per-edge, but tends to have low R^2 because
          per-edge scores are inherently noisy.
        * ``'node'`` -- one row per node. ``y`` is the observed node
          score, ``X = [log_flux_src(n), log_flux_tgt(n)]``. This is
          the form used by the original explainer notebooks and
          typically explains a much larger share of the variance.
    drop_unreachable : bool, default True
        If True, rows whose topology features are zero in either
        direction (i.e. unreachable from the source or to the target
        under the chosen random walk / PageRank) are excluded from the
        OLS fit. They still appear in the output, with
        ``topology_score = sigmoid(intercept)`` (the mean prediction).
        This matches the filter used in the original explainer
        notebooks and prevents a large cluster of "off-path" nodes /
        edges from flattening the fitted slope.
    verbose : bool, default False
        If True, print diagnostics about the topology feature and the
        regression fit to stdout (feature range, fitted slope/intercept,
        R^2, and the implied range of ``topology_score``). Useful to
        check *how much* of the observed-score variance is actually
        explained by topology.
    **method_kwargs
        Additional keyword arguments forwarded to the underlying
        topology method (e.g. ``damping=0.85`` for ``'pagerank'``,
        or ``n_steps``, ``n_restarts``, ``seed`` for ``'random_walk'``).

    Notes
    -----
    The ``topology -> expected_score`` map is fit by OLS in logit
    space (when ``logit=True``). For inference (p-values, CIs) one
    might prefer a Binomial / Beta GLM with a logit link, but for
    the purpose of *ranking* edges by their adjusted score this
    makes no practical difference -- the fitted means are nearly
    identical and the residual ordering is preserved.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with two new columns appended (per-edge for
        ``level='edge'``, per-node for ``level='node'``):

        ``topology_score``
            Expected importance from topology alone, on the same scale
            as ``score_col``. Higher means the row (edge or node) is
            more naturally positioned to carry source -> target flow.

        ``topology_adjusted_score``
            ``df[score_col] - topology_score``. The model's learned
            preference for this row, after subtracting what topology
            already explains.
    """
    if level not in {"edge", "node"}:
        raise ValueError(
            f"Unknown level {level!r}. Must be one of 'edge' or 'node'."
        )
    method_table = _TOPOLOGY_NODE_METHODS if level == "node" else _TOPOLOGY_EDGE_METHODS
    if method not in method_table:
        raise ValueError(
            f"Unknown method {method!r} at level {level!r}. "
            f"Available methods: {sorted(method_table)}"
        )

    # Resolve the design-matrix keys and the graph G.
    if level == "edge":
        edges = _as_edge_frame(df, source_col, target_col)
        if G is None:
            G = nx.from_pandas_edgelist(
                edges,
                source=source_col,
                target=target_col,
                create_using=nx.DiGraph(),
            )
        row_ids_src = edges[source_col].to_numpy()
        row_ids_tgt = edges[target_col].to_numpy()
        y = np.asarray(df[score_col].to_numpy(), dtype=float)
    else:  # level == 'node'
        if G is None:
            raise ValueError(
                "level='node' requires `G` to be provided "
                "(the node dataframe doesn't carry edge structure)."
            )
        if node_col not in df.columns:
            raise ValueError(
                f"level='node' requires column {node_col!r} in df. "
                f"Got columns: {list(df.columns)}."
            )
        nodes = df[node_col].to_numpy()
        row_ids_src = nodes
        row_ids_tgt = nodes
        y = np.asarray(df[score_col].to_numpy(), dtype=float)

    # Compute topology features over G. When `drop_unreachable` is
    # True, zero-flux entries get NaN here so the OLS fit below
    # filters them out automatically.
    feature_fn = method_table[method]
    log_src, log_tgt = feature_fn(
        G,
        source_node=source_node,
        target_node=target_node,
        eps=method_kwargs.pop("topology_eps", 1e-12),
        nan_for_zeros=drop_unreachable,
        **method_kwargs,
    )

    # Build the per-row design matrix:
    #   level='edge': [log_edge_flux_src(u,v), log_edge_flux_tgt(u,v)] per edge
    #   level='node': [log_flux_src(n),        log_flux_tgt(n)]        per node
    if level == "edge":
        keys = list(zip(row_ids_src, row_ids_tgt))
        x_src = np.fromiter(
            (log_src.get(k, np.nan) for k in keys),
            dtype=float, count=len(keys),
        )
        x_tgt = np.fromiter(
            (log_tgt.get(k, np.nan) for k in keys),
            dtype=float, count=len(keys),
        )
    else:
        x_src = np.fromiter(
            (log_src.get(u, np.nan) for u in row_ids_src),
            dtype=float, count=len(row_ids_src),
        )
        x_tgt = np.fromiter(
            (log_tgt.get(v, np.nan) for v in row_ids_tgt),
            dtype=float, count=len(row_ids_tgt),
        )
    X = np.column_stack([x_src, x_tgt])

    if logit:
        y_clip = np.clip(y, eps, 1.0 - eps)
        y_t = np.log(y_clip / (1.0 - y_clip))
    else:
        y_t = y

    # Fit only on rows with finite features; predict on all rows.
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_t)
    if not finite_mask.any():
        raise ValueError(
            "No rows have finite topology features and a finite score; "
            "cannot fit the topology regression."
        )
    A_fit = np.column_stack([np.ones(finite_mask.sum()), X[finite_mask]])
    coef, *_ = np.linalg.lstsq(A_fit, y_t[finite_mask], rcond=None)

    # Predict on every row; for rows with non-finite features, fall back
    # to the intercept (= sigmoid(a) on the natural scale).
    X_filled = np.where(np.isfinite(X), X, 0.0)
    yhat_t = coef[0] + X_filled @ coef[1:]
    yhat_t = np.where(np.isfinite(X).all(axis=1), yhat_t, coef[0])

    topology_score = 1.0 / (1.0 + np.exp(-yhat_t)) if logit else yhat_t

    if verbose:
        feature_names = (
            ["log_edge_flux_src", "log_edge_flux_tgt"]
            if level == "edge"
            else ["log_flux_src", "log_flux_tgt"]
        )
        intercept = float(coef[0])
        slopes = [float(c) for c in coef[1:]]
        X_fit_rows = X[finite_mask]
        y_t_fit = y_t[finite_mask]
        y_fit = y[finite_mask]
        yhat_t_fit = A_fit @ coef
        ss_res = float(np.sum((y_t_fit - yhat_t_fit) ** 2))
        ss_tot = float(np.sum((y_t_fit - y_t_fit.mean()) ** 2))
        r2_fit_logit = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        yhat_fit = 1.0 / (1.0 + np.exp(-yhat_t_fit)) if logit else yhat_t_fit
        ss_res_raw = float(np.sum((y_fit - yhat_fit) ** 2))
        ss_tot_raw = float(np.sum((y_fit - y_fit.mean()) ** 2))
        r2_fit_raw = 1.0 - ss_res_raw / ss_tot_raw if ss_tot_raw > 0 else float("nan")
        space = "logit-space" if logit else "raw-space"
        unit_label = "n_nodes" if level == "node" else "n_edges"
        header = (
            f"{unit_label}_fit={int(finite_mask.sum())}  "
            f"{unit_label}_total={len(df)}"
        )
        print(
            f"[adjust_for_topology] method={method!r}  level={level!r}  "
            f"|G|=({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)  {header}"
        )
        for j, name in enumerate(feature_names):
            xj = X_fit_rows[:, j]
            corr = (
                float(np.corrcoef(xj, y_t_fit)[0, 1])
                if np.std(xj) > 0 and np.std(y_t_fit) > 0
                else float("nan")
            )
            print(
                f"  feature[{j}] {name}:  "
                f"min={np.min(xj):.3f}  median={np.median(xj):.3f}  max={np.max(xj):.3f}  "
                f"std={np.std(xj):.3f}  corr(x, y_t)={corr:+.4f}"
            )
        print(
            f"  observed score ({score_col}) at {level} level:  "
            f"min={np.min(y_fit):.4f}  mean={np.mean(y_fit):.4f}  "
            f"max={np.max(y_fit):.4f}  std={np.std(y_fit):.4f}"
        )
        slopes_str = "  ".join(
            f"{name}={s:+.4f}" for name, s in zip(feature_names, slopes)
        )
        print(f"  OLS fit ({space}):  intercept={intercept:+.4f}  {slopes_str}")
        print(
            f"  R^2 at {level} level: logit-space={r2_fit_logit:.4f}  "
            f"raw-space={r2_fit_raw:.4f}"
        )
        print(
            f"  topology_score range:  "
            f"min={np.min(topology_score):.4f}  max={np.max(topology_score):.4f}  "
            f"std={np.std(topology_score):.4f}"
        )
        print(
            f"  topology_adjusted_score range:  "
            f"min={np.min(y - topology_score):.4f}  "
            f"max={np.max(y - topology_score):.4f}  "
            f"std={np.std(y - topology_score):.4f}"
        )
        if r2_fit_logit < 0.05:
            print(
                "  [warning] R^2 < 0.05 at fit level: topology explains very "
                "little of the observed score. Try passing source_node/target_node, "
                "switching `level` ('node' is usually higher-R^2 than 'edge'), "
                "tuning `damping`, or using method='random_walk' with a tighter "
                "n_steps."
            )

    out = df.copy()
    out["topology_score"] = topology_score
    out["topology_adjusted_score"] = y - topology_score
    return out
