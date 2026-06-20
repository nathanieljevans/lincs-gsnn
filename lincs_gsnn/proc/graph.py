"""Directed graph simplification for LINCS-GSNN function-layer networks."""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import torch

logger = logging.getLogger(__name__)

F2F_KEY = ("function", "to", "function")
I2F_KEY = ("input", "to", "function")
F2O_KEY = ("function", "to", "output")

NodeMap = Dict[str, str]
SupernodeNameFn = Callable[[str, Tuple[str, ...]], str]


def parse_gene_symbol(node: str, *, separator: str = "__") -> Optional[str]:
    """Return the gene symbol suffix from a ``KIND__SYMBOL`` node name."""
    if separator not in node:
        return None
    return node.split(separator, 1)[1]


def compose_node_maps(base: NodeMap, update: Mapping[str, str]) -> NodeMap:
    """Chain two node maps: ``base[old]`` then ``update[...]``."""
    return {old: update.get(base[old], base[old]) for old in base}


def apply_node_map(G: nx.DiGraph, node_map: NodeMap) -> nx.DiGraph:
    """Build a simplified digraph by routing edges through ``node_map``."""
    H = nx.DiGraph()
    for node in G.nodes:
        H.add_node(node_map[node])
    for u, v in G.edges():
        mu, mv = node_map[u], node_map[v]
        if mu != mv:
            H.add_edge(mu, mv)
    return H


def combine_gene_symbol_nodes(
    G: nx.DiGraph,
    *,
    separator: str = "__",
    supernode_name: Union[str, SupernodeNameFn] = "symbol",
    min_cluster_size: int = 2,
    max_cluster_size: Optional[int] = None,
    skip_coarsened: bool = True,
) -> Tuple[nx.DiGraph, NodeMap]:
    """Merge function nodes that share a gene symbol into one super-node.

    Nodes named ``PROTEIN__CDK4`` and ``RNA__CDK4`` are combined into a single
    node named ``CDK4`` by default (bare gene symbol).

    Parameters
    ----------
    G
        Directed graph whose node names follow ``KIND__GENE_SYMBOL`` (or contain
        ``separator``).
    separator
        Delimiter between entity kind and gene symbol in node names.
    supernode_name
        How to name each merged group. ``"symbol"`` uses the shared gene symbol.
        A callable receives ``(symbol, members)`` and returns the super-node name.
    min_cluster_size
        Minimum number of nodes sharing a symbol required to form a merge.
    max_cluster_size
        If set, groups larger than this are left unmerged (avoids large super-nodes).
    skip_coarsened
        If True, skip nodes that already look coarsened (contain ``|``).

    Returns
    -------
    G_new
        Simplified graph with merged nodes and deduplicated edges.
    node_map
        ``old_name -> new_name`` for every node in ``G``. Unmerged nodes map to
        themselves; merged nodes map to the super-node name.
    """
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")

    name_fn: SupernodeNameFn
    if supernode_name == "symbol":
        name_fn = lambda symbol, _members: symbol
    elif callable(supernode_name):
        name_fn = supernode_name
    else:
        raise ValueError(
            "supernode_name must be 'symbol' or a callable(symbol, members) -> str"
        )

    groups: dict[str, list[str]] = defaultdict(list)
    for node in G.nodes:
        if skip_coarsened and "|" in node:
            continue
        symbol = parse_gene_symbol(node, separator=separator)
        if symbol is None:
            continue
        groups[symbol].append(node)

    node_map: NodeMap = {node: node for node in G.nodes}
    for symbol, members in groups.items():
        if len(members) < min_cluster_size:
            continue
        if max_cluster_size is not None and len(members) > max_cluster_size:
            continue
        super_node = name_fn(symbol, tuple(sorted(members)))
        for member in members:
            if member in node_map and node_map[member] != member:
                raise ValueError(
                    f"Node {member!r} would belong to multiple gene-symbol groups"
                )
            node_map[member] = super_node

    return apply_node_map(G, node_map), node_map


def contract_degree_one_nodes(
    G: nx.DiGraph,
    *,
    in_degree: int = 1,
    out_degree: int = 1,
) -> Tuple[nx.DiGraph, NodeMap]:
    """Remove pass-through nodes with fixed in- and out-degree.

    Each removed node ``n`` with edges ``u -> n -> v`` is bypassed by ``u -> v``
    (omitted when ``u == v``). Directed reachability among surviving node names
    is preserved.

    Parameters
    ----------
    G
        Input directed graph.
    in_degree, out_degree
        A node is contracted only when ``G.in_degree(n) == in_degree`` and
        ``G.out_degree(n) == out_degree``.

    Returns
    -------
    G_new
        Graph after exhaustive contraction to a fixed point.
    node_map
        ``old_name -> new_name`` for every node in ``G``. Contracted nodes map to
        the survivor they bypass toward; other nodes map to themselves (or a later
        survivor if the chain contracts further).
    """
    G_new = G.copy()
    node_map: NodeMap = {node: node for node in G.nodes}

    changed = True
    while changed:
        changed = False
        for n in list(G_new.nodes):
            if G_new.in_degree(n) != in_degree or G_new.out_degree(n) != out_degree:
                continue

            u = next(G_new.predecessors(n))
            v = next(G_new.successors(n))
            survivor = v if u != v else u

            G_new.remove_node(n)
            if u != v:
                G_new.add_edge(u, v)

            for old in node_map:
                if node_map[old] == n:
                    node_map[old] = survivor
            node_map[n] = survivor
            changed = True

    return G_new, node_map


def reachability_preserved(
    G_before: nx.DiGraph,
    node_map: Mapping[str, str],
    *,
    G_after: Optional[nx.DiGraph] = None,
) -> bool:
    """Check that every directed path in ``G_before`` survives in the quotient graph.

    Suffices to test each original edge ``u -> v`` (not all descendant pairs).
    Reachability in ``G_after`` is evaluated on the SCC condensation DAG, so
    cost is ``O(|E| + |SCC|·(n_scc + m_scc))`` instead of all-pairs on nodes.

    Parameters
    ----------
    G_before
        Graph before simplification.
    node_map
        ``old_name -> new_name`` (e.g. from :func:`compose_node_maps`).
    G_after
        Pre-built quotient graph; computed via :func:`apply_node_map` when omitted.
    """
    if G_after is None:
        G_after = apply_node_map(G_before, node_map)

    if G_after.number_of_nodes() == 0:
        return not G_before.number_of_edges()

    condensation = nx.condensation(G_after)
    scc_of = condensation.graph["mapping"]
    reachable_scc: dict[int, set[int]] = {}
    for scc in condensation.nodes:
        reachable_scc[scc] = nx.descendants(condensation, scc) | {scc}

    for u, v in G_before.edges():
        mu, mv = node_map[u], node_map[v]
        cu, cv = scc_of[mu], scc_of[mv]
        if cu == cv:
            continue
        if cv not in reachable_scc[cu]:
            return False
    return True


def gene_symbol_cluster_sizes(
    G: nx.DiGraph,
    *,
    separator: str = "__",
    skip_coarsened: bool = True,
) -> List[int]:
    """Sorted sizes of gene-symbol duplicate groups (for diagnostics)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for node in G.nodes:
        if skip_coarsened and "|" in node:
            continue
        symbol = parse_gene_symbol(node, separator=separator)
        if symbol is None:
            continue
        groups[symbol].append(node)
    return sorted((len(m) for m in groups.values() if len(m) > 1), reverse=True)


def build_function_digraph(data: Any) -> nx.DiGraph:
    """Build a directed graph of the function layer (all function nodes + f2f edges)."""
    names = list(data.node_names_dict["function"])
    G = nx.DiGraph()
    G.add_nodes_from(names)
    edge_index = data.edge_index_dict[F2F_KEY]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s_ix, d_ix in zip(src, dst):
        G.add_edge(names[s_ix], names[d_ix])
    return G


def _identity_map(names: List[str]) -> NodeMap:
    return {n: n for n in names}


def _dedupe_edge_index(
    src_ix: List[int],
    dst_ix: List[int],
) -> torch.Tensor:
    seen: Set[Tuple[int, int]] = set()
    src_out: List[int] = []
    dst_out: List[int] = []
    for s, d in zip(src_ix, dst_ix):
        key = (s, d)
        if key in seen:
            continue
        seen.add(key)
        src_out.append(s)
        dst_out.append(d)
    if not src_out:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src_out, dst_out], dtype=torch.long)


def _remap_edge_index(
    edge_index: torch.Tensor,
    src_names: List[str],
    dst_names: List[str],
    *,
    remap_src: bool,
    remap_dst: bool,
    node_map: NodeMap,
    new_src_names: Optional[List[str]] = None,
    new_dst_names: Optional[List[str]] = None,
) -> torch.Tensor:
    new_src_idx = (
        {n: i for i, n in enumerate(new_src_names)}
        if new_src_names is not None
        else {n: i for i, n in enumerate(src_names)}
    )
    new_dst_idx = (
        {n: i for i, n in enumerate(new_dst_names)}
        if new_dst_names is not None
        else {n: i for i, n in enumerate(dst_names)}
    )
    src_out: List[int] = []
    dst_out: List[int] = []
    for s_ix, d_ix in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        s_name = src_names[s_ix]
        d_name = dst_names[d_ix]
        if remap_src:
            s_name = node_map.get(s_name, s_name)
            s_ix = new_src_idx[s_name]
        if remap_dst:
            d_name = node_map.get(d_name, d_name)
            d_ix = new_dst_idx[d_name]
        src_out.append(int(s_ix))
        dst_out.append(int(d_ix))
    return _dedupe_edge_index(src_out, dst_out)


def simplify_function_layer(
    data: Any,
    *,
    simplify_degree_one: bool = False,
    degree_one_in_degree: int = 1,
    degree_one_out_degree: int = 1,
    check_reachability: bool = False,
) -> Any:
    """Simplify the function layer of a bionetwork ``HeteroData``-like object in place.

    When enabled, contracts pass-through function nodes with fixed in/out degree
    (see :func:`contract_degree_one_nodes`). Input/output node names are unchanged;
    edges touching function nodes are remapped and deduplicated.

    PROTEIN__/RNA__ nodes are never merged by gene symbol here; use
    :func:`combine_gene_symbol_nodes` offline (e.g. notebooks) if needed.
    """
    if not simplify_degree_one:
        return data

    old_fn = list(data.node_names_dict["function"])
    n_fn_before = len(old_fn)
    n_f2f_before = int(data.edge_index_dict[F2F_KEY].shape[1])

    G = build_function_digraph(data)
    base = _identity_map(old_fn)

    _, map_contract = contract_degree_one_nodes(
        G,
        in_degree=degree_one_in_degree,
        out_degree=degree_one_out_degree,
    )

    map_full = compose_node_maps(base, map_contract)
    new_fn = sorted(set(map_full.values()))
    fn_changed = any(map_full[n] != n for n in old_fn)

    if check_reachability and not reachability_preserved(G, map_full):
        logger.warning(
            "function_graph_simplify: reachability check failed on f2f layer "
            "(n_fn %d -> %d)",
            n_fn_before,
            len(new_fn),
        )

    input_names = list(data.node_names_dict["input"])
    output_names = list(data.node_names_dict["output"])

    data.node_names_dict["function"] = new_fn
    data.edge_index_dict[I2F_KEY] = _remap_edge_index(
        data.edge_index_dict[I2F_KEY],
        input_names,
        old_fn,
        remap_src=False,
        remap_dst=True,
        node_map=map_full,
        new_dst_names=new_fn,
    )
    data.edge_index_dict[F2F_KEY] = _remap_edge_index(
        data.edge_index_dict[F2F_KEY],
        old_fn,
        old_fn,
        remap_src=True,
        remap_dst=True,
        node_map=map_full,
        new_src_names=new_fn,
        new_dst_names=new_fn,
    )
    data.edge_index_dict[F2O_KEY] = _remap_edge_index(
        data.edge_index_dict[F2O_KEY],
        old_fn,
        output_names,
        remap_src=True,
        remap_dst=False,
        node_map=map_full,
        new_src_names=new_fn,
    )

    if fn_changed:
        data.function_node_map = map_full

    n_f2f_after = int(data.edge_index_dict[F2F_KEY].shape[1])
    logger.info(
        "function_graph_simplify: function nodes %d -> %d, f2f edges %d -> %d",
        n_fn_before,
        len(new_fn),
        n_f2f_before,
        n_f2f_after,
    )
    return data


def remap_edge_df(
    df: Any,
    node_map: Optional[Mapping[str, str]],
    cols: Tuple[str, ...] = ("source", "target"),
) -> Any:
    """Return a copy of ``df`` with function-layer endpoint names remapped."""
    if not node_map:
        return df
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda n: map_function_node(str(n), node_map))
    return out


def gene_symbol_from_node(name: str) -> str:
    """Extract gene symbol from ``KIND__SYMBOL`` or return ``name`` if bare."""
    return name.split("__", 1)[1] if "__" in name else name


def protein_to_rna_edge_mask(
    df: Any,
    *,
    function_node_map: Optional[Mapping[str, str]] = None,
) -> Any:
    """Boolean mask for protein-layer → RNA-layer directed candidate edges."""
    src, tgt = df["source"], df["target"]
    if function_node_map is None:
        return src.str.contains("PROTEIN__", na=False) & tgt.str.contains("RNA__", na=False)
    return (~src.str.startswith("RNA__", na=False)) & (~tgt.str.startswith("PROTEIN__", na=False))


def map_function_node(name: str, node_map: Optional[Mapping[str, str]]) -> str:
    """Map a node name through ``function_node_map``; pass-through for input/output kinds."""
    if not node_map:
        return name
    if name.startswith(("DRUG__", "GENE__", "LINE__")):
        return name
    return node_map.get(name, name)


def remap_eval_spec(eval_spec: dict, node_map: Mapping[str, str]) -> dict:
    """Return a copy of an explanation ``eval:`` block with function nodes remapped."""
    spec = copy.deepcopy(eval_spec)
    for row in spec.get("primary_regulators", []) or []:
        row["target_node"] = map_function_node(row["target_node"], node_map)
        row["regulator"] = map_function_node(row["regulator"], node_map)
    for row in spec.get("expected_edges", []) or []:
        row["source"] = map_function_node(row["source"], node_map)
        row["target"] = map_function_node(row["target"], node_map)
    if spec.get("expected_nodes"):
        spec["expected_nodes"] = [
            map_function_node(n, node_map) for n in spec["expected_nodes"]
        ]
    return spec


def remap_removed_edges_names(
    removed_edges: Any,
    node_map: Mapping[str, str],
) -> Any:
    """Remap ``src_name`` / ``dst_name`` columns in a removed-edges table."""
    if removed_edges is None or len(removed_edges) == 0:
        return removed_edges
    out = removed_edges.copy()
    for col in ("src_name", "dst_name"):
        if col in out.columns:
            out[col] = out[col].map(lambda n: map_function_node(str(n), node_map))
    return out
