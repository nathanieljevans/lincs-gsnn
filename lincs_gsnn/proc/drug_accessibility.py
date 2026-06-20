"""Drug-to-output reachability helpers for the LINCS-GSNN bionetwork.

A landmark output gene is *drug-accessible* when at least one ``DRUG__*`` input
node has a directed path to that ``GENE__*`` output node through the saved
heterogeneous graph (input→function, function→function, function→output).
Cell-line ``LINE__*`` edges are excluded because they do not represent drug
perturbation routes.

The boolean mask is stored on ``bionetwork.pt`` as
``data.drug_accessible_output_genes`` and consumed by pretrain/train scripts
to restrict loss and metrics to reachable genes.
"""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence, Union

import networkx as nx
import torch

_LINE_PREFIX = "LINE__"
_DRUG_PREFIX = "DRUG__"
_GENE_PREFIX = "GENE__"


def _edge_pairs(
    edge_index: torch.Tensor,
    src_names: Sequence[str],
    dst_names: Sequence[str],
) -> list[tuple[str, str]]:
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    return [(src_names[i], dst_names[j]) for i, j in zip(src, dst)]


def bionetwork_to_digraph(data, exclude_line_edges: bool = True) -> nx.DiGraph:
    """Flatten a bionetwork heterograph into a NetworkX ``DiGraph``.

    Parameters
    ----------
    data
        Object with ``node_names_dict`` and ``edge_index_dict`` (as produced by
        ``make_bio_network.py``).
    exclude_line_edges
        When True (default), drop edges whose source or target is a ``LINE__*``
        node.
    """
    node_names_dict = data.node_names_dict
    edge_index_dict = data.edge_index_dict

    input_names = list(node_names_dict["input"])
    function_names = list(node_names_dict["function"])
    output_names = list(node_names_dict["output"])

    pairs: list[tuple[str, str]] = []
    pairs.extend(
        _edge_pairs(
            edge_index_dict[("input", "to", "function")],
            input_names,
            function_names,
        )
    )
    pairs.extend(
        _edge_pairs(
            edge_index_dict[("function", "to", "function")],
            function_names,
            function_names,
        )
    )
    pairs.extend(
        _edge_pairs(
            edge_index_dict[("function", "to", "output")],
            function_names,
            output_names,
        )
    )

    if exclude_line_edges:
        pairs = [
            (s, t)
            for s, t in pairs
            if not s.startswith(_LINE_PREFIX) and not t.startswith(_LINE_PREFIX)
        ]

    return nx.from_edgelist(pairs, create_using=nx.DiGraph())


def _multi_source_reachable(G: nx.DiGraph, sources: Sequence[str]) -> set[str]:
    """Return all nodes reachable from any node in ``sources``."""
    reachable: set[str] = set()
    queue: deque[str] = deque()

    for src in sources:
        if src not in G:
            continue
        if src not in reachable:
            reachable.add(src)
            queue.append(src)

    while queue:
        node = queue.popleft()
        for nbr in G.successors(node):
            if nbr not in reachable:
                reachable.add(nbr)
                queue.append(nbr)

    return reachable


def compute_drug_accessible_output_genes(data) -> torch.Tensor:
    """Compute a boolean mask over ``node_names_dict['output']``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_outputs,)``, dtype ``torch.bool``. Entry ``i`` is True when
        ``output_names[i]`` is reachable from at least one ``DRUG__*`` node.
    """
    G = bionetwork_to_digraph(data, exclude_line_edges=True)
    drug_nodes = [
        n for n in data.node_names_dict["input"] if str(n).startswith(_DRUG_PREFIX)
    ]
    reachable = _multi_source_reachable(G, drug_nodes)
    output_names = data.node_names_dict["output"]
    return torch.tensor(
        [out in reachable for out in output_names],
        dtype=torch.bool,
    )


def get_or_compute_drug_accessible_mask(data) -> torch.Tensor:
    """Return a cached mask on ``data`` or compute it for legacy artifacts."""
    mask = getattr(data, "drug_accessible_output_genes", None)
    if mask is not None:
        return torch.as_tensor(mask, dtype=torch.bool)
    return compute_drug_accessible_output_genes(data)


def accessible_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return integer indices where ``mask`` is True."""
    ix = mask.nonzero(as_tuple=True)[0]
    if ix.numel() == 0:
        raise ValueError(
            "drug_accessible_output_genes: no output genes are reachable from "
            "any drug node; cannot train or evaluate on an empty gene subset."
        )
    return ix


def accessible_gene_slice_mask(
    output_mask: torch.Tensor,
    output_names: Sequence[str],
    input_names: Sequence[str],
    gene_ixs: Union[torch.Tensor, Sequence[int]],
) -> torch.Tensor:
    """Map an output-order accessibility mask to the gene trajectory slice.

    ``gene_ixs`` lists positions in ``input_names`` for ``GENE__*`` nodes (as
    used by :class:`lincs_gsnn.models.ODEFunc.ODEFunc`). The returned mask has
    length ``len(gene_ixs)`` and is aligned with the gene dimension of
    trajectory tensors in odeint training.
    """
    if isinstance(gene_ixs, torch.Tensor):
        gene_ix_list = gene_ixs.tolist()
    else:
        gene_ix_list = list(gene_ixs)

    symbol_to_accessible = {
        name.split("__", 1)[1]: bool(output_mask[i].item())
        for i, name in enumerate(output_names)
        if str(name).startswith(_GENE_PREFIX)
    }

    return torch.tensor(
        [
            symbol_to_accessible[input_names[ix].split("__", 1)[1]]
            for ix in gene_ix_list
        ],
        dtype=torch.bool,
    )
