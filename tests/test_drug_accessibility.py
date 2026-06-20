"""Unit tests for :mod:`lincs_gsnn.proc.drug_accessibility`."""

from __future__ import annotations

import types

import pytest
import torch

from lincs_gsnn.proc.drug_accessibility import (
    accessible_gene_slice_mask,
    accessible_indices,
    bionetwork_to_digraph,
    compute_drug_accessible_output_genes,
    get_or_compute_drug_accessible_mask,
)


def _build_reachable_bionet():
    """DRUG__D -> PROTEIN__P -> RNA__Y -> GENE__Y (accessible)."""
    node_names_dict = {
        "input": ["LINE__A", "DRUG__D", "GENE__Y", "GENE__Z"],
        "function": ["PROTEIN__P", "RNA__Y", "RNA__Z"],
        "output": ["GENE__Y", "GENE__Z"],
    }
    input_to_fn = torch.tensor([[1], [0]], dtype=torch.long)
    fn_to_fn = torch.tensor([[0], [1]], dtype=torch.long)
    fn_to_out = torch.tensor([[1, 2], [0, 1]], dtype=torch.long)
    edge_index_dict = {
        ("input", "to", "function"): input_to_fn,
        ("function", "to", "function"): fn_to_fn,
        ("function", "to", "output"): fn_to_out,
    }
    return types.SimpleNamespace(
        node_names_dict=node_names_dict,
        edge_index_dict=edge_index_dict,
    )


def _build_line_only_bionet():
    """LINE__A -> RNA__Y exists, but no drug path to GENE__Y."""
    node_names_dict = {
        "input": ["LINE__A", "DRUG__D"],
        "function": ["RNA__Y"],
        "output": ["GENE__Y"],
    }
    edge_index_dict = {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.tensor([[], []], dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0], [0]], dtype=torch.long),
    }
    return types.SimpleNamespace(
        node_names_dict=node_names_dict,
        edge_index_dict=edge_index_dict,
    )


def test_drug_reachable_output_is_accessible():
    data = _build_reachable_bionet()
    mask = compute_drug_accessible_output_genes(data)
    assert mask.tolist() == [True, False]


def test_output_without_drug_path_is_inaccessible():
    data = _build_reachable_bionet()
    mask = compute_drug_accessible_output_genes(data)
    assert not mask[1].item()


def test_line_edges_do_not_make_outputs_accessible():
    data = _build_line_only_bionet()
    mask = compute_drug_accessible_output_genes(data)
    assert mask.tolist() == [False]


def test_bionetwork_to_digraph_excludes_line_edges():
    data = _build_line_only_bionet()
    G = bionetwork_to_digraph(data, exclude_line_edges=True)
    assert ("LINE__A", "RNA__Y") not in G.edges()


def test_get_or_compute_uses_cached_mask():
    data = _build_reachable_bionet()
    data.drug_accessible_output_genes = torch.tensor([True, False], dtype=torch.bool)
    mask = get_or_compute_drug_accessible_mask(data)
    assert mask.tolist() == [True, False]


def test_accessible_indices_raises_when_empty():
    with pytest.raises(ValueError, match="no output genes are reachable"):
        accessible_indices(torch.tensor([False, False], dtype=torch.bool))


def test_accessible_gene_slice_mask_maps_by_gene_symbol():
    output_mask = torch.tensor([True, False], dtype=torch.bool)
    output_names = ["GENE__Y", "GENE__Z"]
    input_names = ["LINE__A", "DRUG__D", "GENE__Z", "GENE__Y"]
    gene_ixs = torch.tensor([2, 3], dtype=torch.long)

    slice_mask = accessible_gene_slice_mask(
        output_mask, output_names, input_names, gene_ixs
    )
    assert slice_mask.tolist() == [False, True]


def test_accessible_indices_returns_reachable_positions():
    mask = torch.tensor([False, True, True], dtype=torch.bool)
    ix = accessible_indices(mask)
    assert ix.tolist() == [1, 2]
