"""Unit tests for :mod:`lincs_gsnn.proc.graph` bionetwork simplification."""

from __future__ import annotations

import types

import torch

from lincs_gsnn.proc.graph import (
    build_function_digraph,
    contract_degree_one_nodes,
    map_function_node,
    reachability_preserved,
    remap_eval_spec,
    simplify_function_layer,
)


def _toy_bionet_with_pass_through():
    """PROTEIN__A -> MID -> PROTEIN__B; MID is contracted away."""
    node_names_dict = {
        "input": ["DRUG__D"],
        "function": ["PROTEIN__A", "MID", "PROTEIN__B"],
        "output": ["GENE__A", "GENE__B"],
    }
    edge_index_dict = {
        ("input", "to", "function"): torch.tensor([[0], [0]], dtype=torch.long),
        ("function", "to", "function"): torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        ("function", "to", "output"): torch.tensor([[0, 2], [0, 1]], dtype=torch.long),
    }
    return types.SimpleNamespace(
        node_names_dict=node_names_dict,
        edge_index_dict=edge_index_dict,
    )


def test_simplify_contracts_degree_one_pass_through():
    data = _toy_bionet_with_pass_through()
    simplify_function_layer(data, simplify_degree_one=True)
    fn = data.node_names_dict["function"]
    assert "MID" not in fn
    assert "PROTEIN__A" in fn
    assert "PROTEIN__B" in fn
    assert hasattr(data, "function_node_map")
    assert data.function_node_map["MID"] == "PROTEIN__B"
    f2f = data.edge_index_dict[("function", "to", "function")]
    names = fn
    edges = {(names[s], names[d]) for s, d in zip(f2f[0].tolist(), f2f[1].tolist())}
    assert ("PROTEIN__A", "PROTEIN__B") in edges


def test_remap_eval_spec_maps_function_nodes():
    node_map = {"PROTEIN__MAPK3": "MAPK3", "RNA__DUSP6": "DUSP6"}
    spec = {
        "primary_regulators": [
            {"target_node": "RNA__DUSP6", "regulator": "PROTEIN__ETS1"},
        ],
        "expected_edges": [
            {"source": "PROTEIN__MAP2K1", "target": "PROTEIN__MAPK3"},
            {"source": "DRUG__BRD-X", "target": "PROTEIN__MAP2K1"},
        ],
        "expected_nodes": ["PROTEIN__MAPK3", "GENE__DUSP6", "DRUG__BRD-X"],
        "expected_paths": ["MAP2K1 -> MAPK3 -> ETS1 -> DUSP6"],
    }
    out = remap_eval_spec(spec, node_map)
    assert out["primary_regulators"][0]["target_node"] == "DUSP6"
    assert out["primary_regulators"][0]["regulator"] == "PROTEIN__ETS1"
    assert out["expected_edges"][0]["target"] == "MAPK3"
    assert out["expected_edges"][1]["source"] == "DRUG__BRD-X"
    assert out["expected_nodes"][0] == "MAPK3"
    assert out["expected_nodes"][1] == "GENE__DUSP6"
    assert out["expected_paths"] == spec["expected_paths"]


def test_map_function_node_pass_through():
    node_map = {"PROTEIN__X": "X"}
    assert map_function_node("GENE__Y", node_map) == "GENE__Y"
    assert map_function_node("DRUG__D", node_map) == "DRUG__D"
    assert map_function_node("PROTEIN__X", node_map) == "X"


def test_reachability_preserved_after_degree_one_contract():
    data = _toy_bionet_with_pass_through()
    G = build_function_digraph(data)
    G2, node_map = contract_degree_one_nodes(G)
    assert reachability_preserved(G, node_map, G_after=G2)


def test_simplify_noop_when_disabled():
    data = _toy_bionet_with_pass_through()
    before_fn = list(data.node_names_dict["function"])
    simplify_function_layer(data, simplify_degree_one=False)
    assert list(data.node_names_dict["function"]) == before_fn
    assert not hasattr(data, "function_node_map")
