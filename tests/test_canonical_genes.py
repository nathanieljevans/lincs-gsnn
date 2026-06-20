"""Unit tests for :mod:`lincs_gsnn.proc.canonical_genes`."""

from __future__ import annotations

import pandas as pd
import pytest

from lincs_gsnn.proc.canonical_genes import (
    build_uniprot_to_protein_map,
    resolve_lincs_to_rna_node,
    rna_nodes_by_symbol,
    uniprot_to_func_name,
    uniprot_to_func_names,
)


def _sample_func_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"func_name": "PROTEIN__TP53", "uniprot": "P04637", "gene_symbol": "TP53"},
            {"func_name": "RNA__TP53", "uniprot": "P04637", "gene_symbol": "TP53"},
            {"func_name": "PROTEIN__MAP2K1", "uniprot": "Q02750", "gene_symbol": "MAP2K1"},
            {"func_name": "RNA__MAP2K1", "uniprot": "Q02750", "gene_symbol": "MAP2K1"},
            {"func_name": "PROTEIN__OLD1", "uniprot": "P04637", "gene_symbol": "OLD1"},
            {"func_name": "RNA__ALIAS2", "uniprot": "Q99999", "gene_symbol": "ALIAS2"},
        ]
    )


def test_uniprot_to_func_names_includes_ambiguous():
    func_nodes = _sample_func_nodes()
    mapping, ambiguous = uniprot_to_func_names(func_nodes, prefix="PROTEIN__")
    assert mapping["P04637"] == ["PROTEIN__OLD1", "PROTEIN__TP53"]
    assert mapping["Q02750"] == ["PROTEIN__MAP2K1"]
    assert len(ambiguous) == 1
    assert ambiguous[0]["uniprot"] == "P04637"
    assert set(ambiguous[0]["func_names"]) == {"PROTEIN__TP53", "PROTEIN__OLD1"}


def test_uniprot_to_func_name_unambiguous_only():
    func_nodes = _sample_func_nodes()
    mapping, ambiguous = uniprot_to_func_name(func_nodes, prefix="PROTEIN__")
    assert "P04637" not in mapping
    assert mapping["Q02750"] == "PROTEIN__MAP2K1"
    assert len(ambiguous) == 1


def test_build_uniprot_to_protein_map_multi_target():
    func_nodes = _sample_func_nodes()
    mapping, ambiguous = build_uniprot_to_protein_map(func_nodes)
    assert mapping["Q02750"] == ["PROTEIN__MAP2K1"]
    assert set(mapping["P04637"]) == {"PROTEIN__TP53", "PROTEIN__OLD1"}
    assert ambiguous[0]["uniprot"] == "P04637"


def test_dti_edges_explode_ambiguous_uniprot():
    mapping = {"P04637": ["PROTEIN__TP53", "PROTEIN__OLD1"], "Q02750": ["PROTEIN__MAP2K1"]}
    tge = pd.DataFrame(
        {"pert_id": ["drug-a", "drug-b"], "uniprot_id": ["P04637", "Q02750"]},
    )
    tge["dst"] = tge["uniprot_id"].map(mapping)
    tge["src"] = "DRUG__" + tge["pert_id"]
    edges = (
        tge[["src", "dst"]]
        .explode("dst")
        .dropna(subset=["dst"])
        .drop_duplicates()
    )
    assert len(edges) == 3
    assert set(edges[edges.src == "DRUG__drug-a"].dst) == {"PROTEIN__TP53", "PROTEIN__OLD1"}


def test_resolve_direct_symbol():
    func_nodes = _sample_func_nodes()
    row = {"lincs_symbol": "TP53", "aliases": "TP53", "uniprot_ids": "P04637"}
    node, method = resolve_lincs_to_rna_node("TP53", row, func_nodes)
    assert node == "RNA__TP53"
    assert method == "direct_symbol"


def test_resolve_uniprot_protein_bridge():
    func_nodes = _sample_func_nodes()
    row = {"lincs_symbol": "MAP2K1", "aliases": "MAP2K1", "uniprot_ids": "Q02750"}
    node, method = resolve_lincs_to_rna_node("MAP2K1", row, func_nodes)
    assert node == "RNA__MAP2K1"
    assert method == "direct_symbol"


def test_resolve_alias_symbol():
    func_nodes = _sample_func_nodes()
    row = {"lincs_symbol": "NEWNAME", "aliases": "ALIAS2;NEWNAME", "uniprot_ids": ""}
    node, method = resolve_lincs_to_rna_node("NEWNAME", row, func_nodes)
    assert node == "RNA__ALIAS2"
    assert method == "alias_symbol"


def test_rna_nodes_by_symbol_uppercase():
    func_nodes = _sample_func_nodes()
    by_sym = rna_nodes_by_symbol(func_nodes)
    assert by_sym["TP53"] == "RNA__TP53"
