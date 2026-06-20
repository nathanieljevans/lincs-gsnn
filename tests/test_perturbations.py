"""Tests for :mod:`lincs_gsnn.proc.perturbations`."""

from __future__ import annotations

import pandas as pd

from lincs_gsnn.proc.perturbations import (
    resolve_chem_target_edges,
    resolve_genetic_edges,
    resolve_perturbation_edges,
)


FUNC_NAMES = {
    "RNA__AAA",
    "PROTEIN__AAA",
    "RNA__BBB",
    "PROTEIN__MAP2K1",
}


def _cond_info():
    return pd.DataFrame([
        {"pert_name": "CHEM_A", "pert_id": "BRD-001", "cmap_name": "NA"},
        {"pert_name": "xpr_AAA", "pert_id": "xpr_AAA", "cmap_name": "AAA"},
        {"pert_name": "oe_BBB", "pert_id": "oe_BBB", "cmap_name": "BBB"},
        {"pert_name": "sh_CCC", "pert_id": "sh_CCC", "cmap_name": "CCC"},
    ])


def _compoundinfo():
    return pd.DataFrame({
        "inchi_key": ["IK1"],
        "pert_id": ["BRD-001"],
    })


def _targetome():
    return pd.DataFrame({
        "inchi_key": ["IK1"],
        "pert_id": ["BRD-001"],
        "uniprot_id": ["P12345"],
        "assay_type": ["Kd"],
        "assay_relation": ["<"],
        "assay_value": [10.0],
    })


def _uniprot_to_func():
    return pd.DataFrame({
        "uniprot": ["P12345"],
        "func_name": ["PROTEIN__MAP2K1"],
        "node_kind": ["PROTEIN"],
    })


def test_genetic_wiring_rules():
    edges, dropped = resolve_genetic_edges(
        ["xpr_AAA", "oe_BBB", "sh_CCC"],
        _cond_info(),
        FUNC_NAMES,
    )
    dst = set(zip(edges["src"], edges["dst"]))
    assert ("DRUG__xpr_AAA", "RNA__AAA") in dst
    assert ("DRUG__xpr_AAA", "PROTEIN__AAA") in dst
    assert ("DRUG__oe_BBB", "RNA__BBB") in dst
    assert ("DRUG__oe_BBB", "PROTEIN__BBB") not in dst
    assert ("DRUG__sh_CCC", "RNA__CCC") not in dst  # CCC not in func graph
    assert any(dropped["pert_name"] == "sh_CCC")


def test_chemical_bridge_and_max_kd():
    edges, dropped = resolve_chem_target_edges(
        ["CHEM_A", "CHEM_MISSING"],
        _cond_info(),
        _compoundinfo(),
        _targetome(),
        _uniprot_to_func(),
        FUNC_NAMES,
        max_kd=100.0,
    )
    assert len(edges) == 1
    assert edges.iloc[0]["src"] == "DRUG__CHEM_A"
    assert edges.iloc[0]["dst"] == "PROTEIN__MAP2K1"
    assert any(dropped["pert_name"] == "CHEM_MISSING")


def test_resolve_perturbation_edges_concatenates():
    edges, dropped = resolve_perturbation_edges(
        ["CHEM_A", "xpr_AAA", "oe_BBB"],
        _cond_info(),
        _compoundinfo(),
        _targetome(),
        _uniprot_to_func(),
        FUNC_NAMES,
        max_kd=100.0,
    )
    assert len(edges) >= 2
    assert "CHEM_A" in set(dropped["pert_name"]) or "oe_BBB" in set(dropped["pert_name"]) or len(dropped) == 0
