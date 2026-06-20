"""Unit tests for :mod:`lincs_gsnn.proc.node_activity`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from lincs_gsnn.proc import node_activity as na


def _minimal_node_names_dict():
    return {
        "input": ["LINE__A", "LINE__B", "DRUG__D"],
        "function": ["PROTEIN__TP53", "RNA__TP53"],
        "output": ["GENE__TP53"],
    }


def test_build_cell_line_activity_tensor_shape_and_onehot():
    line_vocab = ["A", "B"]
    t = na.build_cell_line_activity_tensor(
        cell_inames=["A", "B"],
        line_vocab=line_vocab,
        n_function_nodes=3,
    )
    assert t.shape == (2, 3, 2)
    assert t[0, 0, 0] == 1.0 and t[0, 0, 1] == 0.0
    assert t[1, 2, 0] == 0.0 and t[1, 2, 1] == 1.0


def test_build_x_fn_cell_line_only_skips_depmap_and_expr_mut():
    node_names_dict = _minimal_node_names_dict()

    with patch.object(
        na, "build_cell_iname_to_modelid_map"
    ) as mock_map, patch.object(
        na, "create_node_activity_inputs"
    ) as mock_expr, patch.object(
        na, "create_damaging_mutation_inputs"
    ) as mock_mut:
        x_fn_by_ciname, metadata = na.build_x_fn_lookup_from_bionet(
            node_names_dict=node_names_dict,
            data_root="/unused",
            features=["cell_line"],
        )

    mock_map.assert_not_called()
    mock_expr.assert_not_called()
    mock_mut.assert_not_called()

    assert set(x_fn_by_ciname.keys()) == {"A", "B"}
    assert metadata["activity_dim"] == 2
    assert metadata["activity_features"] == ["cell_line"]
    for iname, row in x_fn_by_ciname.items():
        assert row.shape == (2, 2)  # n_fn, n_LINE__
        assert row.sum().item() == 2.0  # one-hot broadcast across fn nodes


def test_build_x_fn_is_protein_only_skips_depmap_and_expr_mut():
    node_names_dict = _minimal_node_names_dict()

    with patch.object(
        na, "build_cell_iname_to_modelid_map"
    ) as mock_map, patch.object(
        na, "create_node_activity_inputs"
    ) as mock_expr, patch.object(
        na, "create_damaging_mutation_inputs"
    ) as mock_mut:
        x_fn_by_ciname, metadata = na.build_x_fn_lookup_from_bionet(
            node_names_dict=node_names_dict,
            data_root="/unused",
            features=["is_protein"],
        )

    mock_map.assert_not_called()
    mock_expr.assert_not_called()
    mock_mut.assert_not_called()
    assert metadata["activity_dim"] == 1
    assert x_fn_by_ciname["A"][0, 0] == 1.0  # PROTEIN__TP53


def test_cell_line_in_activity_feature_builders():
    assert "cell_line" in na.ACTIVITY_FEATURE_BUILDERS
    assert na.ACTIVITY_FEATURE_BUILDERS["cell_line"][0] is None
