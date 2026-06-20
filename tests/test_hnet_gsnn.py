"""Smoke tests for the cell-line-conditioned hypernetwork helpers in
:mod:`lincs_gsnn.models.HnetGSNN`.

Run from the repo root with the ``lincs-gsnn`` conda env active:

    conda activate lincs-gsnn
    pytest -q tests/test_hnet_gsnn.py

These tests deliberately use a tiny synthetic graph so the entire test
file runs in a few seconds on CPU.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import types

import pytest
import torch

# ----------------------------------------------------------------------------
# Make the workflow scripts importable as modules without invoking their
# ``__main__`` block, so we can introspect their argparse default values.
# ----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "workflow", "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ----------------------------------------------------------------------------
# Tiny synthetic bionetwork-like object the helpers can consume.
# ----------------------------------------------------------------------------
def _build_tiny_bionet():
    """Return a namespace with the minimal attributes the helpers and GSNN
    constructor read: ``edge_index_dict`` and ``node_names_dict``."""
    node_names_dict = {
        "input":    ["LINE__A", "LINE__B", "LINE__C", "GENE__X", "DRUG__Z"],
        "function": ["F1", "F2", "F3"],
        "output":   ["GENE__Y"],
    }
    edge_index_dict = {
        # input -> function
        ("input", "to", "function"): torch.tensor(
            [[0, 1, 2, 3, 4],
             [0, 1, 2, 1, 0]],
            dtype=torch.long,
        ),
        # function -> function
        ("function", "to", "function"): torch.tensor(
            [[0, 1, 2],
             [1, 2, 0]],
            dtype=torch.long,
        ),
        # function -> output
        ("function", "to", "output"): torch.tensor(
            [[2],
             [0]],
            dtype=torch.long,
        ),
    }
    return types.SimpleNamespace(
        node_names_dict=node_names_dict,
        edge_index_dict=edge_index_dict,
    )


_GSNN_KWARGS = dict(
    channels=2,
    layers=2,
    share_layers=True,
    dropout=0.0,
    checkpoint=False,
    init="xavier_normal",
    norm="rms",
    add_function_self_edges=True,
    bias=False,
    residual=True,
    node_mlp=False,
    node_mlp_hidden=4,
    node_attn=False,
    attn_mlp_hidden=4,
)


def _make_gsnn(data):
    from lincs_gsnn.models.HnetGSNN import build_gsnn_template
    return build_gsnn_template(data, copy.deepcopy(_GSNN_KWARGS))


def _make_hnet(gsnn, n_cells, **overrides):
    from lincs_gsnn.models.HnetGSNN import build_hnet
    cfg = dict(
        stochastic_channels=2,
        width=4,
        pz="normal",
        learn_pz=False,
        affine=False,
        norm="none",
        dropout=0.0,
        bias=False,
    )
    cfg.update(overrides)
    return build_hnet(gsnn, n_cells, cfg)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_build_and_sample():
    """``hnet.sample(C, z=zeros)`` returns a state-dict whose keys/shapes
    exactly match the underlying GSNN's named parameters."""
    from lincs_gsnn.models.HnetGSNN import cell_lines_from_bionet, cell_onehot

    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = cell_lines_from_bionet(data)
    assert cells == ["A", "B", "C"]

    hnet = _make_hnet(gsnn, n_cells=len(cells))

    C = cell_onehot("A", cells)
    z = torch.zeros_like(hnet.mu)
    state_dict = hnet.sample(C=C, z=z)

    expected_names = {n for n, _ in gsnn.named_parameters()}
    assert set(state_dict.keys()) == expected_names
    for name, p in gsnn.named_parameters():
        assert state_dict[name].shape == p.shape, name


def test_functional_call_forward():
    """Materialized theta + functional_call yields a (B, n_out) tensor and
    one MSE backward + optimizer step runs without error."""
    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = ["A", "B", "C"]
    hnet = _make_hnet(gsnn, n_cells=len(cells))

    from lincs_gsnn.models.HnetGSNN import cell_onehot

    n_in = len(data.node_names_dict["input"])
    n_out = len(data.node_names_dict["output"])

    optim = torch.optim.AdamW(hnet.parameters(), lr=1e-3)
    optim.zero_grad()

    C = cell_onehot("A", cells)
    z = hnet._sample_z()
    state_dict = hnet.sample(C=C, z=z)

    x = torch.randn(4, n_in)
    y = torch.randn(4, n_out)

    yhat = torch.func.functional_call(hnet.model, state_dict, x)
    assert yhat.shape == (4, n_out)
    assert yhat.requires_grad

    loss = torch.nn.functional.mse_loss(yhat, y)
    loss.backward()
    # Verify at least some leaf hnet params received gradients.
    grad_count = sum(
        int(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for p in hnet.parameters()
        if p.is_leaf
    )
    assert grad_count > 0
    optim.step()


def test_materialize_gsnn_conditioning_matters():
    """Materializing different cells with the SAME z yields different theta
    (and therefore generally different forward outputs). Materializing the
    same cell twice yields identical state dicts."""
    from lincs_gsnn.models.HnetGSNN import cell_onehot, materialize_gsnn

    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = ["A", "B", "C"]
    hnet = _make_hnet(gsnn, n_cells=len(cells))
    hnet.eval()

    n_in = len(data.node_names_dict["input"])
    x = torch.randn(2, n_in)

    Ca = cell_onehot("A", cells)
    Cb = cell_onehot("B", cells)
    z = torch.zeros_like(hnet.mu)

    with torch.no_grad():
        ma = materialize_gsnn(hnet, Ca, z=z).eval()
        mb = materialize_gsnn(hnet, Cb, z=z).eval()
        ma2 = materialize_gsnn(hnet, Ca, z=z).eval()

        ya = ma(x)
        yb = mb(x)
        ya2 = ma2(x)

    # Same cell + same z => identical thetas => identical outputs.
    assert torch.allclose(ya, ya2, atol=1e-6), "deterministic re-materialization failed"
    # Different cells should produce at least some difference in thetas.
    diff = sum(
        (pa - pb).abs().sum().item()
        for (na, pa), (nb, pb) in zip(ma.named_parameters(), mb.named_parameters())
    )
    assert diff > 0.0, "cell conditioning had no effect on theta"


def test_cell_line_router_dispatch():
    """The router dispatches to m1/m2 by data_ptr when given the registered
    references, falls through to half-batch dispatch on a cat-batch, and
    raises on inputs that match neither pattern."""
    from lincs_gsnn.models.HnetGSNN import (
        CellLineRouter,
        cell_onehot,
        materialize_gsnn,
    )

    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = ["A", "B", "C"]
    hnet = _make_hnet(gsnn, n_cells=len(cells))
    hnet.eval()

    n_in = len(data.node_names_dict["input"])
    x1 = torch.randn(1, n_in)
    x2 = torch.randn(1, n_in)
    x_cat = torch.cat([x1.repeat(3, 1), x2.repeat(3, 1)], dim=0)  # (6, n_in)

    z = torch.zeros_like(hnet.mu)
    with torch.no_grad():
        m1 = materialize_gsnn(hnet, cell_onehot("A", cells), z=z).eval()
        m2 = materialize_gsnn(hnet, cell_onehot("B", cells), z=z).eval()

    router = CellLineRouter(m1, m2, x1, x2)

    with torch.no_grad():
        # Identity dispatch (data_ptr matches).
        y1_router = router(x1)
        y2_router = router(x2)
        y1_direct = m1(x1)
        y2_direct = m2(x2)
        assert torch.allclose(y1_router, y1_direct, atol=1e-6)
        assert torch.allclose(y2_router, y2_direct, atol=1e-6)

        # Half-batch dispatch (cat shape, even, >= 2, no data_ptr match).
        y_cat = router(x_cat)
        y_cat_expected = torch.cat(
            [m1(x_cat[:3]), m2(x_cat[3:])], dim=0,
        )
        assert torch.allclose(y_cat, y_cat_expected, atol=1e-6)

    # Unrecognized dispatch (odd batch size, no id match) must raise.
    with pytest.raises(RuntimeError):
        router(torch.randn(3, n_in))


def test_cell_line_router_survives_deepcopy_and_to_device():
    """The contrastive explainers internally ``copy.deepcopy(model).to(device)``;
    the router's data_ptr-based dispatch must continue to work afterwards."""
    from lincs_gsnn.models.HnetGSNN import (
        CellLineRouter,
        cell_onehot,
        materialize_gsnn,
    )

    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = ["A", "B", "C"]
    hnet = _make_hnet(gsnn, n_cells=len(cells))
    hnet.eval()

    n_in = len(data.node_names_dict["input"])
    x1 = torch.randn(1, n_in)
    x2 = torch.randn(1, n_in)

    z = torch.zeros_like(hnet.mu)
    with torch.no_grad():
        m1 = materialize_gsnn(hnet, cell_onehot("A", cells), z=z).eval()
        m2 = materialize_gsnn(hnet, cell_onehot("B", cells), z=z).eval()
    router = CellLineRouter(m1, m2, x1, x2)

    router_copy = copy.deepcopy(router)
    # Same device (cpu) -> .to is a no-op for tensors -> data_ptrs preserved.
    router_copy = router_copy.to("cpu")

    with torch.no_grad():
        # Calls on x1/x2 still dispatch correctly through the deepcopy.
        assert torch.allclose(router_copy(x1), router(x1), atol=1e-6)
        assert torch.allclose(router_copy(x2), router(x2), atol=1e-6)


def test_artifact_roundtrip(tmp_path):
    """``save_hnet_artifact`` -> ``load_hnet_artifact`` reproduces a hnet that
    samples the same theta given the same (C, z)."""
    from lincs_gsnn.models.HnetGSNN import (
        cell_lines_from_bionet,
        cell_onehot,
        load_hnet_artifact,
        save_hnet_artifact,
    )

    data = _build_tiny_bionet()
    gsnn = _make_gsnn(data)
    cells = cell_lines_from_bionet(data)
    hnet_cfg = dict(
        stochastic_channels=2, width=4, pz="normal", learn_pz=False,
        affine=False, norm="none", dropout=0.0, bias=False,
    )
    hnet = _make_hnet(gsnn, n_cells=len(cells), **hnet_cfg)
    hnet.eval()

    # Strip bionetwork dicts, mirroring the real save path in pretrain.
    gsnn_kwargs_serializable = dict(_GSNN_KWARGS)

    path = str(tmp_path / "hnet.pt")
    save_hnet_artifact(path, hnet, cells, gsnn_kwargs_serializable, hnet_cfg)
    loaded = load_hnet_artifact(path, data)

    hnet2 = loaded["hnet"].eval()
    C = cell_onehot("B", cells)
    z = torch.zeros_like(hnet.mu)
    with torch.no_grad():
        sd1 = hnet.sample(C=C, z=z)
        sd2 = hnet2.sample(C=C, z=z)
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k], atol=1e-6), k


def test_pretrain_argparser_legacy_defaults_unchanged():
    """The pretrain script's argparser must produce the same defaults for
    the legacy fields when --use_hypernetwork is NOT passed. This guards
    against accidentally changing legacy behavior while adding the
    hypernetwork knobs."""
    pretrain = __import__("pretrain_gsnn_with_dxdt")
    parser = pretrain.get_args.__wrapped__ if hasattr(pretrain.get_args, "__wrapped__") else None
    # get_args() calls parse_args(); we instead build a parser by introspection
    # via the module's argparse.ArgumentParser used inside get_args. Easiest:
    # call get_args() with patched sys.argv that supplies only the required
    # --model_id, then inspect the resulting Namespace.
    old_argv = sys.argv
    try:
        sys.argv = ["pretrain_gsnn_with_dxdt.py", "--model_id", "model_0"]
        ns = pretrain.get_args()
    finally:
        sys.argv = old_argv

    # Hypernetwork is OFF by default.
    assert ns.use_hypernetwork is False

    # A snapshot of legacy defaults that pre-existed before the hypernet
    # changes. If a future edit changes any of these, this test will fail
    # loudly so the change can be reviewed deliberately.
    legacy_expected = {
        "batch_size": 64,
        "num_workers": 4,
        "epochs": 100,
        "lr": 1e-3,
        "wd": 1e-4,
        "patience": 10,
        "channels": 64,
        "layers": 3,
        "share_layers": False,
        "dropout": 0.1,
        "norm": "batch",
        "checkpoint": False,
        "init": "degree_normalized",
        "add_function_self_edges": True,
        "bias": True,
        "residual": True,
        "node_mlp": True,
        "node_mlp_dim": 128,
        "node_attn": False,
        "attn_mlp_hidden": 32,
    }
    for k, v in legacy_expected.items():
        assert getattr(ns, k) == v, f"legacy default for --{k} changed: got {getattr(ns, k)!r}, expected {v!r}"
