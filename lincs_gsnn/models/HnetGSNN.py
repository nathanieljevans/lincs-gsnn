"""Cell-line-conditioned hypernetwork helpers for the GSNN pipeline.

This module is *only* imported by the train/explain scripts when the user
opts into hypernetwork training via ``hypernetwork.enabled = true`` in the
workflow config. Everything here is a thin layer on top of the
:mod:`hnet.models.HyperNet` package living at
``/home/exacloud/gscratch/mcweeney_lab/evans/hypernet`` plus the
:class:`gsnn.models.GSNN.GSNN` base model.

The default training/eval path deliberately avoids :func:`torch.func.vmap`:
weights are sampled with :meth:`hnet.models.HyperNet.HyperNet.sample` and
applied via :func:`torch.func.functional_call`. ``vmap`` is only used when the
caller explicitly requests ensemble training (``n_train_samples > 1``).
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional

import torch

from gsnn.models.GSNN import GSNN
from hnet.models.HyperNet import HyperNet


_LINE_PREFIX = "LINE__"


def cell_lines_from_bionet(data) -> List[str]:
    """Extract the canonical cell-line vocabulary from a bionetwork object.

    The order of the returned list is the order of ``LINE__*`` entries in
    ``data.node_names_dict['input']`` and is therefore stable across
    train/explain so long as the same bionetwork artifact is used.
    """
    return [
        n[len(_LINE_PREFIX):]
        for n in data.node_names_dict["input"]
        if n.startswith(_LINE_PREFIX)
    ]


def cell_onehot(
    cell_iname: str,
    cell_lines: Iterable[str],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a one-hot ``Tensor[len(cell_lines)]`` for ``cell_iname``."""
    cell_lines = list(cell_lines)
    try:
        idx = cell_lines.index(cell_iname)
    except ValueError as e:
        raise ValueError(
            f"cell_iname={cell_iname!r} not found in cell_lines vocabulary "
            f"(size={len(cell_lines)})."
        ) from e
    C = torch.zeros(len(cell_lines), dtype=dtype)
    C[idx] = 1.0
    if device is not None:
        C = C.to(device)
    return C


def soft_mean_C(
    cell_lines: Iterable[str],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform-weighted soft-C across all cell lines.

    Used to materialize a "mean cell line" GSNN as a backward-compatibility
    artifact so legacy code that simply ``torch.load``s
    ``pretrained_model_{sample}.pt`` still gets a usable, deterministic GSNN.
    """
    cell_lines = list(cell_lines)
    n = len(cell_lines)
    C = torch.full((n,), 1.0 / max(n, 1), dtype=dtype)
    if device is not None:
        C = C.to(device)
    return C


def build_gsnn_template(data, gsnn_kwargs: dict) -> GSNN:
    """Construct a fresh :class:`GSNN` from a bionetwork + kwargs.

    The same kwargs must be used at train and explain time so the parameter
    state-dict produced by the hypernet can be loaded into a freshly
    constructed clone.
    """
    return GSNN(
        edge_index_dict=data.edge_index_dict,
        node_names_dict=data.node_names_dict,
        **gsnn_kwargs,
    )


def build_hnet(gsnn: GSNN, n_cell_lines: int, hnet_cfg: dict) -> HyperNet:
    """Wrap ``gsnn`` in a :class:`HyperNet` with ``cond_dim = n_cell_lines``.

    ``hnet_cfg`` mirrors the top-level keys of the ``hypernetwork:`` block in
    the workflow config (only the ones HyperNet itself accepts).
    """
    hnet_kwargs = dict(
        stochastic_channels=int(hnet_cfg.get("stochastic_channels", 8)),
        width=int(hnet_cfg.get("width", 10)),
        cond_dim=int(n_cell_lines),
        nonlin=str(hnet_cfg.get("nonlin", "elu")),
        dropout=float(hnet_cfg.get("dropout", 0.0)),
        norm=str(hnet_cfg.get("norm", "none")),
        bias=bool(hnet_cfg.get("bias", False)),
        affine=bool(hnet_cfg.get("affine", False)),
        pz=str(hnet_cfg.get("pz", "normal")),
        learn_pz=bool(hnet_cfg.get("learn_pz", False)),
    )
    return HyperNet(model=gsnn, **hnet_kwargs)


def materialize_gsnn(
    hnet: HyperNet,
    C: torch.Tensor,
    *,
    z: Optional[torch.Tensor] = None,
    template: Optional[GSNN] = None,
) -> GSNN:
    """Return a *vanilla* GSNN clone whose parameters are sampled from ``hnet``.

    Args:
        hnet: a trained :class:`HyperNet` wrapping a :class:`GSNN`.
        C: condition vector of shape ``(cond_dim,)``.
        z: latent code; defaults to zeros (mean of standard normal prior).
        template: optional pre-built clone of ``hnet.model`` to load weights
            into. When ``None`` a :func:`copy.deepcopy` of ``hnet.model`` is
            used (preserving all buffers like ``edge_index``).

    Returns:
        A :class:`GSNN` instance with parameters set from
        ``hnet.sample(C=C, z=z)``. All non-trainable buffers come from the
        template and are therefore identical to those of ``hnet.model``.
    """
    if template is None:
        template = copy.deepcopy(hnet.model)
    if z is None:
        z = torch.zeros_like(hnet.mu)
    C = C.to(hnet.mu.device)
    z = z.to(hnet.mu.device)

    state_dict = hnet.sample(C=C, z=z)
    # ``hnet.sample`` only sets *parameters*; load non-strict so existing
    # buffers (edge_index, channel_groups, ...) are kept from the template.
    template.load_state_dict(state_dict, strict=False)
    return template


class CellLineRouter(torch.nn.Module):
    """Dispatch forwards to one of two cell-line-specific GSNN clones.

    Used to feed the existing single-model contrastive explainers in
    :mod:`gsnn.interpret` *without* modifying explainer code. The router
    accepts three call patterns observed in those explainers:

    1. ``model(x1, ...)`` / ``model(x2, ...)`` separate calls
       (``ContrastiveGSNNExplainer`` and ``ContrastiveOcclusionExplainer``'s
       ``_compute_diff_*`` baselines).
    2. ``model(cat([x1_batch, x2_batch], dim=0), ...)`` joint calls where the
       first half of the batch is x1-derived and the second half is
       x2-derived (``ContrastiveIGExplainer`` and the inner loop of
       ``ContrastiveOcclusionExplainer``). Both halves carry the same edge
       mask / node mask, also of shape ``(2*B', E|N)``.

    Dispatch heuristic:

    * If ``x.data_ptr()`` matches a registered x1 reference -> ``m1``.
    * If ``x.data_ptr()`` matches a registered x2 reference -> ``m2``.
    * Else if the leading dim is even and >= 2, split the batch (and any
      ``edge_mask`` / ``node_mask`` kwargs) into first/second halves and
      send each half through ``m1`` / ``m2``, then concatenate the outputs.
    * Else raise.

    ``data_ptr`` is used (instead of ``id``) because the explainers
    sometimes slice ``x[0:1]`` from the original; for slices starting at
    offset 0 of a contiguous tensor, ``slice.data_ptr() == src.data_ptr()``,
    which gives us a stable handle.

    Attributes mirrored from the wrapped GSNN are required by explainers
    (``edge_index``, ``homo_names``, ``num_nodes``).
    """

    def __init__(
        self,
        m1: torch.nn.Module,
        m2: torch.nn.Module,
        x1_ref: torch.Tensor,
        x2_ref: torch.Tensor,
    ) -> None:
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        # data_ptrs of registered reference tensors, kept in plain lists so
        # nn.Module's __setattr__ Tensor auto-registration is not triggered.
        self._x1_ptrs = [int(x1_ref.data_ptr())]
        self._x2_ptrs = [int(x2_ref.data_ptr())]
        # Mirror attributes the explainers read.
        self.edge_index = m1.edge_index
        self.homo_names = m1.homo_names
        self.num_nodes = m1.num_nodes

    def register_pair(self, x1_ref: torch.Tensor, x2_ref: torch.Tensor) -> None:
        """Register an additional accepted ``(x1, x2)`` pair (e.g. trajectory
        rollouts ``xt_hat_w_inputs_1`` / ``xt_hat_w_inputs_2``)."""
        self._x1_ptrs.append(int(x1_ref.data_ptr()))
        self._x2_ptrs.append(int(x2_ref.data_ptr()))

    @staticmethod
    def _split_kwargs(kwargs: dict, half: int) -> tuple:
        """Return ``(kwargs1, kwargs2)`` where any ``edge_mask`` / ``node_mask``
        with a leading dim of ``2*half`` is split in halves; other entries
        pass through unchanged."""
        kw1, kw2 = {}, {}
        for k, v in kwargs.items():
            if (
                isinstance(v, torch.Tensor)
                and v.dim() >= 1
                and v.size(0) == 2 * half
            ):
                kw1[k] = v[:half]
                kw2[k] = v[half:]
            else:
                kw1[k] = v
                kw2[k] = v
        return kw1, kw2

    def forward(self, x: torch.Tensor, **kwargs):
        ptr = int(x.data_ptr())
        if ptr in self._x1_ptrs:
            return self.m1(x, **kwargs)
        if ptr in self._x2_ptrs:
            return self.m2(x, **kwargs)

        if x.dim() >= 1 and x.size(0) >= 2 and x.size(0) % 2 == 0:
            half = x.size(0) // 2
            x1_part = x[:half]
            x2_part = x[half:]
            kw1, kw2 = self._split_kwargs(kwargs, half)
            out1 = self.m1(x1_part, **kw1)
            out2 = self.m2(x2_part, **kw2)
            return torch.cat([out1, out2], dim=0)

        raise RuntimeError(
            "CellLineRouter could not dispatch input of shape "
            f"{tuple(x.shape)}: data_ptr does not match any registered "
            "reference and batch size is not splittable into matched halves."
        )


def gsnn_init_dict(gsnn: GSNN) -> dict:
    """Build ``{param_name -> (mean, var)}`` from a freshly-initialized GSNN.

    Used to warm-start the hypernet with :func:`hnet.train.hnet.init_hnet`
    so that sampled thetas land in the same distribution as standard GSNN
    initialization.
    """
    init = {}
    for name, p in gsnn.named_parameters():
        if p.numel() > 2:
            mu = p.detach().mean().cpu()
            var = torch.clamp(p.detach().var().cpu(), 1e-4, 10.0)
            init[name] = (mu, var)
    return init


def save_hnet_artifact(
    path: str,
    hnet: HyperNet,
    cell_lines: List[str],
    gsnn_kwargs: dict,
    hnet_cfg: dict,
) -> None:
    """Serialize a hypernet so it can be reconstructed at explain time.

    Saves a plain dict (no class pickling beyond standard tensors / Python
    primitives) so the artifact loads cleanly with ``weights_only=False``
    regardless of how ``HyperNet`` evolves.
    """
    payload = {
        "kind": "hnet_gsnn_v1",
        "hnet_state_dict": hnet.state_dict(),
        "gsnn_kwargs": gsnn_kwargs,
        "hnet_cfg": hnet_cfg,
        "cell_lines": list(cell_lines),
    }
    torch.save(payload, path)


def load_hnet_artifact(path: str, data) -> dict:
    """Reload an artifact written by :func:`save_hnet_artifact`.

    Returns a dict with keys ``hnet`` (a :class:`HyperNet` ready for
    :meth:`HyperNet.sample`), ``gsnn_template`` (the template clone used to
    materialize per-cell-line GSNNs), ``cell_lines``, ``hnet_cfg``,
    ``gsnn_kwargs``.

    Raises if the saved cell-line vocabulary disagrees with the bionetwork
    currently in use, since that would silently corrupt cell-line
    conditioning.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "hnet_gsnn_v1":
        raise ValueError(
            f"Unrecognized artifact kind={payload.get('kind')!r} at {path}. "
            f"Expected 'hnet_gsnn_v1'."
        )

    cell_lines = list(payload["cell_lines"])
    bionet_cells = cell_lines_from_bionet(data)
    if cell_lines != bionet_cells:
        raise ValueError(
            "Cell-line vocabulary mismatch between hnet artifact and "
            f"bionetwork:\n  artifact: {cell_lines[:5]}... (n={len(cell_lines)})"
            f"\n  bionet:   {bionet_cells[:5]}... (n={len(bionet_cells)})"
        )

    gsnn_kwargs = dict(payload["gsnn_kwargs"])
    hnet_cfg = dict(payload["hnet_cfg"])

    gsnn_template = build_gsnn_template(data, gsnn_kwargs)
    hnet = build_hnet(gsnn_template, n_cell_lines=len(cell_lines), hnet_cfg=hnet_cfg)
    hnet.load_state_dict(payload["hnet_state_dict"])
    hnet.eval()

    return {
        "hnet": hnet,
        "gsnn_template": gsnn_template,
        "cell_lines": cell_lines,
        "hnet_cfg": hnet_cfg,
        "gsnn_kwargs": gsnn_kwargs,
    }
