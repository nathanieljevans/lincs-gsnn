"""Helpers for the BIOGSNN ``gene_norm`` feature.

This module copies the per-gene control-population mean / std used by the
upstream ``lincs-traj`` pipeline (``workflow/scripts/make_proc.py``) to
z-score LINCS Level 3 landmark expression into a small versioned artifact
(``gene_norm.pt``) that lives next to ``bionetwork.pt`` and is consumed by
:class:`lincs_gsnn.models.BIOGSNN.BIOGSNN`.

BIOGSNN uses the stats to back-transform its z-scored gene-input state into
an abundance-like proxy ``level3 = relu(mu + sigma * x_z)`` so the
degradation term ``- gamma * level3`` is guaranteed non-positive (i.e.
strictly removes mRNA). See the BIOGSNN module docstring for the math.

Upstream source contract
------------------------
``gene_stats.dict`` is a flat dict saved by ``lincs-traj``'s
``make_proc.py``::

    torch.save({'means': xx_mean, 'xx_std': xx_std}, '.../gene_stats.dict')

where ``xx_mean`` / ``xx_std`` are length-978 numpy arrays of per-gene
control-population statistics. The same 978 landmark genes are listed (in
the same order) in ``predict_grid/gene_names.csv`` (``gene_names`` column),
which is the canonical alignment used everywhere else in this repo.
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ARTIFACT_KIND = "gene_norm_v1"


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def build_gene_norm_artifact(
    gene_stats_path: str,
    gene_names_csv_path: str,
    output_names: Optional[Sequence[str]] = None,
) -> Dict:
    """Construct a ``gene_norm`` payload from upstream artifacts.

    Parameters
    ----------
    gene_stats_path
        Path to ``gene_stats.dict`` produced by
        ``lincs-traj/workflow/scripts/make_proc.py``. Must contain
        ``means`` and ``xx_std`` arrays of equal length.
    gene_names_csv_path
        Path to ``predict_grid/gene_names.csv`` (column ``gene_names``).
        Defines the positional alignment of ``means`` / ``xx_std``.
    output_names
        Optional bionetwork ``node_names_dict['output']``. When provided,
        every ``GENE__X`` entry must have a matching ``X`` in the
        gene_names list; raises ``ValueError`` otherwise.

    Returns
    -------
    dict
        Payload ready for :func:`save_gene_norm_artifact`.
    """
    stats = torch.load(gene_stats_path, map_location="cpu", weights_only=False)

    if not isinstance(stats, Mapping) or "means" not in stats or "xx_std" not in stats:
        raise ValueError(
            f"{gene_stats_path}: expected a dict with keys 'means' and 'xx_std' "
            f"(produced by lincs-traj/make_proc.py); got keys "
            f"{list(stats.keys()) if isinstance(stats, Mapping) else type(stats)!r}."
        )

    mu = torch.as_tensor(np.asarray(stats["means"]), dtype=torch.float32).flatten()
    sigma = torch.as_tensor(np.asarray(stats["xx_std"]), dtype=torch.float32).flatten()

    if mu.shape != sigma.shape:
        raise ValueError(
            f"{gene_stats_path}: shape mismatch between means {tuple(mu.shape)} "
            f"and xx_std {tuple(sigma.shape)}."
        )

    gene_names = (
        pd.read_csv(gene_names_csv_path)["gene_names"].astype(str).tolist()
    )

    if len(gene_names) != mu.numel():
        raise ValueError(
            f"gene_names ({len(gene_names)}) and gene_stats ({mu.numel()}) "
            "have different lengths; cannot align."
        )

    if output_names is not None:
        gene_set = set(gene_names)
        missing = [
            n for n in output_names
            if n.startswith("GENE__") and n.split("__", 1)[1] not in gene_set
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(output_names)} output genes have no "
                f"matching entry in gene_stats / gene_names.csv "
                f"(showing up to 5): {missing[:5]}."
            )

    payload = {
        "kind": ARTIFACT_KIND,
        "gene_names": gene_names,
        "mu": mu.contiguous(),
        "sigma": sigma.contiguous(),
        "source": os.fspath(gene_stats_path),
    }
    return payload


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def save_gene_norm_artifact(path: str, payload: Mapping) -> None:
    """Serialize a ``gene_norm`` payload to ``path``."""
    if payload.get("kind") != ARTIFACT_KIND:
        raise ValueError(
            f"refusing to save payload with kind={payload.get('kind')!r}; "
            f"expected {ARTIFACT_KIND!r}."
        )
    out = {
        "kind": ARTIFACT_KIND,
        "gene_names": list(payload["gene_names"]),
        "mu": torch.as_tensor(payload["mu"], dtype=torch.float32).detach().cpu().contiguous(),
        "sigma": torch.as_tensor(payload["sigma"], dtype=torch.float32).detach().cpu().contiguous(),
        "source": str(payload.get("source", "")),
    }
    torch.save(out, path)


def load_gene_norm_artifact(
    path: str,
    output_names: Optional[Sequence[str]] = None,
) -> Dict:
    """Load an artifact written by :func:`save_gene_norm_artifact`.

    When ``output_names`` is provided, every ``GENE__X`` entry must have a
    matching ``X`` in the artifact's ``gene_names``; raises ``ValueError``
    otherwise. The returned payload always contains the full (978,) ``mu``
    / ``sigma`` in the original (``gene_names.csv``) order; per-output
    reordering is done by :func:`mu_sigma_for_outputs` so callers can
    align to whatever bionetwork they have at hand.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != ARTIFACT_KIND:
        raise ValueError(
            f"Unrecognized gene_norm artifact kind={payload.get('kind')!r} at "
            f"{path}. Expected {ARTIFACT_KIND!r}."
        )

    mu = torch.as_tensor(payload["mu"], dtype=torch.float32).flatten()
    sigma = torch.as_tensor(payload["sigma"], dtype=torch.float32).flatten()
    gene_names = [str(n) for n in payload["gene_names"]]
    if not (mu.numel() == sigma.numel() == len(gene_names)):
        raise ValueError(
            f"gene_norm artifact at {path} is internally inconsistent: "
            f"len(gene_names)={len(gene_names)}, mu={mu.numel()}, "
            f"sigma={sigma.numel()}."
        )

    if output_names is not None:
        gene_set = set(gene_names)
        missing = [
            n for n in output_names
            if n.startswith("GENE__") and n.split("__", 1)[1] not in gene_set
        ]
        if missing:
            raise ValueError(
                f"gene_norm artifact at {path} is missing {len(missing)} of "
                f"{len(output_names)} output genes (showing up to 5): "
                f"{missing[:5]}. Rebuild via "
                f"`make_bio_network.py --gene_stats_path ...`."
            )

    payload["mu"] = mu
    payload["sigma"] = sigma
    payload["gene_names"] = gene_names
    return payload


# ----------------------------------------------------------------------------
# Per-output reorder helper
# ----------------------------------------------------------------------------
def mu_sigma_for_outputs(
    payload: Mapping,
    output_names: Sequence[str],
    *,
    sigma_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-order ``mu`` / ``sigma`` to bionetwork output-gene order.

    Output entries must be ``GENE__<SYMBOL>``; the matching symbol is
    looked up in the artifact's ``gene_names``. Raises ``ValueError`` on
    any missing output gene (callers should also validate this at load
    time via ``load_gene_norm_artifact(..., output_names=...)``).

    ``sigma_eps`` is added to every returned sigma so the back-transform
    is numerically safe (matches the ``+ 1e-8`` floor that lincs-traj's
    ``make_proc.py`` uses at z-score time, so the round-trip is exactly
    invertible).
    """
    name_to_idx = {n: i for i, n in enumerate(payload["gene_names"])}

    bad: List[str] = []
    inv_idx: List[int] = []
    for n in output_names:
        sym = n.split("__", 1)[1] if "__" in n else n
        i = name_to_idx.get(sym)
        if i is None:
            bad.append(n)
            inv_idx.append(0)
        else:
            inv_idx.append(i)
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(output_names)} output genes are absent from "
            f"the gene_norm artifact (showing up to 5): {bad[:5]}."
        )

    idx = torch.tensor(inv_idx, dtype=torch.long)
    mu_g = torch.as_tensor(payload["mu"], dtype=torch.float32).flatten()[idx].contiguous()
    sigma_g = torch.as_tensor(payload["sigma"], dtype=torch.float32).flatten()[idx].contiguous()
    sigma_g = sigma_g + float(sigma_eps)
    return mu_g, sigma_g
