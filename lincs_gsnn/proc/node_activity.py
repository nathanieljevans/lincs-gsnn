"""Helpers for the GSNN ``node_activity`` feature.

This module builds the per-cell-line *function-node activity* tensor ``x_fn``
that a :class:`gsnn.models.GSNN.GSNN` constructed with ``node_activity=True``
expects on every forward pass.

The tensor has shape ``(n_function_nodes, activity_dim)`` per cell line; the
canonical runtime representation is a dict keyed by ``cell_iname`` so the
training dataset and the explanation scripts can look up the right row by
cell line.  An on-disk artifact (``node_activity.pt``) is produced once at
network-build time and consumed by both training and explanation.

The default feature is expression z-score (``activity_dim=1``).  Additional
per-function-node channels are selected via the ``features`` argument of
:func:`build_x_fn_lookup_from_bionet` (e.g. ``features=("expr", "mut")``
produces ``activity_dim=2`` with channel order matching the list).  See
:data:`ACTIVITY_FEATURE_BUILDERS` for the registry of supported features.
Any caller that wants channels beyond the registry should build a
``(n_cells, n_function_nodes, activity_dim>1)`` tensor with the same
cell-line ordering and call :func:`save_node_activity_artifact` directly.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch


_LINE_PREFIX = "LINE__"
ARTIFACT_KIND = "node_activity_v1"


# ----------------------------------------------------------------------------
# Bionet helpers
# ----------------------------------------------------------------------------
def function_nodes_to_gene_symbols(node_names_dict: Mapping[str, Sequence[str]]) -> List[str]:
    """Return one gene symbol per function node (bionet order).

    Function nodes are named ``<TYPE>__<NAME>`` (e.g. ``PROTEIN__TP53``,
    ``RNA__TP53``). This strips the prefix and uppercases the suffix so it
    matches the column ordering of the DepMap expression matrix.  Function
    nodes whose suffix is not a real gene symbol (e.g. miRNA, reaction ids)
    are returned verbatim; they will be zero-filled downstream by
    :func:`create_node_activity_inputs`.
    """
    out: List[str] = []
    for name in node_names_dict["function"]:
        parts = str(name).split("__", 1)
        suffix = parts[1] if len(parts) == 2 else parts[0]
        out.append(suffix.upper())
    return out


def cell_inames_from_bionet(node_names_dict: Mapping[str, Sequence[str]]) -> List[str]:
    """Return the ``LINE__*`` cell-iname vocabulary from a bionetwork.

    Order matches the bionetwork's ``input`` node ordering so downstream code
    that already uses :func:`lincs_gsnn.models.HnetGSNN.cell_lines_from_bionet`
    (which uses the same prefix) stays in lock-step.
    """
    return [n[len(_LINE_PREFIX):] for n in node_names_dict["input"] if str(n).startswith(_LINE_PREFIX)]


# ----------------------------------------------------------------------------
# cell_iname <-> DepMap ModelID resolution
# ----------------------------------------------------------------------------
def build_cell_iname_to_modelid_map(data_root: str) -> Dict[str, str]:
    """Build a ``cell_iname -> DepMap ModelID`` map.

    Uses DepMap's ``Model.csv`` which provides a clean
    ``StrippedCellLineName`` column (e.g. ``MDAMB231``) alongside
    ``ModelID`` (e.g. ``ACH-000768``).  ``StrippedCellLineName`` aligns
    1:1 with the LINCS ``cell_iname`` vocabulary used elsewhere in the
    workflow, which avoids the brittle ``ccle_name.split('_')[0]``
    heuristic used previously.

    Lookup is case-insensitive on the ``StrippedCellLineName`` side; the
    returned map keys are uppercased so callers should compare against
    ``cell_iname.upper()``.  Cell lines whose stripped name collides
    across multiple ``ModelID`` rows keep their first occurrence
    (deterministic by file order) and a single warning is emitted listing
    the dropped duplicates.
    """
    info_path = os.path.join(data_root, "Model.csv")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"Required mapping file not found: {info_path}. "
            "Set `dirs.data` in workflow/train/config.yaml so it points to a "
            "directory containing DepMap's `Model.csv`."
        )

    info = pd.read_csv(
        info_path,
        usecols=["ModelID", "StrippedCellLineName"],
        low_memory=False,
    )
    info = info.dropna(subset=["ModelID", "StrippedCellLineName"])
    info = info[info["StrippedCellLineName"].astype(str).str.len() > 0]

    iname2modelid: Dict[str, str] = {}
    duplicates: List[Tuple[str, str, str]] = []
    for model_id, stripped in zip(
        info["ModelID"].astype(str), info["StrippedCellLineName"].astype(str)
    ):
        iname = stripped.upper()
        if iname in iname2modelid:
            duplicates.append((iname, iname2modelid[iname], model_id))
            continue
        iname2modelid[iname] = model_id

    if duplicates:
        sample = duplicates[:5]
        warnings.warn(
            f"build_cell_iname_to_modelid_map: dropped {len(duplicates)} duplicate "
            f"StrippedCellLineName(s) keeping the first occurrence (showing up to 5): {sample}",
            RuntimeWarning,
        )

    return iname2modelid


def resolve_cell_inames(
    cell_inames: Iterable[str],
    iname2modelid: Mapping[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Split ``cell_inames`` into (kept_inames, kept_modelids, dropped_inames).

    Comparison is case-insensitive on the cell_iname side; the returned
    ``kept_inames`` preserve the original casing supplied by the caller so
    downstream lookups against LINCS metadata still match.
    """
    kept_inames: List[str] = []
    kept_modelids: List[str] = []
    dropped: List[str] = []
    for iname in cell_inames:
        key = str(iname).upper()
        if key in iname2modelid:
            kept_inames.append(iname)
            kept_modelids.append(iname2modelid[key])
        else:
            dropped.append(iname)
    return kept_inames, kept_modelids, dropped


def _load_mutation_modelids(data_root: str) -> set:
    """Return the set of ``ModelID`` strings present in
    ``OmicsSomaticMutationsMatrixDamaging.csv``.

    Mirrors :func:`_load_expression_modelids` so the cell-line filter pass
    in :func:`build_x_fn_lookup_from_bionet` can intersect ModelID coverage
    across all selected activity features.
    """
    mut_path = os.path.join(data_root, "OmicsSomaticMutationsMatrixDamaging.csv")
    if not os.path.exists(mut_path):
        raise FileNotFoundError(
            f"Required mutation file not found: {mut_path}. "
            "Set `dirs.data` in workflow/train/config.yaml so it points to a "
            "directory containing `OmicsSomaticMutationsMatrixDamaging.csv`."
        )
    modelids = pd.read_csv(mut_path, usecols=["ModelID"], low_memory=False)["ModelID"]
    return set(modelids.dropna().astype(str).tolist())


def _load_expression_modelids(data_root: str) -> set:
    """Return the set of ``ModelID`` strings present in
    ``OmicsExpressionTPMLogp1HumanAllGenes.csv``.

    Only the ``ModelID`` column is read so this is cheap relative to the
    full expression-matrix load.  Used by :func:`build_x_fn_lookup_from_bionet`
    to filter out cells whose ``Model.csv`` ModelID has no expression row,
    consistent with the pipeline's "drop with warning" policy.
    """
    expr_path = os.path.join(data_root, "OmicsExpressionTPMLogp1HumanAllGenes.csv")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(
            f"Required expression file not found: {expr_path}. "
            "Set `dirs.data` in workflow/train/config.yaml so it points to a "
            "directory containing `OmicsExpressionTPMLogp1HumanAllGenes.csv`."
        )
    modelids = pd.read_csv(expr_path, usecols=["ModelID"], low_memory=False)["ModelID"]
    return set(modelids.dropna().astype(str).tolist())


# ----------------------------------------------------------------------------
# Feature builders
# ----------------------------------------------------------------------------
def create_node_activity_inputs(function_genes, cell_lines, data_root, eps=1e-6):
    """Expression-z-score feature tensor.

    Parameters
    ----------
    function_genes : Sequence[str]
        Per-function-node gene symbols (uppercase). Use
        :func:`function_nodes_to_gene_symbols` to derive these from a
        bionetwork.
    cell_lines : Sequence[str]
        DepMap ``ModelID`` strings (e.g. ``ACH-001113``).  Use
        :func:`build_cell_iname_to_modelid_map` to translate from LINCS
        ``cell_iname`` if needed.
    data_root : str
        Directory containing ``Gene.csv`` and
        ``OmicsExpressionTPMLogp1HumanAllGenes.csv``.
    eps : float
        Stabilizer added to the std-dev when computing the z-score.

    Returns
    -------
    torch.Tensor of shape ``(n_cells, n_function_nodes, 1)``
        Float32 expression z-score for every function-node-gene, in the
        provided ``cell_lines`` order.  Genes not present in the expression
        matrix are zero-filled (consistent with non-gene function nodes).

    Notes
    -----
    Slow (~2.5 minutes on the all-genes matrix); cache the resulting tensor
    via :func:`save_node_activity_artifact`.
    """
    geneinfo = pd.read_csv(f"{data_root}/Gene.csv", low_memory=False)[["ensembl_gene_id", "symbol"]]
    ens2sym = dict(zip(geneinfo["ensembl_gene_id"], geneinfo["symbol"]))
    expr = pd.read_csv(f"{data_root}/OmicsExpressionTPMLogp1HumanAllGenes.csv")

    expr = expr.drop(columns=["Unnamed: 0"])
    expr.columns = expr.columns[:5].tolist() + [x.split(" ")[0].upper() for x in expr.columns[5:]]
    expr.columns = [ens2sym[x] if x in ens2sym else x for x in expr.columns]
    expr = expr.set_index("ModelID")[expr.columns[5:]]

    # ``OmicsExpressionTPMLogp1HumanAllGenes.csv`` occasionally contains
    # multiple rows per ``ModelID`` (technical/biological replicates of the
    # same cell line).  Collapse them with a mean BEFORE z-scoring so:
    #  1. Population mean/std aren't biased toward replicate-heavy lines.
    #  2. ``expr_zscore.loc[ModelID]`` returns a single row downstream, so
    #     `.values.reshape(n_cells, n_genes, 1)` matches `n_cells` exactly.
    if not expr.index.is_unique:
        n_before = len(expr)
        expr = expr.groupby(level=0).mean()
        n_after = len(expr)
        warnings.warn(
            f"create_node_activity_inputs: collapsed {n_before - n_after} duplicate "
            f"ModelID row(s) in expression matrix by averaging "
            f"({n_before} rows -> {n_after} unique cell lines).",
            RuntimeWarning,
        )

    expr_zscore = (expr - expr.mean(axis=0)) / (expr.std(axis=0) + eps)
    expr_genes = expr.columns.tolist()

    missing_modelids = [c for c in cell_lines if c not in expr_zscore.index]
    if missing_modelids:
        raise KeyError(
            f"create_node_activity_inputs: {len(missing_modelids)} ModelID(s) not "
            f"present in OmicsExpressionTPMLogp1HumanAllGenes.csv "
            f"(showing up to 5): {missing_modelids[:5]}. Filter unresolvable "
            "cell lines upstream (see resolve_cell_inames)."
        )
    expr_zscore = expr_zscore.loc[list(cell_lines)]

    missing_genes = set(function_genes) - set(expr_genes)
    if missing_genes:
        missing_df = pd.DataFrame(0, index=expr_zscore.index, columns=list(missing_genes))
        expr_zscore = pd.concat([expr_zscore, missing_df], axis=1)

    n_genes = len(function_genes)
    n_cells = len(cell_lines)
    n_feats = 1

    arr = expr_zscore.loc[list(cell_lines), list(function_genes)].values
    out = torch.tensor(arr, dtype=torch.float32).reshape(n_cells, n_genes, n_feats)
    return out


def create_damaging_mutation_inputs(function_genes, cell_lines, data_root):
    """Damaging-somatic-mutation indicator tensor.

    Parameters
    ----------
    function_genes : Sequence[str]
        Per-function-node gene symbols (uppercase). Use
        :func:`function_nodes_to_gene_symbols` to derive these from a
        bionetwork.
    cell_lines : Sequence[str]
        DepMap ``ModelID`` strings (e.g. ``ACH-001113``).
    data_root : str
        Directory containing ``OmicsSomaticMutationsMatrixDamaging.csv``.

    Returns
    -------
    torch.Tensor of shape ``(n_cells, n_function_nodes, 1)``
        Binary (0/1) damaging-mutation status per (cell line, function-node
        gene), kept as-is (no z-score).  Function-node genes that don't
        appear as columns in the damaging-mutation matrix (e.g. non-coding
        function nodes, reaction ids) are zero-filled, matching the policy
        used by :func:`create_node_activity_inputs`.
    """
    mut_path = os.path.join(data_root, "OmicsSomaticMutationsMatrixDamaging.csv")
    if not os.path.exists(mut_path):
        raise FileNotFoundError(
            f"Required mutation file not found: {mut_path}. "
            "Set `dirs.data` in workflow/train/config.yaml so it points to a "
            "directory containing `OmicsSomaticMutationsMatrixDamaging.csv`."
        )

    mut = pd.read_csv(mut_path, low_memory=False)
    mut = mut.drop(columns=["Unnamed: 0"], errors="ignore")

    # Header convention: 5 metadata cols (ModelID, SequencingID,
    # ModelConditionID, IsDefaultEntryForModel, IsDefaultEntryForMC) followed
    # by per-gene damaging-mutation indicators named like "TP53 (7157)".
    gene_cols_raw = mut.columns.tolist()[5:]
    gene_names = [str(x).split(" ")[0].upper() for x in gene_cols_raw]
    mut.columns = mut.columns[:5].tolist() + gene_names
    mut = mut.set_index("ModelID")[gene_names]

    # Some ModelIDs have multiple sequencing rows; collapse with max (binary
    # OR) so the per-cell tensor stays {0,1}. Same pattern as the expression
    # builder's groupby-mean, but max is the right reduction for indicators.
    if not mut.index.is_unique:
        n_before = len(mut)
        mut = mut.groupby(level=0).max()
        n_after = len(mut)
        warnings.warn(
            f"create_damaging_mutation_inputs: collapsed {n_before - n_after} "
            f"duplicate ModelID row(s) in damaging-mutation matrix by max "
            f"({n_before} rows -> {n_after} unique cell lines).",
            RuntimeWarning,
        )

    missing_modelids = [c for c in cell_lines if c not in mut.index]
    if missing_modelids:
        raise KeyError(
            f"create_damaging_mutation_inputs: {len(missing_modelids)} ModelID(s) "
            f"not present in OmicsSomaticMutationsMatrixDamaging.csv "
            f"(showing up to 5): {missing_modelids[:5]}. Filter unresolvable "
            "cell lines upstream (see build_x_fn_lookup_from_bionet)."
        )
    mut = mut.loc[list(cell_lines)]

    missing_genes = set(function_genes) - set(mut.columns)
    if missing_genes:
        missing_df = pd.DataFrame(0, index=mut.index, columns=list(missing_genes))
        mut = pd.concat([mut, missing_df], axis=1)

    n_genes = len(function_genes)
    n_cells = len(cell_lines)
    n_feats = 1

    arr = mut.loc[list(cell_lines), list(function_genes)].values
    out = torch.tensor(arr, dtype=torch.float32).reshape(n_cells, n_genes, n_feats)
    return out


# ----------------------------------------------------------------------------
# Feature registry: maps short feature names (used in config.yaml) to a
# (per-feature ModelID set loader, per-feature tensor builder). Adding a new
# activity channel only requires extending this dict.
# ----------------------------------------------------------------------------
def _build_expr_feature(function_genes, function_node_names, cell_lines, data_root, eps):
    del function_node_names
    return create_node_activity_inputs(function_genes, cell_lines, data_root, eps=eps)


def _build_mut_feature(function_genes, function_node_names, cell_lines, data_root, eps):
    del function_node_names, eps  # eps only stabilizes z-scoring; mutation status is binary.
    return create_damaging_mutation_inputs(function_genes, cell_lines, data_root)


def _broadcast_node_mask(mask_bools: Sequence[bool], n_cells: int) -> torch.Tensor:
    """Lift a per-function-node bool mask to a (n_cells, n_fn, 1) float tensor.

    Node-only features (``is_protein``, ``is_rna``, ``is_mirna``, ...) are
    constant across cell lines, so we materialize one row per cell line by
    broadcasting.  Using ``.expand`` + ``.contiguous`` keeps storage cheap
    while matching the (n_cells, n_fn, 1) layout the concatenation step
    expects.
    """
    mask = torch.tensor(list(mask_bools), dtype=torch.float32).reshape(1, -1, 1)
    return mask.expand(int(n_cells), -1, -1).contiguous()


def _build_is_protein_feature(function_genes, function_node_names, cell_lines, data_root, eps):
    del function_genes, data_root, eps
    flags = [str(n).startswith("PROTEIN__") for n in function_node_names]
    return _broadcast_node_mask(flags, len(cell_lines))


def _build_is_rna_feature(function_genes, function_node_names, cell_lines, data_root, eps):
    del function_genes, data_root, eps
    flags = [str(n).startswith("RNA__") for n in function_node_names]
    return _broadcast_node_mask(flags, len(cell_lines))


def _build_is_mirna_feature(function_genes, function_node_names, cell_lines, data_root, eps):
    del function_genes, data_root, eps
    # miRNA function nodes use the ``RNA__`` prefix and a mature-miRNA name
    # that always contains both ``hsa`` and ``miR`` (e.g. ``RNA__hsa-miR-21-5p``).
    # Match on the post-prefix suffix and require BOTH substrings so we don't
    # accidentally flag a non-miRNA RNA whose symbol happens to contain one
    # substring.
    flags: List[bool] = []
    for n in function_node_names:
        s = str(n)
        if not s.startswith("RNA__"):
            flags.append(False)
            continue
        suffix = s.split("__", 1)[1]
        flags.append(("hsa" in suffix) and ("miR" in suffix))
    return _broadcast_node_mask(flags, len(cell_lines))


def build_cell_line_activity_tensor(
    cell_inames: Sequence[str],
    line_vocab: Sequence[str],
    n_function_nodes: int,
) -> torch.Tensor:
    """One-hot cell-line features broadcast to every function node.

    Parameters
    ----------
    cell_inames : Sequence[str]
        Rows to materialize (typically the post-filter ``kept_inames`` list).
    line_vocab : Sequence[str]
        Full ``LINE__`` vocabulary in bionetwork input order (stripped names).
    n_function_nodes : int
        Number of function nodes to broadcast each one-hot row across.

    Returns
    -------
    torch.Tensor of shape ``(len(cell_inames), n_function_nodes, len(line_vocab))``
    """
    line_vocab = list(line_vocab)
    iname_to_idx = {str(n): i for i, n in enumerate(line_vocab)}
    n_cells = len(cell_inames)
    n_lines = len(line_vocab)
    n_fn = int(n_function_nodes)
    out = torch.zeros(n_cells, n_fn, n_lines, dtype=torch.float32)
    for i, iname in enumerate(cell_inames):
        key = str(iname)
        if key not in iname_to_idx:
            raise KeyError(
                f"build_cell_line_activity_tensor: cell_iname={key!r} not in "
                f"line_vocab (size={n_lines})."
            )
        out[i, :, iname_to_idx[key]] = 1.0
    return out


def _build_cell_line_feature(
    function_genes,
    function_node_names,
    cell_inames,
    line_vocab,
    data_root,
    eps,
):
    del function_genes, data_root, eps
    return build_cell_line_activity_tensor(
        cell_inames, line_vocab, len(function_node_names)
    )


# Registry entry per supported feature:
#   feature_name -> (modelid_loader_or_None, tensor_builder)
# A loader of ``None`` marks a node-only feature whose value is determined
# entirely from the bionetwork's function-node names; such features do not
# restrict cell-line coverage in the intersection pass below.
ACTIVITY_FEATURE_BUILDERS: Dict[str, Tuple] = {
    "expr":       (_load_expression_modelids, _build_expr_feature),
    "mut":        (_load_mutation_modelids,   _build_mut_feature),
    "is_protein": (None,                       _build_is_protein_feature),
    "is_rna":     (None,                       _build_is_rna_feature),
    "is_mirna":   (None,                       _build_is_mirna_feature),
    "cell_line":  (None,                       _build_cell_line_feature),
}


def _validate_features(features: Sequence[str]) -> List[str]:
    """Return a normalized, non-empty list of feature names, or raise."""
    feats = list(features)
    if not feats:
        raise ValueError(
            "build_x_fn_lookup_from_bionet: `features` must be non-empty; "
            f"supported: {sorted(ACTIVITY_FEATURE_BUILDERS)}."
        )
    unknown = [f for f in feats if f not in ACTIVITY_FEATURE_BUILDERS]
    if unknown:
        raise ValueError(
            f"build_x_fn_lookup_from_bionet: unknown activity feature(s) {unknown}; "
            f"supported: {sorted(ACTIVITY_FEATURE_BUILDERS)}."
        )
    seen = set()
    deduped: List[str] = []
    for f in feats:
        if f in seen:
            warnings.warn(
                f"build_x_fn_lookup_from_bionet: duplicate feature {f!r} "
                "ignored; each activity channel is included once.",
                RuntimeWarning,
            )
            continue
        seen.add(f)
        deduped.append(f)
    return deduped


def build_x_fn_lookup_from_bionet(
    node_names_dict: Mapping[str, Sequence[str]],
    data_root: str,
    cell_inames: Optional[Sequence[str]] = None,
    eps: float = 1e-6,
    features: Sequence[str] = ("expr",),
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """End-to-end builder for an ``x_fn`` lookup keyed by ``cell_iname``.

    Returns ``(x_fn_by_ciname, metadata)`` where ``x_fn_by_ciname`` maps
    ``cell_iname -> Tensor[n_function_nodes, activity_dim]`` and
    ``metadata`` contains the canonical orderings, the unresolved cell
    inames, the per-channel feature names, and the activity_dim so callers
    can persist a self-describing artifact.

    ``cell_inames`` defaults to all ``LINE__*`` entries in
    ``node_names_dict['input']``.  Inames that cannot be resolved to a DepMap
    ModelID are dropped with a single warning (per the "drop with warning"
    policy chosen for this pipeline).

    Parameters
    ----------
    features : Sequence[str]
        Ordered list of per-function-node activity channels to include
        (default ``("expr",)``, which reproduces the original
        ``activity_dim=1`` artifact).  Each entry must be a key of
        :data:`ACTIVITY_FEATURE_BUILDERS` (currently ``"expr"`` and
        ``"mut"``).  The resulting per-cell tensor has shape
        ``(n_function_nodes, len(features))`` with channel order matching
        ``features``.  Cell lines absent from ANY selected feature's matrix
        are dropped (with a single warning) so the artifact stays
        self-consistent across channels.
    """
    function_genes = function_nodes_to_gene_symbols(node_names_dict)
    function_node_names = list(node_names_dict["function"])

    feature_list = _validate_features(features)

    if cell_inames is None:
        cell_inames = cell_inames_from_bionet(node_names_dict)
    cell_inames = list(cell_inames)

    line_vocab = cell_inames_from_bionet(node_names_dict)
    needs_depmap = any(
        ACTIVITY_FEATURE_BUILDERS[f][0] is not None for f in feature_list
    )

    dropped_inames: List[str] = []
    no_coverage_inames: List[str] = []

    if needs_depmap:
        iname2modelid = build_cell_iname_to_modelid_map(data_root)
        kept_inames, kept_modelids, dropped_inames = resolve_cell_inames(
            cell_inames, iname2modelid
        )

        if dropped_inames:
            sample = dropped_inames[:5]
            warnings.warn(
                f"build_x_fn_lookup_from_bionet: {len(dropped_inames)} cell_iname(s) "
                f"could not be resolved to a DepMap ModelID and will be dropped "
                f"(showing up to 5): {sample}.",
                RuntimeWarning,
            )

        # ------------------------------------------------------------------
        # Second filter pass: even after iname->ModelID resolution, some
        # ModelIDs may be absent from one or more selected feature matrices
        # (e.g. MCF10A in newer DepMap expression releases, or cell lines
        # never sequenced for the damaging-mutation panel).  Intersect ModelID
        # coverage across ALL selected features so each per-feature builder
        # only sees ModelIDs guaranteed to be present in its matrix; this
        # keeps the concatenated tensor row-aligned across channels.
        # ------------------------------------------------------------------
        coverage_sets: Dict[str, Optional[set]] = {
            feat: (
                ACTIVITY_FEATURE_BUILDERS[feat][0](data_root)
                if ACTIVITY_FEATURE_BUILDERS[feat][0] is not None
                else None
            )
            for feat in feature_list
        }

        missing_per_feature: Dict[str, List[Tuple[str, str]]] = {
            f: [] for f in feature_list
        }
        keep_mask: List[bool] = []
        for iname, mid in zip(kept_inames, kept_modelids):
            keep = True
            for feat in feature_list:
                cov = coverage_sets[feat]
                if cov is None:
                    continue
                if mid not in cov:
                    missing_per_feature[feat].append((iname, mid))
                    keep = False
            keep_mask.append(keep)

        no_coverage_inames = [n for n, k in zip(kept_inames, keep_mask) if not k]
        no_coverage_modelids = [
            m for m, k in zip(kept_modelids, keep_mask) if not k
        ]

        if no_coverage_inames:
            summary = ", ".join(
                f"{feat}: {len(missing_per_feature[feat])}" for feat in feature_list
            )
            sample = list(zip(no_coverage_inames, no_coverage_modelids))[:5]
            warnings.warn(
                f"build_x_fn_lookup_from_bionet: {len(no_coverage_inames)} resolved "
                f"ModelID(s) absent from one or more selected feature matrices and "
                f"will be dropped (per-feature missing counts -> {summary}; showing "
                f"up to 5 dropped (iname, modelid): {sample}).",
                RuntimeWarning,
            )
            kept_inames = [n for n, k in zip(kept_inames, keep_mask) if k]
            kept_modelids = [m for m, k in zip(kept_modelids, keep_mask) if k]

        if not kept_inames:
            raise ValueError(
                "build_x_fn_lookup_from_bionet: no cell_iname could be resolved to a "
                f"DepMap ModelID with data for every requested feature {feature_list}; "
                "cannot build node_activity artifact."
            )
    else:
        kept_inames = list(cell_inames)
        kept_modelids = []

    if not kept_inames:
        raise ValueError(
            "build_x_fn_lookup_from_bionet: no cell_iname available to build "
            f"node_activity artifact for features {feature_list}."
        )

    # Build each per-feature tensor on the same (cell_lines, function_genes)
    # ordering, then concatenate on the channel axis. Resulting shape:
    # (n_cells, n_function_nodes, sum_of_per_feature_dims).
    feature_tensors: List[torch.Tensor] = []
    for feat in feature_list:
        if feat == "cell_line":
            feature_tensors.append(
                _build_cell_line_feature(
                    function_genes,
                    function_node_names,
                    kept_inames,
                    line_vocab,
                    data_root,
                    eps,
                )
            )
            continue
        builder = ACTIVITY_FEATURE_BUILDERS[feat][1]
        cell_arg = (
            kept_modelids
            if ACTIVITY_FEATURE_BUILDERS[feat][0] is not None
            else kept_inames
        )
        feature_tensors.append(
            builder(function_genes, function_node_names, cell_arg, data_root, eps)
        )
    tensor = torch.cat(feature_tensors, dim=-1) if len(feature_tensors) > 1 else feature_tensors[0]

    activity_dim = int(tensor.shape[-1])
    x_fn_by_ciname: Dict[str, torch.Tensor] = {
        iname: tensor[i].contiguous().clone() for i, iname in enumerate(kept_inames)
    }

    metadata = dict(
        kind=ARTIFACT_KIND,
        function_genes=list(function_genes),
        function_node_names=function_node_names,
        cell_iname_order=list(kept_inames),
        cell_modelid_order=list(kept_modelids),
        dropped_cell_inames=list(dropped_inames),
        dropped_cell_inames_no_expression=list(no_coverage_inames),
        activity_dim=activity_dim,
        activity_features=list(feature_list),
        line_vocab_order=list(line_vocab),
    )
    return x_fn_by_ciname, metadata


# ----------------------------------------------------------------------------
# Artifact persistence
# ----------------------------------------------------------------------------
def save_node_activity_artifact(
    path: str,
    x_fn_by_ciname: Mapping[str, torch.Tensor],
    metadata: Mapping,
) -> None:
    """Serialize an ``x_fn`` lookup to ``path``.

    Stores a flat dict containing the per-cell tensors plus the metadata
    needed to validate the artifact at load time. Saved tensors are
    materialized on CPU.
    """
    payload = {
        "kind": ARTIFACT_KIND,
        "x_fn_by_ciname": {k: torch.as_tensor(v).detach().cpu() for k, v in x_fn_by_ciname.items()},
        "function_genes": list(metadata["function_genes"]),
        "function_node_names": list(metadata["function_node_names"]),
        "cell_iname_order": list(metadata["cell_iname_order"]),
        "cell_modelid_order": list(metadata.get("cell_modelid_order", [])),
        "dropped_cell_inames": list(metadata.get("dropped_cell_inames", [])),
        "dropped_cell_inames_no_expression": list(
            metadata.get("dropped_cell_inames_no_expression", [])
        ),
        "activity_dim": int(metadata["activity_dim"]),
        # Default preserves backward-compatible behavior for callers that
        # built artifacts via the pre-multi-feature API (expression only).
        "activity_features": list(metadata.get("activity_features", ["expr"])),
        "line_vocab_order": list(metadata.get("line_vocab_order", [])),
    }
    torch.save(payload, path)


def load_node_activity_artifact(
    path: str,
    node_names_dict: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict:
    """Load an artifact written by :func:`save_node_activity_artifact`.

    When ``node_names_dict`` is provided, validate that the artifact's
    ``function_node_names`` matches the current bionetwork's function-node
    ordering (the ``x_fn`` rows are positionally aligned to it, so a
    mismatch would silently miswire features to nodes).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != ARTIFACT_KIND:
        raise ValueError(
            f"Unrecognized node_activity artifact kind={payload.get('kind')!r} at {path}. "
            f"Expected {ARTIFACT_KIND!r}."
        )

    if node_names_dict is not None:
        bionet_fn_names = list(node_names_dict["function"])
        art_fn_names = list(payload["function_node_names"])
        if bionet_fn_names != art_fn_names:
            raise ValueError(
                "node_activity artifact function-node ordering disagrees with the "
                "current bionetwork; rebuild the artifact. "
                f"Artifact has {len(art_fn_names)} function nodes, bionet has "
                f"{len(bionet_fn_names)}."
            )

    return payload
