"""Perturbation -> function-node edge resolution for make_bio_network."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

_GENETIC_PREFIXES = ("xpr_", "oe_", "sh_")


def _is_genetic(pert_name: str) -> bool:
    return any(pert_name.startswith(p) for p in _GENETIC_PREFIXES)


def _gene_from_token(pert_name: str) -> str:
    for prefix in _GENETIC_PREFIXES:
        if pert_name.startswith(prefix):
            return pert_name[len(prefix) :]
    return pert_name


_MISSING_CMAP = ("NA", "nan", "", "None")


def _build_cmap_lookup(cond_info: pd.DataFrame) -> dict[str, str]:
    """Build a ``pert_name -> cmap_name`` dict once (first non-null wins).

    Avoids the O(n_perts x n_cond_info) full-column scan that a per-token
    ``cond_info.loc[...]`` filter incurs.
    """
    if "pert_name" not in cond_info.columns or "cmap_name" not in cond_info.columns:
        return {}
    sub = cond_info[["pert_name", "cmap_name"]].dropna(subset=["pert_name"])
    # Keep the first occurrence per pert_name (matches prior .iloc[0] behavior).
    sub = sub.drop_duplicates(subset=["pert_name"], keep="first")
    lookup: dict[str, str] = {}
    for pert_name, cmap in zip(sub["pert_name"].astype(str), sub["cmap_name"]):
        if pd.notna(cmap) and str(cmap) not in _MISSING_CMAP:
            lookup[pert_name] = str(cmap)
    return lookup


def _resolve_gene_symbol(pert_name: str, cmap_lookup: dict[str, str]) -> str | None:
    gene = cmap_lookup.get(str(pert_name))
    if gene:
        return gene
    token_gene = _gene_from_token(pert_name)
    return token_gene if token_gene else None


def resolve_chem_target_edges(
    pert_names: Sequence[str],
    cond_info: pd.DataFrame,
    compoundinfo: pd.DataFrame,
    targetome: pd.DataFrame,
    uniprot_to_func: pd.DataFrame,
    func_names: set[str],
    max_kd: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map chemical pert_name tokens to DRUG__ -> PROTEIN__ edges via DTI.

    Returns
    -------
    edges : DataFrame with columns src, dst
    dropped : DataFrame audit of perts with no resolvable edges
    """
    chem_names = [p for p in pert_names if not _is_genetic(p)]
    if not chem_names:
        return (
            pd.DataFrame(columns=["src", "dst"]),
            pd.DataFrame(columns=["pert_name", "reason"]),
        )

    name_to_brd = (
        cond_info.loc[cond_info["pert_name"].isin(chem_names), ["pert_name", "pert_id"]]
        .drop_duplicates(subset=["pert_name"])
        .set_index("pert_name")["pert_id"]
        .astype(str)
        .to_dict()
    )

    clue_mapping = compoundinfo[["inchi_key", "pert_id"]].drop_duplicates()
    tge = targetome[
        targetome["assay_type"].isin(["Kd", "Ki"])
        & targetome["assay_relation"].isin(["<", "<=", "="])
        & (targetome["assay_value"] <= max_kd)
    ].copy()
    tge = tge.merge(clue_mapping, on="inchi_key", how="inner")
    if "pert_id" not in tge.columns:
        if "pert_id_y" in tge.columns:
            tge = tge.rename(columns={"pert_id_y": "pert_id"})
        elif "pert_id_x" in tge.columns:
            tge = tge.rename(columns={"pert_id_x": "pert_id"})
        else:
            raise KeyError("merged targetome/compoundinfo frame lacks pert_id column")

    brd_ids = set(name_to_brd.values())
    tge = tge[tge["pert_id"].astype(str).isin(brd_ids)]
    tge = tge[["pert_id", "uniprot_id"]].drop_duplicates()
    tge["uniprot_id"] = tge["uniprot_id"].astype(str).str.strip()

    u2fn = uniprot_to_func.loc[
        uniprot_to_func["node_kind"] == "PROTEIN",
        ["uniprot", "func_name"],
    ].drop_duplicates(subset=["uniprot", "func_name"])
    u2fn = u2fn[u2fn["func_name"].isin(func_names)]

    merged = tge.merge(
        u2fn,
        left_on="uniprot_id",
        right_on="uniprot",
        how="left",
    )

    brd_to_targets: dict[str, list[str]] = {}
    for brd, grp in merged.dropna(subset=["func_name"]).groupby("pert_id"):
        brd_to_targets[str(brd)] = sorted(grp["func_name"].unique().tolist())

    records = []
    dropped = []
    for pert_name in chem_names:
        brd = name_to_brd.get(pert_name)
        if brd is None:
            dropped.append({"pert_name": pert_name, "reason": "no_brd_in_cond_info"})
            continue
        targets = brd_to_targets.get(brd, [])
        if not targets:
            dropped.append({"pert_name": pert_name, "reason": "no_dti_targets"})
            continue
        src = "DRUG__" + pert_name
        for dst in targets:
            records.append({"src": src, "dst": dst})

    edges = (
        pd.DataFrame(records, columns=["src", "dst"]).drop_duplicates()
        if records
        else pd.DataFrame(columns=["src", "dst"])
    )
    dropped_df = (
        pd.DataFrame(dropped, columns=["pert_name", "reason"])
        if dropped
        else pd.DataFrame(columns=["pert_name", "reason"])
    )
    return edges, dropped_df


def resolve_genetic_edges(
    pert_names: Sequence[str],
    cond_info: pd.DataFrame,
    func_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map genetic pert_name tokens to DRUG__ -> RNA__/PROTEIN__ edges.

    Wiring rules:
    - sh_, oe_ -> RNA__<gene> only
    - xpr_ -> RNA__<gene> and PROTEIN__<gene>
    """
    genetic = [p for p in pert_names if _is_genetic(p)]
    records = []
    dropped = []

    cmap_lookup = _build_cmap_lookup(cond_info)

    for pert_name in genetic:
        gene = _resolve_gene_symbol(pert_name, cmap_lookup)
        if not gene:
            dropped.append({"pert_name": pert_name, "reason": "no_gene_symbol"})
            continue

        src = "DRUG__" + pert_name
        targets: list[str] = []
        if pert_name.startswith("sh_") or pert_name.startswith("oe_"):
            targets = [f"RNA__{gene}"]
        elif pert_name.startswith("xpr_"):
            targets = [f"RNA__{gene}", f"PROTEIN__{gene}"]

        added = False
        for dst in targets:
            if dst in func_names:
                records.append({"src": src, "dst": dst})
                added = True
        if not added:
            dropped.append({"pert_name": pert_name, "reason": "target_not_in_func_graph"})

    edges = (
        pd.DataFrame(records, columns=["src", "dst"]).drop_duplicates()
        if records
        else pd.DataFrame(columns=["src", "dst"])
    )
    dropped_df = (
        pd.DataFrame(dropped, columns=["pert_name", "reason"])
        if dropped
        else pd.DataFrame(columns=["pert_name", "reason"])
    )
    return edges, dropped_df


def resolve_perturbation_edges(
    pert_names: Sequence[str],
    cond_info: pd.DataFrame,
    compoundinfo: pd.DataFrame,
    targetome: pd.DataFrame,
    uniprot_to_func: pd.DataFrame,
    func_names: Iterable[str],
    max_kd: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve chemical + genetic input edges and concatenate dropped audits."""
    func_set = set(func_names)
    print('\tresoliving chemical edges...')
    chem_edges, chem_dropped = resolve_chem_target_edges(
        pert_names,
        cond_info,
        compoundinfo,
        targetome,
        uniprot_to_func,
        func_set,
        max_kd,
    )
    print('\tresoliving genetic edges...')
    gen_edges, gen_dropped = resolve_genetic_edges(pert_names, cond_info, func_set)
    edges = pd.concat([chem_edges, gen_edges], ignore_index=True).drop_duplicates()
    dropped = pd.concat([chem_dropped, gen_dropped], ignore_index=True)
    return edges, dropped
