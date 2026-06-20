"""Canonical landmark gene table and function-graph node resolution.

Builds a per-L1000-gene record (LINCS symbol, Entrez id, Ensembl id, aliases,
UniProt accessions) and resolves each landmark to an ``RNA__*`` function node
in the bionetwork using direct symbol match, aliases, and UniProt bridges.
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ARTIFACT_KIND = "canonical_genes_v1"
_RNA_PREFIX = "RNA__"
_PROTEIN_PREFIX = "PROTEIN__"


def func_name_set(func_nodes: pd.DataFrame) -> set[str]:
    """Return the set of function-graph node names."""
    return set(func_nodes["func_name"].astype(str))


def rna_nodes_by_symbol(func_nodes: pd.DataFrame) -> Dict[str, str]:
    """Map uppercased gene_symbol suffix to ``RNA__*`` func_name."""
    rna = func_nodes[func_nodes["func_name"].astype(str).str.startswith(_RNA_PREFIX)]
    out: Dict[str, str] = {}
    for _, row in rna.iterrows():
        sym = str(row["gene_symbol"]).upper()
        out.setdefault(sym, str(row["func_name"]))
    return out


def uniprot_to_func_names(
    func_nodes: pd.DataFrame,
    prefix: str = _PROTEIN_PREFIX,
) -> Tuple[Dict[str, List[str]], List[Dict[str, object]]]:
    """Map UniProt accession to ``func_name`` list for nodes with the given prefix.

    Returns
    -------
    mapping
        Accession -> one or more ``func_name`` values (all matching nodes).
    ambiguous
        Rows with keys ``uniprot``, ``func_names`` when multiple nodes share one
        accession (subset of ``mapping`` where len(func_names) > 1).
    """
    sub = func_nodes[func_nodes["func_name"].astype(str).str.startswith(prefix)].copy()
    sub = sub[sub["uniprot"].notna()]
    if sub.empty:
        return {}, []

    sub["uniprot"] = sub["uniprot"].astype(str).str.strip()
    mapping: Dict[str, List[str]] = {}
    ambiguous: List[Dict[str, object]] = []

    for up, group in sub.groupby("uniprot"):
        names = sorted(group["func_name"].astype(str).unique().tolist())
        mapping[up] = names
        if len(names) > 1:
            ambiguous.append({"uniprot": up, "func_names": names})

    return mapping, ambiguous


def uniprot_to_func_name(
    func_nodes: pd.DataFrame,
    prefix: str = _PROTEIN_PREFIX,
) -> Tuple[Dict[str, str], List[Dict[str, object]]]:
    """Map UniProt accession to a single ``func_name`` (unambiguous accessions only).

    See :func:`uniprot_to_func_names` for the multi-target mapping used by DTI.
    """
    mapping, ambiguous = uniprot_to_func_names(func_nodes, prefix=prefix)
    single = {up: names[0] for up, names in mapping.items() if len(names) == 1}
    return single, ambiguous


def build_uniprot_to_protein_map(
    func_nodes: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], List[Dict[str, object]]]:
    """DTI mapping: UniProt -> all matching ``PROTEIN__*`` func_name nodes."""
    return uniprot_to_func_names(func_nodes, prefix=_PROTEIN_PREFIX)


def _normalize_uniprot(acc: object) -> Optional[str]:
    if acc is None or (isinstance(acc, float) and pd.isna(acc)):
        return None
    s = str(acc).strip()
    return s if s and s.lower() != "nan" else None


def _batch_uniprot_for_landmarks(
    symbols: Sequence[str],
    symbol_to_ensembl: Dict[str, str],
    verbose: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Batch-map landmark Ensembl IDs (primary) and symbols (fallback) via UniProt REST.

    On a hard network failure (after the client's retry/backoff is exhausted)
    this degrades gracefully: it logs a warning and returns empty maps so the
    bionetwork build can still complete using geneinfo Ensembl IDs and direct/
    alias symbol resolution. UniProt enrichment (bridges/synonyms) is then
    skipped rather than aborting the whole run.
    """
    from lincs_gsnn.proc.uniprot_client import (
        NetworkError,
        idmapping_batch,
        mapping_to_dict,
    )

    to_db = "UniProtKB"

    ens_clean = list(dict.fromkeys(symbol_to_ensembl.values()))
    ens_map: Dict[str, List[str]] = {}
    if ens_clean:
        if verbose:
            print(f"  Ensembl -> {to_db} ({len(ens_clean)} ids)...")
        try:
            ens_map = mapping_to_dict(
                idmapping_batch(ens_clean, from_db="Ensembl", to_db=to_db, verbose=verbose)
            )
        except NetworkError as exc:
            print(
                f"  WARNING: UniProt Ensembl idmapping unreachable ({type(exc).__name__}: {exc}). "
                "Continuing without UniProt enrichment; landmark resolution will rely on "
                "direct/alias symbol matching only.",
                flush=True,
            )
            return {}, {}

    # Gene_Name only for symbols without Ensembl (avoids many ortholog hits per symbol).
    symbols_needing_name = [s for s in symbols if s not in symbol_to_ensembl]
    sym_map: Dict[str, List[str]] = {}
    if symbols_needing_name:
        if verbose:
            print(f"  Gene_Name -> {to_db} ({len(symbols_needing_name)} symbols without Ensembl)...")
        try:
            sym_map = mapping_to_dict(
                idmapping_batch(symbols_needing_name, from_db="Gene_Name", to_db=to_db, verbose=verbose)
            )
        except NetworkError as exc:
            print(
                f"  WARNING: UniProt Gene_Name idmapping unreachable ({type(exc).__name__}: {exc}). "
                "Continuing with Ensembl-derived UniProt accessions only.",
                flush=True,
            )
            sym_map = {}
    return sym_map, ens_map


def _aliases_from_uniprot_names(
    lincs_symbol: str,
    uniprot_ids: Sequence[str],
    acc_to_names: Dict[str, List[str]],
) -> List[str]:
    """Collect symbol + UniProt-reported gene names as alias set."""
    aliases = {lincs_symbol}
    for up in uniprot_ids:
        for name in acc_to_names.get(up, []):
            if name:
                aliases.add(str(name))
    return sorted(aliases)


def _filter_uniprot_accessions(
    accessions: Sequence[str],
    func_uniprots: Optional[set[str]] = None,
) -> List[str]:
    """Prefer graph-known and canonical-style accessions (drop TrEMBL A0A* when possible)."""
    accs = [str(a).strip() for a in accessions if str(a).strip()]
    if func_uniprots:
        in_graph = [a for a in accs if a in func_uniprots]
        if in_graph:
            accs = in_graph
    canonical_style = [a for a in accs if not str(a).startswith("A0A")]
    return list(dict.fromkeys(canonical_style if canonical_style else accs))


def build_canonical_gene_table(
    gene_names: Sequence[str],
    geneinfo_path: str,
    verbose: bool = True,
    fetch_uniprot_synonyms: bool = True,
    func_nodes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build one canonical row per L1000 landmark gene.

    Parameters
    ----------
    gene_names
        Ordered landmark symbols (e.g. from ``gene_names.csv``).
    geneinfo_path
        Path to ``geneinfo_beta.txt``; rows with ``feature_space == landmark``
        supply ``gene_id`` and ``ensembl_id``.
    """
    symbols = [str(s) for s in gene_names]
    gi = pd.read_csv(geneinfo_path, sep="\t", low_memory=False)
    land = gi.loc[gi["feature_space"] == "landmark", ["gene_symbol", "gene_id", "ensembl_id"]].copy()
    land["gene_symbol"] = land["gene_symbol"].astype(str)
    sym_to_info = land.set_index("gene_symbol", drop=False)

    symbol_to_ensembl: Dict[str, str] = {}
    for s in symbols:
        if s not in sym_to_info.index:
            continue
        info = sym_to_info.loc[s]
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        ens = info["ensembl_id"]
        if ens is not None and not (isinstance(ens, float) and pd.isna(ens)):
            symbol_to_ensembl[s] = str(ens).strip()

    if verbose:
        print(
            "canonical_genes: batch UniProt idmapping "
            f"(Ensembl primary, Gene_Name fallback; n_ensembl={len(symbol_to_ensembl)})..."
        )
    sym_map, ens_map = _batch_uniprot_for_landmarks(symbols, symbol_to_ensembl, verbose)

    missing_geneinfo = [s for s in symbols if s not in sym_to_info.index]

    # Optional: restrict each landmark's UniProt list to accessions that appear on
    # function-graph nodes (typically 1–2 per gene). NOT used to drive synonym fetch
    # over the whole graph (~10k accessions).
    func_uniprots: Optional[set[str]] = None
    if func_nodes is not None and len(func_nodes):
        func_uniprots = {
            str(u).strip()
            for u in func_nodes["uniprot"].dropna().astype(str)
            if str(u).strip() and str(u).lower() != "nan"
        }

    # Pass 1: per-landmark UniProt IDs (Ensembl-first, Gene_Name fallback).
    landmark_uniprots: Dict[str, List[str]] = {}
    for lincs_symbol in symbols:
        raw_ids: List[str] = []
        ens_key = symbol_to_ensembl.get(lincs_symbol)
        if ens_key:
            raw_ids.extend(ens_map.get(ens_key, []))
        if not raw_ids:
            raw_ids.extend(sym_map.get(lincs_symbol, []))
        landmark_uniprots[lincs_symbol] = _filter_uniprot_accessions(raw_ids, func_uniprots)

    # Pass 2: synonyms only for landmark UniProt accessions (~978–2k), never all func_nodes.
    landmark_accs = list(
        dict.fromkeys(up for ups in landmark_uniprots.values() for up in ups)
    )
    acc_to_names: Dict[str, List[str]] = {}
    if fetch_uniprot_synonyms and landmark_accs:
        from lincs_gsnn.proc.uniprot_client import (
            NetworkError,
            fetch_gene_names_for_accessions,
        )

        if verbose:
            print(
                f"canonical_genes: fetching UniProt synonyms for {len(landmark_accs)} "
                f"landmark accessions ({len(symbols)} genes)..."
            )
        try:
            acc_to_names = fetch_gene_names_for_accessions(landmark_accs, verbose=verbose)
        except NetworkError as exc:
            print(
                f"canonical_genes: WARNING UniProt synonym fetch unreachable "
                f"({type(exc).__name__}: {exc}). Continuing without synonym aliases.",
                flush=True,
            )
            acc_to_names = {}

    rows = []
    for lincs_symbol in symbols:
        if lincs_symbol in sym_to_info.index:
            info = sym_to_info.loc[lincs_symbol]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            gene_id = info["gene_id"]
            ensembl_id = info["ensembl_id"]
        else:
            gene_id = pd.NA
            ensembl_id = pd.NA

        uniprot_ids = landmark_uniprots[lincs_symbol]
        aliases = _aliases_from_uniprot_names(lincs_symbol, uniprot_ids, acc_to_names)

        rows.append(
            {
                "lincs_symbol": lincs_symbol,
                "gene_id": gene_id,
                "ensembl_id": ensembl_id,
                "aliases": ";".join(aliases),
                "uniprot_ids": ";".join(uniprot_ids),
            }
        )

    if verbose and missing_geneinfo:
        print(
            f"canonical_genes: {len(missing_geneinfo)} symbol(s) missing from "
            f"geneinfo landmark table (first 5): {missing_geneinfo[:5]}"
        )
    if verbose:
        n_with_up = sum(1 for r in rows if r["uniprot_ids"])
        print(f"canonical_genes: {n_with_up}/{len(rows)} landmarks with >=1 UniProt accession")

    return pd.DataFrame(rows)


def _parse_semicolon_list(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def resolve_lincs_to_rna_node(
    lincs_symbol: str,
    canonical_row: Mapping[str, object],
    func_nodes: pd.DataFrame,
    *,
    fn_set: Optional[set[str]] = None,
    rna_by_sym: Optional[Dict[str, str]] = None,
    uniprot_to_rna: Optional[Dict[str, List[str]]] = None,
    uniprot_to_protein: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[str], str]:
    """Resolve a LINCS landmark symbol to an ``RNA__*`` function node.

    Returns
    -------
    func_name or None
        Resolved ``RNA__*`` node name, or None if not found.
    method
        Resolution method tag for reporting.
    """
    lincs_symbol = str(lincs_symbol)
    if fn_set is None:
        fn_set = func_name_set(func_nodes)
    if rna_by_sym is None:
        rna_by_sym = rna_nodes_by_symbol(func_nodes)
    if uniprot_to_rna is None:
        uniprot_to_rna, _ = uniprot_to_func_names(func_nodes, prefix=_RNA_PREFIX)
    if uniprot_to_protein is None:
        uniprot_to_protein, _ = uniprot_to_func_names(func_nodes, prefix=_PROTEIN_PREFIX)

    direct = f"{_RNA_PREFIX}{lincs_symbol}"
    if direct in fn_set:
        return direct, "direct_symbol"

    for alias in _parse_semicolon_list(canonical_row.get("aliases")):
        node = f"{_RNA_PREFIX}{alias}"
        if node in fn_set:
            return node, "alias_symbol"

        key = alias.upper()
        if key in rna_by_sym:
            return rna_by_sym[key], "alias_symbol_upper"

    for up in _parse_semicolon_list(canonical_row.get("uniprot_ids")):
        up = _normalize_uniprot(up)
        if not up:
            continue
        if up in uniprot_to_rna:
            return uniprot_to_rna[up][0], "uniprot_rna"

        if up in uniprot_to_protein:
            for prot_name in uniprot_to_protein[up]:
                sym = prot_name.split("__", 1)[1]
                candidate = f"{_RNA_PREFIX}{sym}"
                if candidate in fn_set:
                    return candidate, "uniprot_protein_bridge"

    return None, "unresolved"


def resolve_all_lincs_to_rna(
    canonical: pd.DataFrame,
    func_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve every landmark; return a resolution report dataframe."""
    records = []
    fn_set = func_name_set(func_nodes)
    rna_by_sym = rna_nodes_by_symbol(func_nodes)
    uniprot_to_rna, _ = uniprot_to_func_names(func_nodes, prefix=_RNA_PREFIX)
    uniprot_to_protein, _ = uniprot_to_func_names(func_nodes, prefix=_PROTEIN_PREFIX)

    for _, row in canonical.iterrows():
        lincs_symbol = str(row["lincs_symbol"])
        resolved, method = resolve_lincs_to_rna_node(
            lincs_symbol,
            row,
            func_nodes,
            fn_set=fn_set,
            rna_by_sym=rna_by_sym,
            uniprot_to_rna=uniprot_to_rna,
            uniprot_to_protein=uniprot_to_protein,
        )
        records.append(
            {
                "lincs_symbol": lincs_symbol,
                "resolved_rna_node": resolved if resolved is not None else f"{_RNA_PREFIX}{lincs_symbol}",
                "method": method,
                "in_function_graph": resolved in fn_set if resolved is not None else False,
            }
        )

    return pd.DataFrame(records)


def build_lincs_gene_edges(
    canonical: pd.DataFrame,
    func_nodes: pd.DataFrame,
    edge_kind: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build GENE<->RNA edges and resolution report.

    Parameters
    ----------
    edge_kind
        ``'output'`` for RNA->GENE (function to output), ``'input'`` for
        GENE->RNA (input to function).
    """
    resolution = resolve_all_lincs_to_rna(canonical, func_nodes)
    src_list, dst_list = [], []

    for _, res in resolution.iterrows():
        lincs = res["lincs_symbol"]
        rna = res["resolved_rna_node"]
        gene = f"GENE__{lincs}"
        if edge_kind == "output":
            src_list.append(rna)
            dst_list.append(gene)
        elif edge_kind == "input":
            src_list.append(gene)
            dst_list.append(rna)
        else:
            raise ValueError(f"edge_kind must be 'input' or 'output', got {edge_kind!r}")

    edges = pd.DataFrame({"src": src_list, "dst": dst_list})
    gene_names = [f"GENE__{s}" for s in canonical["lincs_symbol"].astype(str)]
    return edges, resolution


def save_canonical_gene_artifact(path: str, table: pd.DataFrame) -> None:
    """Write canonical gene table to CSV."""
    out = table.copy()
    out["kind"] = ARTIFACT_KIND
    out.to_csv(path, index=False)


def load_canonical_gene_artifact(path: str) -> pd.DataFrame:
    """Load canonical gene table written by :func:`save_canonical_gene_artifact`."""
    df = pd.read_csv(path)
    if "kind" in df.columns:
        kind = df["kind"].iloc[0] if len(df) else None
        if kind != ARTIFACT_KIND:
            raise ValueError(
                f"Unrecognized canonical_genes artifact kind={kind!r} at {path}; "
                f"expected {ARTIFACT_KIND!r}."
            )
        df = df.drop(columns=["kind"])
    return df
