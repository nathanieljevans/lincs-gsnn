#!/usr/bin/env python
"""Smoke validation for UniProt DTI + canonical L1000 mapping.

Runs without OmniPath when ``--func-nodes-csv`` points at a saved func_nodes table
(from a prior get_bio_interactions call). Otherwise calls get_bio_interactions
(requires network).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch

# repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lincs_gsnn.proc.canonical_genes import (
    build_canonical_gene_table,
    build_lincs_gene_edges,
    load_canonical_gene_artifact,
)
from lincs_gsnn.proc.drug_accessibility import compute_drug_accessible_output_genes
from gsnn.proc.construct import GSNNNetworkConstructor

_FUNC_EDGE_CACHE_COLS = [
    "src", "dst", "source_uniprot", "target_uniprot", "edge_type",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/exacloud/gscratch/mcweeney_lab/evans/data")
    p.add_argument(
        "--meta",
        default="/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-traj/runs/exp/default_v03/output/predict_grid",
    )
    p.add_argument("--out", default=os.path.join(ROOT, ".tmp_bionet_validate"))
    p.add_argument("--old-bionet", default="/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/lincs-gsnn/exp_01/bionetwork/bionetwork.pt")
    p.add_argument("--func-nodes-csv", default=None, help="optional cached func_nodes CSV")
    p.add_argument(
        "--func-edges-csv",
        default=None,
        help="optional cached func_edges CSV (src,dst,source_uniprot,target_uniprot,edge_type)",
    )
    p.add_argument("--max-dti-kd", type=float, default=100.0)
    p.add_argument("--filter-depth", type=int, default=8)
    p.add_argument("--skip-full-bionet", action="store_true")
    return p.parse_args()


def _load_targetome_smoke(data, meta, uniprot_symbol_map, max_dti_kd):
    clue_mapping = pd.read_csv(f"{data}/compoundinfo_beta.txt", sep="\t")[["inchi_key", "pert_id"]].drop_duplicates()
    drugs = pd.read_csv(f"{meta}/pert_ids.csv").pert_id.tolist()
    tge = pd.read_csv(f"{data}/targetome_extended-01-23-25.csv").merge(clue_mapping, on="inchi_key", how="inner")
    tge = tge[tge.pert_id.isin(drugs)]
    tge = tge[tge.assay_type.isin(["Kd", "Ki"]) & tge.assay_relation.isin(["=", "<", "<="])]
    tge = tge[tge.assay_value <= max_dti_kd]
    tge = tge[["pert_id", "uniprot_id"]].drop_duplicates()
    tge["uniprot_id"] = tge["uniprot_id"].astype(str).str.strip()

    u2fn = (
        uniprot_symbol_map.loc[
            uniprot_symbol_map["node_kind"] == "PROTEIN",
            ["uniprot", "func_name"],
        ]
        .drop_duplicates()
    )
    merged = tge.merge(u2fn, left_on="uniprot_id", right_on="uniprot", how="left")
    n_unmapped_up = int(merged.loc[merged["func_name"].isna(), "uniprot_id"].nunique())

    mapped = merged.dropna(subset=["func_name"]).copy()
    mapped["src"] = "DRUG__" + mapped["pert_id"].astype(str)
    edges = (
        mapped[["src", "func_name"]]
        .rename(columns={"func_name": "dst"})
        .drop_duplicates()
    )
    return edges, n_unmapped_up


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    geneinfo = os.path.join(args.data, "geneinfo_beta.txt")
    lincs_symbols = pd.read_csv(f"{args.meta}/gene_names.csv")["gene_names"].astype(str).tolist()

    if args.func_nodes_csv and os.path.isfile(args.func_nodes_csv):
        print(f"Loading cached func_nodes from {args.func_nodes_csv}")
        func_nodes = pd.read_csv(args.func_nodes_csv)
        if args.func_edges_csv and os.path.isfile(args.func_edges_csv):
            func_edges = pd.read_csv(args.func_edges_csv)
            missing = set(_FUNC_EDGE_CACHE_COLS) - set(func_edges.columns)
            if missing:
                raise SystemExit(
                    f"--func-edges-csv missing columns {sorted(missing)}; "
                    f"re-run without cache or use edges with {_FUNC_EDGE_CACHE_COLS}"
                )
            from gsnn.proc.bio import build_uniprot_symbol_map

            uniprot_symbol_map = build_uniprot_symbol_map(func_edges)
        else:
            raise SystemExit("--func-edges-csv required with --func-nodes-csv")
    else:
        from gsnn.proc.bio import get_bio_interactions

        print("Calling get_bio_interactions (network required)...")
        func_nodes, func_edges, uniprot_symbol_map = get_bio_interactions(
            include_dorothea=True,
            include_omnipath=True,
            dorothea_levels=["A", "B"],
            complex_handling="remove",
            gene_symbol=True,
            verbose=True,
            return_uniprot_map=True,
        )
        func_nodes.to_csv(os.path.join(args.out, "func_nodes_cache.csv"), index=False)
        func_edges[_FUNC_EDGE_CACHE_COLS].to_csv(
            os.path.join(args.out, "func_edges_cache.csv"), index=False,
        )

    assert list(func_nodes.columns) == ["func_name", "uniprot", "gene_symbol"], func_nodes.columns

    n_prot_up = int(
        uniprot_symbol_map.loc[uniprot_symbol_map["node_kind"] == "PROTEIN", "uniprot"].nunique()
    )
    print(f"PROTEIN__ uniprot map (m:m): {n_prot_up} accessions")

    dti_edges, n_unmapped_up = _load_targetome_smoke(
        args.data, args.meta, uniprot_symbol_map, args.max_dti_kd,
    )
    print(
        f"DTI edges: {len(dti_edges)} from {dti_edges['src'].nunique()} drugs "
        f"({n_unmapped_up} unmapped uniprot accessions)"
    )

    # canonical + L1000
    print("Building canonical table (PyPath for aliases/uniprot)...")
    canonical = build_canonical_gene_table(lincs_symbols, geneinfo, verbose=True)
    canonical.to_csv(os.path.join(args.out, "canonical_genes.csv"), index=False)

    out_edges, resolution = build_lincs_gene_edges(canonical, func_nodes, edge_kind="output")
    resolution.to_csv(os.path.join(args.out, "lincs_fn_resolution.csv"), index=False)
    print("L1000 resolution:")
    print(resolution["method"].value_counts().to_string())
    print(f"in_function_graph: {resolution['in_function_graph'].sum()} / {len(resolution)}")

    # spot check MAP2K1
    mek = func_nodes.loc[func_nodes["gene_symbol"] == "MAP2K1"]
    print(f"MAP2K1 func_nodes rows:\n{mek.to_string(index=False)}")

    if args.skip_full_bionet:
        print("Skipping full bionetwork build (--skip-full-bionet)")
        return 0

    gene_names = [f"GENE__{s}" for s in canonical["lincs_symbol"]]
    in_edges, _ = build_lincs_gene_edges(canonical, func_nodes, edge_kind="input")

    constructor = GSNNNetworkConstructor(depth=args.filter_depth, verbose=True)
    data = constructor.build(
        input_edges=pd.concat([dti_edges, in_edges], ignore_index=True),
        output_edges=out_edges,
        function_edges=func_edges[["src", "dst"]],
        mediator_edges=None,
        input_names=gene_names,
        output_names=gene_names,
    )
    mask = compute_drug_accessible_output_genes(data)
    n_acc = int(mask.sum())
    print(f"Drug-accessible outputs: {n_acc} / {len(gene_names)}")

    torch.save(data, os.path.join(args.out, "bionetwork.pt"))

    if os.path.isfile(args.old_bionet):
        old = torch.load(args.old_bionet, map_location="cpu", weights_only=False)
        old_mask = getattr(old, "drug_accessible_output_genes", None)
        if old_mask is None:
            old_mask = compute_drug_accessible_output_genes(old)
        old_acc = int(old_mask.sum())
        old_drugs = sum(1 for n in old.node_names_dict["input"] if str(n).startswith("DRUG__"))
        new_drugs = sum(1 for n in data.node_names_dict["input"] if str(n).startswith("DRUG__"))
        print(f"OLD bionet: drug-accessible {old_acc}/{len(old.node_names_dict['output'])}, drugs {old_drugs}")
        print(f"NEW bionet: drug-accessible {n_acc}/{len(gene_names)}, drugs {new_drugs}")

    print("Smoke validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
