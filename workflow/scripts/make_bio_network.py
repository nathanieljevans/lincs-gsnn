'''
Constructs a heterogeneous biological network graph for LINCS-GSNN project. 

This script integrates multiple biological data sources to create a comprehensive network 
representation that captures drug-target interactions, biological pathway relationships, 
and gene regulatory networks. The resulting graph serves as the foundation for a 
graph structured neural network that predicts drug-induced gene expression changes.

Data Sources:
- Drug-target interactions (DTI): Filtered from Targetome Extended database using 
  affinity thresholds (Kd/Ki ≤ X nM) and direct binding assays
- Biological interactions: Protein-protein interactions, transcription factor binding, 
  and regulatory networks from OmniPath database
- LINCS landmark genes: Core gene set for expression prediction
- Cell line metadata: Experimental cellular context for drug treatments 

Network Structure:
The heterogeneous graph contains three node types:
1. Input nodes: Drugs (DRUG__), cell lines (LINE__), and genes (GENE__)
2. Function nodes: Proteins (PROTEIN__) and RNAs (RNA__) representing biological entities; Note DNA and RNA are collapsed into a single "RNA" node. 
3. Output nodes: Genes (GENE__) for which expression changes are predicted

Edge types include:
- Input→Function: Drug-target binding, cell line context, gene-protein mapping
- Function→Function: Biological interactions (PPI, TF binding, regulatory networks)
- Function→Output: Protein/RNA to gene expression mapping

The network undergoes filtering to ensure connectivity between drugs and landmark genes 
through biological pathways, with configurable depth for upstream drug targets and 
downstream gene effects.

Output:
- PyTorch Geometric HeteroData object saved as 'bionetwork.pt'
- Contains edge indices and node name mappings for graph neural network training
- Includes comprehensive validation checks for network integrity

This network architecture enables the GSNN to learn drug-specific gene expression 
responses by propagating information through biologically relevant pathways.
'''



import argparse
import os
import time
import pandas as pd 
import torch_geometric as pyg
import numpy as np
import torch

from gsnn.proc.bio import get_bio_interactions 
from gsnn.proc.construct import GSNNNetworkConstructor
from lincs_gsnn.proc.canonical_genes import (
    build_canonical_gene_table,
    build_lincs_gene_edges,
    save_canonical_gene_artifact,
)
from lincs_gsnn.proc.drug_accessibility import compute_drug_accessible_output_genes
from lincs_gsnn.proc.graph import remap_removed_edges_names, simplify_function_layer


def _log(msg: str) -> None:
    """Print a flushed, timestamped progress line (visible under Snakemake/nohup)."""
    ts = time.strftime("%H:%M:%S")
    print(f"[make_bio_network {ts}] {msg}", flush=True)



def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                   help="path to data directory")
    parser.add_argument("--meta",               type=str,               default='../../../predict_grid/',                   help="path to metadata directory")
    parser.add_argument("--extdata",            type=str,               default='../../../extdata/',                help="path to extdata directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                help="path to data directory")
    parser.add_argument("--geneinfo_path",      type=str,               default=None,                               help="path to geneinfo_beta.txt (default: {data}/geneinfo_beta.txt)")
    parser.add_argument("--no_uniprot_synonyms", action='store_true',  default=False,                              help="skip UniProt synonym fetch when building canonical_genes (faster; fewer alias_symbol resolutions)")
    parser.add_argument("--include_tf_mirna",       action='store_true', default=False, help="include TF/miRNA interactions (TF->miRNA, miRNA->mRNA) in the function graph")
    parser.add_argument("--include_pathway_extra",  action='store_true', default=False, help="include extra pathway interactions from OmniPath without literature references")
    parser.add_argument("--include_kinase_extra",   action='store_true', default=False, help="include extra kinase-substrate interactions from OmniPath without literature references")
    parser.add_argument("--include_ligrec_extra",   action='store_true', default=False, help="include extra ligand-receptor interactions from OmniPath without literature references")
    parser.add_argument("--include_collecTRI",      action='store_true', default=False, help="include CollecTRI transcription factor regulons")
    parser.add_argument("--include_dorothea",       action='store_true', default=False, help="include DoRothEA TF regulons")
    parser.add_argument("--include_omnipath",       action='store_true', default=False, help="include OmniPath interactions")
    parser.add_argument('--dorothea_levels',    type=str,               default='ABCD',                               help='the dorothea levels to include in the function graph [A-D]')
    parser.add_argument("--complex_handling",   type=str,               default='link',                             help="how to handle protein complexes in interaction data (passed to get_bio_interactions)")
    parser.add_argument("--min_n_references",   type=int,               default=None,                               help="minimum number of literature references required per interaction (None = no filter)")
    parser.add_argument("--min_curation_effort", type=int,              default=None,                               help="minimum curation effort score required per interaction (None = no filter)")
    parser.add_argument("--max_dti_kd",         type=float,             default=1000.0,                             help="maximum DTI affinity (Kd) to include in the graph")
    parser.add_argument("--filter_depth",       type=int,               default=4,                                 help="the depth to search for upstream drugs and downstream lincs in the node filter process")
    parser.add_argument("--remove_output_edges", action='store_true',   default=False,                              help="remove output edges from the function graph")
    parser.add_argument("--n_edges_to_remove",  type=int,               default=1,                               help="the number of edges to remove per output gene when --remove_output_edges is set")
    parser.add_argument("--holdout_N_val_edges", type=int,            default=0,                                 help="number of function->function edges to hold out for validation (global random sample)")
    parser.add_argument("--holdout_N_test_edges", type=int,           default=0,                                 help="number of function->function edges to hold out for testing (global random sample)")
    parser.add_argument("--seed",               type=int,               default=42,                                help="the random seed to use for the edge removal process")
    parser.add_argument("--val_cells_per_drug", type=int,               default=1,                                 help="number of cell lines held out per drug for validation (default 1 of 11)")
    parser.add_argument("--no_cell_line_edges", action='store_true',    default=False,                              help="if set, do not include LINE->function (cell line) edges in the graph; LINE__ names are still force-included in node_names_dict['input'] so the dataset feature layout is unchanged")
    parser.add_argument("--include_gene_inputs", action='store_true',    default=False,                              help="if set, include GENE__ -> RNA__ input edges (legacy behavior). When off (default), GENE__ names are still force-included in node_names_dict['input'] for DXDTDataset feature-layout compatibility, but they have no outgoing edges and are therefore not pruning roots. Only DRUG__ nodes act as roots.")

    # ------------------------------------------------------------------
    # gene_norm artifact: per-gene control-population mean / std copied
    # from lincs-traj's gene_stats.dict, aligned to predict_grid/
    # gene_names.csv. Required by BIOGSNN's log1p-back degradation term.
    # ------------------------------------------------------------------
    parser.add_argument("--gene_stats_path", type=str, default=None,
                        help="path to lincs-traj's gene_stats.dict (per-gene control-population means/stds). "
                             "When provided, a versioned gene_norm.pt artifact is saved next to bionetwork.pt; "
                             "BIOGSNN requires this artifact at train/explain time.")
    parser.add_argument("--cond_info_path", type=str, default=None,
                        help="path to lincs-traj's cond_info.csv (pert_name <-> pert_id bridge and genetic cmap_name). "
                             "Defaults to cond_info.csv alongside --gene_stats_path when set.")

    # ------------------------------------------------------------------
    # Optional: build the per-cell-line node-activity (x_fn) artifact.
    # When --node_activity is NOT passed, nothing below changes on disk
    # (legacy artifact set is byte-identical).
    # ------------------------------------------------------------------
    parser.add_argument("--node_activity",      action='store_true',    default=False,                              help="if set, build a node_activity.pt artifact (per-cell x_fn lookup) alongside bionetwork.pt; required when training with --node_activity")
    parser.add_argument("--node_activity_eps",  type=float,             default=1e-6,                               help="stabilizer added to expression std-dev when z-scoring (node_activity only)")
    parser.add_argument("--node_activity_features", nargs='+', default=['expr'], choices=['expr', 'mut', 'is_protein', 'is_rna', 'is_mirna', 'cell_line'], help="per-function-node activity channels to include (in order). 'expr' = expression z-score; 'mut' = damaging-somatic-mutation indicator (0/1); 'is_protein'/'is_rna'/'is_mirna' = per-function-node binary masks derived from the bionet node name (constant across cell lines); 'cell_line' = one-hot over LINE__ input nodes broadcast to every function node. Channel order in the resulting (n_fn, activity_dim) tensor matches the order given here. Keep in sync with lincs_gsnn.proc.node_activity.ACTIVITY_FEATURE_BUILDERS.")

    # Optional function-layer graph simplification (legacy runs omit these flags).
    parser.add_argument("--simplify_degree_one", action="store_true", default=False,
                        help="contract pass-through function nodes with fixed in/out degree")
    parser.add_argument("--simplify_degree_one_in_degree", type=int, default=1)
    parser.add_argument("--simplify_degree_one_out_degree", type=int, default=1)
    parser.add_argument("--simplify_check_reachability", action="store_true", default=False,
                        help="log a warning if f2f reachability is not preserved after simplification")

    args = parser.parse_args()
    if args.geneinfo_path is None:
        args.geneinfo_path = os.path.join(args.data, "geneinfo_beta.txt")
    return args


def _write_dti_ambiguous_uniprot(uniprot_symbol_map: pd.DataFrame, out_dir: str) -> int:
    """Write diagnostic CSV for uniprots mapping to multiple PROTEIN__ nodes."""
    prot = uniprot_symbol_map.loc[uniprot_symbol_map['node_kind'] == 'PROTEIN']
    ambig = (
        prot.groupby('uniprot')['func_name']
        .apply(lambda s: sorted(s.unique().tolist()))
        .reset_index(name='func_names')
    )
    ambig = ambig[ambig['func_names'].str.len() > 1]
    if len(ambig):
        ambig['func_names'] = ambig['func_names'].apply(';'.join)
        ambig.to_csv(os.path.join(out_dir, 'dti_ambiguous_uniprot.csv'), index=False)
    return len(ambig)


def load_perturbation_edges(args, uniprot_symbol_map: pd.DataFrame, func_nodes, out_dir: str):
    from lincs_gsnn.proc.perturbations import resolve_perturbation_edges

    _log("Resolving perturbation input edges (chemical + genetic)...")

    pert_path = f'{args.meta}/pert_ids.csv'
    _log(f"  reading {pert_path}")
    pert_names = pd.read_csv(pert_path)['pert_id'].astype(str).tolist()
    n_chem = sum(1 for p in pert_names if not any(p.startswith(x) for x in ('xpr_', 'oe_', 'sh_')))
    n_gen = len(pert_names) - n_chem
    _log(f"  pert_ids.csv: {len(pert_names)} tokens ({n_chem} chemical, {n_gen} genetic)")

    cond_info_path = args.cond_info_path
    if cond_info_path is None and args.gene_stats_path:
        cond_info_path = os.path.join(os.path.dirname(args.gene_stats_path), 'cond_info.csv')
    if not cond_info_path or not os.path.exists(cond_info_path):
        raise FileNotFoundError(
            "cond_info.csv is required for pert_name -> BRD mapping and genetic targets. "
            "Pass --cond_info_path or --gene_stats_path pointing to the proc directory."
        )
    _log(f"  reading cond_info: {cond_info_path}")
    cond_info = pd.read_csv(cond_info_path)
    _log(f"  cond_info: {len(cond_info)} rows, {cond_info['pert_name'].nunique()} unique pert_name")

    compound_path = f'{args.data}/compoundinfo_beta.txt'
    _log(f"  reading compoundinfo (may take a moment): {compound_path}")
    compoundinfo = pd.read_csv(compound_path, sep='\t')
    _log(f"  compoundinfo: {len(compoundinfo)} rows")

    targetome_path = f'{args.data}/targetome_extended-01-23-25.csv'
    _log(f"  reading targetome (may take a moment): {targetome_path}")
    targetome = pd.read_csv(targetome_path)
    _log(f"  targetome: {len(targetome)} rows")

    u2fn = (
        uniprot_symbol_map.loc[
            uniprot_symbol_map['node_kind'] == 'PROTEIN',
            ['uniprot', 'func_name', 'gene_symbol', 'node_kind'],
        ]
        .drop_duplicates(subset=['uniprot', 'func_name'])
    )
    map_path = os.path.join(out_dir, 'dti_uniprot_protein_map.csv')
    u2fn[['uniprot', 'func_name', 'gene_symbol']].to_csv(map_path, index=False)
    _log(f"  wrote UniProt->PROTEIN map ({len(u2fn)} rows) -> {map_path}")

    func_names = set(func_nodes['func_name'].astype(str).tolist())
    _log(
        f"  resolving edges (max_dti_kd={args.max_dti_kd}, "
        f"{len(func_names)} function nodes in graph)..."
    )
    t0 = time.perf_counter()
    edges, dropped = resolve_perturbation_edges(
        pert_names=pert_names,
        cond_info=cond_info,
        compoundinfo=compoundinfo,
        targetome=targetome,
        uniprot_to_func=u2fn,
        func_names=func_names,
        max_kd=args.max_dti_kd,
    )
    _log(
        f"  resolve_perturbation_edges done in {time.perf_counter() - t0:.1f}s: "
        f"{len(edges)} edges, {edges['src'].nunique() if len(edges) else 0} perts kept, "
        f"{len(dropped)} dropped"
    )

    dropped_path = os.path.join(out_dir, 'dropped_perts.csv')
    dropped.to_csv(dropped_path, index=False)
    _log(f"  wrote dropped perts audit -> {dropped_path}")
    print(
        f'>perturbation edges: {len(edges)} from {edges["src"].nunique()} perts '
        f'({n_chem} chemical, {n_gen} genetic tokens in pert_ids.csv; '
        f'{len(dropped)} dropped with no resolvable target)',
        flush=True,
    )
    return edges


def load_gene_inputs(args, func_nodes, canonical): 
    gene_input_edges, _ = build_lincs_gene_edges(canonical, func_nodes, edge_kind='input')
    gene_names = [f'GENE__{s}' for s in canonical['lincs_symbol'].astype(str)]
    return gene_input_edges, gene_names


def load_gene_outputs(args, func_nodes, canonical): 
    gene_output_edges, resolution = build_lincs_gene_edges(canonical, func_nodes, edge_kind='output')
    gene_names = [f'GENE__{s}' for s in canonical['lincs_symbol'].astype(str)]
    return gene_output_edges, gene_names, resolution


def load_cell_inputs(args, func_nodes): 
    ''' add an edge from every cell to every function node'''
    lines = pd.read_csv(f'{args.meta}/cell_inames.csv').cell_iname.values.tolist() 
    func_candidates = func_nodes['func_name'].tolist()
    n_edges = len(lines) * len(func_candidates)
    _log(
        f"Building LINE->function mediator edges: {len(lines)} cell lines x "
        f"{len(func_candidates)} function nodes = {n_edges:,} edges"
    )
    df = {'src': [], 'dst': []}
    for line in lines: 
        for f in func_candidates: 
            df['src'].append('LINE__' + line)
            df['dst'].append(f)
    df = pd.DataFrame(df)
    _log(f"  LINE->function edges built ({len(df):,} rows)")
    return df 




def load_data(args): 
    _log('Loading biological interaction graph (get_bio_interactions)...')
    t0 = time.perf_counter()

    func_nodes, func_edges, uniprot_symbol_map = get_bio_interactions(
        include_tf_mirna=args.include_tf_mirna,
        include_pathway_extra=args.include_pathway_extra,
        include_kinase_extra=args.include_kinase_extra,
        include_ligrec_extra=args.include_ligrec_extra,
        include_collecTRI=args.include_collecTRI,
        include_dorothea=args.include_dorothea,
        include_omnipath=args.include_omnipath,
        dorothea_levels=list(args.dorothea_levels),
        min_n_references=args.min_n_references,
        min_curation_effort=args.min_curation_effort,
        complex_handling=args.complex_handling,
        gene_symbol=True,
        verbose=True,
        return_uniprot_map=True,
    )
    _log(
        f"get_bio_interactions done in {time.perf_counter() - t0:.1f}s: "
        f"{len(func_nodes)} function nodes, {len(func_edges)} function edges"
    )

    os.makedirs(args.out, exist_ok=True)
    n_ambig = _write_dti_ambiguous_uniprot(uniprot_symbol_map, args.out)
    if n_ambig:
        print(
            f'DTI: {n_ambig} ambiguous UniProt accessions map to multiple '
            f'PROTEIN__ nodes (see dti_ambiguous_uniprot.csv; edges kept for all targets)',
            flush=True,
        )

    dti_input_edges = load_perturbation_edges(args, uniprot_symbol_map, func_nodes, args.out)

    lincs_symbols = pd.read_csv(f'{args.meta}/gene_names.csv')['gene_names'].astype(str).tolist()
    _log(
        f"Building canonical landmark gene table ({len(lincs_symbols)} genes; "
        f"fetch_uniprot_synonyms={not args.no_uniprot_synonyms})..."
    )
    if not args.no_uniprot_synonyms:
        _log("  UniProt synonym fetch may take several minutes (network API)...")
    t0 = time.perf_counter()
    canonical = build_canonical_gene_table(
        lincs_symbols,
        args.geneinfo_path,
        verbose=True,
        fetch_uniprot_synonyms=not args.no_uniprot_synonyms,
        func_nodes=func_nodes,
    )
    _log(f"canonical_genes done in {time.perf_counter() - t0:.1f}s")
    save_canonical_gene_artifact(os.path.join(args.out, 'canonical_genes.csv'), canonical)
    _log(f"  saved canonical_genes.csv ({len(canonical)} rows)")

    _log("Building L1000 gene output edges...")
    gene_output_edges, gene_names, lincs_resolution = load_gene_outputs(args, func_nodes, canonical)
    lincs_resolution.to_csv(os.path.join(args.out, 'lincs_fn_resolution.csv'), index=False)
    method_counts = lincs_resolution['method'].value_counts()
    print('L1000 function-node resolution:', flush=True)
    for method, count in method_counts.items():
        print(f'  {method}: {count}', flush=True)
    n_in_graph = int(lincs_resolution['in_function_graph'].sum())
    print(f'  in_function_graph: {n_in_graph} / {len(lincs_resolution)}', flush=True)

    _log("Building gene input edges...")
    gene_input_edges, _ = load_gene_inputs(args, func_nodes, canonical)
    cell_input_edges = load_cell_inputs(args, func_nodes)
    _log(
        f"load_data complete: dti={len(dti_input_edges)}, gene_in={len(gene_input_edges)}, "
        f"gene_out={len(gene_output_edges)}, func={len(func_edges)}, "
        f"cell_mediator={len(cell_input_edges)}"
    )

    return dti_input_edges, gene_input_edges, gene_output_edges, func_edges, cell_input_edges, gene_names


if __name__ == '__main__': 

    args = get_args() 
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)
    print() 
    print() 

    os.makedirs(args.out, exist_ok=True)

    dti_input_edges, gene_input_edges, gene_output_edges, func_edges, cell_input_edges, gene_names = load_data(args) 

    if args.include_gene_inputs:
        input_edges = pd.concat([dti_input_edges, gene_input_edges], axis=0)
    else:
        print('--include_gene_inputs NOT set: dropping GENE__ -> RNA__ input edges; '
              'GENE__ names still force-included as inputs. Drugs are the sole pruning roots.')
        input_edges = dti_input_edges
    output_edges = gene_output_edges
    mediator_edges = cell_input_edges

    func_edge_table = func_edges[['src', 'dst']].copy()

    # Optionally drop LINE->function edges. We still want LINE__ names to appear in
    # data.node_names_dict['input'] so DXDTDataset's `input_names.index("LINE__"+cell)`
    # lookup keeps working; we force-include them via the constructor's input_names arg.
    if args.no_cell_line_edges:
        print('--no_cell_line_edges set: dropping LINE->function edges; force-including LINE__ names as inputs')
        line_names = cell_input_edges['src'].drop_duplicates().tolist()
        force_input_names = list(gene_names) + line_names
        mediator_edges = None
    else:
        force_input_names = gene_names

    print('network pre-processing and construction...', flush=True)
    _log(
        f"GSNNNetworkConstructor.build (filter_depth={args.filter_depth}): "
        f"input={len(input_edges)}, output={len(output_edges)}, "
        f"function={len(func_edge_table)}, mediator={0 if mediator_edges is None else len(mediator_edges)}"
    )
    t0 = time.perf_counter()
    constructor = GSNNNetworkConstructor(depth=args.filter_depth, 
                                         verbose=True)

    data = constructor.build(input_edges=input_edges, 
                             output_edges=output_edges, 
                             function_edges=func_edge_table, 
                             mediator_edges=mediator_edges, 
                             input_names=force_input_names,     # force all gene inputs (and LINE__ names when --no_cell_line_edges) to be included in graph even if there are no edges 
                             output_names=gene_names)           # force all gene outputs to be included in graph even if there are no edges 
    _log(f"constructor.build done in {time.perf_counter() - t0:.1f}s")


    # we want to remove function -> output edges, however, output edges go from RNA__NODE -> GENE__NODE 
    # so we need to remove edges from FUNCTION__NODE -> RNA__NODE --> GENE__NODE 

    removed_edges_records = []

    if args.remove_output_edges:

        print('-'*40)
        print('Removing output edges...')
        print('-'*40)
        print()

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_edges_before = data.edge_index_dict['function', 'to', 'function'].shape[1]
        fn_node_labels = np.array(data.node_names_dict['function'])
        for output_node in data.node_names_dict['output']:

            # get RNA__NODE
            rna_node = 'RNA__' + output_node.split('__')[1]
            if rna_node not in data.node_names_dict['function']:
                print(f'WARNING: {output_node} has no RNA__NODE, skipping...')
                continue
            rna_idx = data.node_names_dict['function'].index(rna_node)

            # candidate edges (to remove)
            src, dst = data.edge_index_dict['function', 'to', 'function']
            edge_candidate_ixs = (dst == rna_idx).nonzero(as_tuple=True)[0].view(-1).detach().cpu().numpy()

            if len(edge_candidate_ixs) <= (args.n_edges_to_remove):
                print(f'WARNING: {output_node} has only {len(edge_candidate_ixs)} edges [n_edges_to_remove={args.n_edges_to_remove}], skipping...')
                continue

            ixs_to_remove = np.random.choice(edge_candidate_ixs, size=args.n_edges_to_remove, replace=False)

            # remove edges
            mask_to_keep = np.ones(src.shape[0], dtype=bool)
            mask_to_keep[ixs_to_remove] = False
            data.edge_index_dict['function', 'to', 'function'] = data.edge_index_dict['function', 'to', 'function'][:, mask_to_keep]

            src_ixs = src[ixs_to_remove].tolist()
            dst_ixs = dst[ixs_to_remove].tolist()
            for s_ix, d_ix in zip(src_ixs, dst_ixs):
                removed_edges_records.append({
                    'src_idx': s_ix,
                    'dst_idx': d_ix,
                    'src_name': fn_node_labels[s_ix],
                    'dst_name': fn_node_labels[d_ix],
                    'split': 'test',
                })

        n_edges_after = data.edge_index_dict['function', 'to', 'function'].shape[1]
        n_removed = n_edges_before - n_edges_after
        assert n_removed == len(removed_edges_records), (
            'number of removed output edges does not match recorded number of removed edges'
        )
        print(f'removed {n_removed} output edges, {n_edges_after} edges remaining')
        print('-'*40)
        print('-'*40)

    n_val = int(args.holdout_N_val_edges)
    n_test = int(args.holdout_N_test_edges)
    if n_val > 0 or n_test > 0:
        print('-'*40)
        print(f'Removing {n_val} val + {n_test} test function->function edges (global random)...')
        print('-'*40)
        print()

        # Separate RNG branch so --remove_output_edges behaviour is unchanged for a given seed.
        rng = np.random.RandomState(int(args.seed) + 10007)
        n_holdout = n_val + n_test
        src, dst = data.edge_index_dict['function', 'to', 'function']
        n_edges = int(src.shape[0])
        if n_holdout > n_edges:
            raise ValueError(
                f'holdout_N_val_edges + holdout_N_test_edges ({n_holdout}) exceeds '
                f'available function->function edges ({n_edges})'
            )

        ixs_to_remove = rng.choice(n_edges, size=n_holdout, replace=False)
        val_ixs = set(ixs_to_remove[:n_val].tolist())
        test_ixs = set(ixs_to_remove[n_val:].tolist())

        mask_to_keep = np.ones(n_edges, dtype=bool)
        mask_to_keep[list(ixs_to_remove)] = False
        data.edge_index_dict['function', 'to', 'function'] = (
            data.edge_index_dict['function', 'to', 'function'][:, mask_to_keep]
        )

        fn_node_labels = np.array(data.node_names_dict['function'])
        for edge_ix in ixs_to_remove.tolist():
            s_ix = int(src[edge_ix].item())
            d_ix = int(dst[edge_ix].item())
            split = 'val' if edge_ix in val_ixs else 'test'
            removed_edges_records.append({
                'src_idx': s_ix,
                'dst_idx': d_ix,
                'src_name': fn_node_labels[s_ix],
                'dst_name': fn_node_labels[d_ix],
                'split': split,
            })

        n_edges_after = data.edge_index_dict['function', 'to', 'function'].shape[1]
        print(f'removed {n_holdout} holdout edges ({n_val} val, {n_test} test), '
              f'{n_edges_after} edges remaining')
        print('-'*40)
        print('-'*40)

    if removed_edges_records:
        removed_edges = pd.DataFrame(removed_edges_records)
    else:
        removed_edges = pd.DataFrame(columns=['src_idx', 'dst_idx', 'src_name', 'dst_name', 'split'])

    if args.simplify_degree_one:
        print('-' * 40)
        print('Simplifying function graph (degree-1 contraction)...')
        print('-' * 40)
        simplify_function_layer(
            data,
            simplify_degree_one=True,
            degree_one_in_degree=args.simplify_degree_one_in_degree,
            degree_one_out_degree=args.simplify_degree_one_out_degree,
            check_reachability=args.simplify_check_reachability,
        )
        node_map = getattr(data, 'function_node_map', None)
        if node_map is not None and len(removed_edges):
            removed_edges = remap_removed_edges_names(removed_edges, node_map)
        print('-' * 40)

    # Always write removed_edges.csv so downstream MEI eval rules can depend on it.
    removed_edges.to_csv(f'{args.out}/removed_edges.csv', index=False)
    if len(removed_edges):
        print(f'Wrote {len(removed_edges)} removed edges to {args.out}/removed_edges.csv')

    _log('Computing drug-accessible output genes...')
    data.drug_accessible_output_genes = compute_drug_accessible_output_genes(data)
    n_acc = int(data.drug_accessible_output_genes.sum())
    n_total = len(data.node_names_dict['output'])
    print(f'Drug-accessible output genes: {n_acc} / {n_total}', flush=True)

    _log(f'Saving bionetwork.pt -> {args.out}/bionetwork.pt')
    torch.save(data, f'{args.out}/bionetwork.pt')
    _log('bionetwork.pt saved')

    # ------------------------------------------------------------------
    # Cell-drug train/val split: per drug, hold out val_cells_per_drug cell
    # lines (default 1 of 11) for validation; all dose/time rows inherit the
    # pair-level partition via pretrain/train scripts.
    # ------------------------------------------------------------------
    from lincs_gsnn.proc.cell_drug_split import (
        build_cell_drug_split,
        save_cell_drug_split,
        summarize_split,
    )

    _log('Building cell-drug train/val split...')
    print('-'*40, flush=True)
    print('Building cell-drug train/val split...', flush=True)
    print('-'*40, flush=True)
    pert_ids = pd.read_csv(f'{args.meta}/pert_ids.csv')['pert_id'].astype(str).tolist()
    cell_inames = pd.read_csv(f'{args.meta}/cell_inames.csv')['cell_iname'].astype(str).tolist()
    split_df = build_cell_drug_split(
        pert_ids=pert_ids,
        cell_inames=cell_inames,
        n_val=args.val_cells_per_drug,
        seed=args.seed,
    )
    split_path = f'{args.out}/cell_drug_split.csv'
    save_cell_drug_split(split_path, split_df)
    summary = summarize_split(split_df)
    print(
        f'saved cell_drug_split to {split_path} '
        f'(n_pairs={summary["n_pairs"]}, n_drugs={summary["n_drugs"]}, '
        f'n_cells={summary["n_cells"]}, train={summary["n_train_pairs"]}, '
        f'val={summary["n_val_pairs"]}, val_fraction={summary["val_fraction"]:.4f})'
    )
    print() 

    # ------------------------------------------------------------------
    # gene_norm artifact: per-gene control-population mu/sigma from
    # lincs-traj's gene_stats.dict, aligned to predict_grid/
    # gene_names.csv. BIOGSNN requires this artifact for its log1p-back
    # degradation term.
    # ------------------------------------------------------------------
    if args.gene_stats_path is not None:
        from lincs_gsnn.proc.gene_norm import (
            build_gene_norm_artifact,
            save_gene_norm_artifact,
        )

        _log(f'Building gene_norm artifact from {args.gene_stats_path}...')
        print('-'*40, flush=True)
        print('Building gene_norm artifact...', flush=True)
        print('-'*40, flush=True)

        gn_payload = build_gene_norm_artifact(
            gene_stats_path=args.gene_stats_path,
            gene_names_csv_path=f'{args.meta}/gene_names.csv',
            output_names=data.node_names_dict['output'],
        )
        gn_path = f'{args.out}/gene_norm.pt'
        save_gene_norm_artifact(gn_path, gn_payload)
        print(
            f'saved gene_norm artifact to {gn_path} '
            f'(n_genes={len(gn_payload["gene_names"])}, '
            f'source={gn_payload["source"]})'
        )
        print()
    else:
        print('WARNING: --gene_stats_path not provided; gene_norm.pt will not be built. '
              'BIOGSNN training/explain will fail until this artifact exists.')
        print()

    # ------------------------------------------------------------------
    # Optional: build the node-activity (x_fn) artifact.
    # Gated on --node_activity so existing runs are byte-identical.
    # ------------------------------------------------------------------
    if args.node_activity:
        from lincs_gsnn.proc.node_activity import (
            build_x_fn_lookup_from_bionet,
            save_node_activity_artifact,
        )

        _log('Building node_activity (x_fn) artifact...')
        print('-'*40, flush=True)
        print('Building node_activity (x_fn) artifact...', flush=True)
        if 'expr' in args.node_activity_features:
            print('  This reads OmicsExpressionTPMLogp1HumanAllGenes.csv and may take ~2.5 minutes.', flush=True)
        print('-'*40, flush=True)

        x_fn_by_ciname, metadata = build_x_fn_lookup_from_bionet(
            node_names_dict=data.node_names_dict,
            data_root=args.data,
            cell_inames=None,           # default: all LINE__* in the bionet
            eps=args.node_activity_eps,
            features=args.node_activity_features,
        )

        artifact_path = f'{args.out}/node_activity.pt'
        save_node_activity_artifact(artifact_path, x_fn_by_ciname, metadata)
        print(
            f'saved node_activity artifact to {artifact_path} '
            f'(n_cells={len(metadata["cell_iname_order"])}, '
            f'n_function_nodes={len(metadata["function_genes"])}, '
            f'activity_dim={metadata["activity_dim"]}, '
            f'activity_features={metadata.get("activity_features", ["expr"])}, '
            f'dropped={len(metadata["dropped_cell_inames"])} unresolved cell_iname(s), '
            f'{len(metadata.get("dropped_cell_inames_no_expression", []))} '
            f'with missing feature row(s))'
        )
        print() 
