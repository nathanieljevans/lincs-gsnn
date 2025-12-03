'''
Constructs a heterogeneous biological network graph for LINCS-GSNN project. 

This script integrates multiple biological data sources to create a comprehensive network 
representation that captures drug-target interactions, biological pathway relationships, 
and gene regulatory networks. The resulting graph serves as the foundation for a 
graph structured neural network that predicts drug-induced gene expression changes.

Data Sources:
- Drug-target interactions (DTI): Filtered from Targetome Extended database using 
  affinity thresholds (Kd/Ki ≤ 1000 nM) and direct binding assays
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
import pandas as pd 
import torch_geometric as pyg
import numpy as np
import torch

from gsnn.proc.bio import get_bio_interactions 
from gsnn.proc.bio import uniprot2symbol
from gsnn.proc.construct import GSNNNetworkConstructor



def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                   help="path to data directory")
    parser.add_argument("--meta",               type=str,               default='../../../predict_grid/',                   help="path to metadata directory")
    parser.add_argument("--extdata",            type=str,               default='../../../extdata/',                help="path to extdata directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                help="path to data directory")
    parser.add_argument("--include_mirna",      action='store_true',    default=False,                              help="include miRNA interactions (TF->miRNA, miRNA->mRNA) in the function graph")
    parser.add_argument("--include_extra",      action='store_true',    default=False,                              help="include extra interactions (pathways, kinases) in the function graph; these are interactions from omnipath that don't have literature references")
    parser.add_argument('--dorothea_levels',    type=str,               default='ABCD',                               help='the dorothea levels to include in the function graph [A-D]')
    parser.add_argument("--max_dti_kd",         type=float,             default=1000.0,                             help="maximum DTI affinity (Kd) to include in the graph")
    parser.add_argument("--filter_depth",       type=int,               default=4,                                 help="the depth to search for upstream drugs and downstream lincs in the node filter process")
    parser.add_argument("--remove_output_edges", action='store_true',   default=False,                              help="remove output edges from the function graph")
    parser.add_argument("--n_edges_to_remove",  type=int,               default=1,                               help="the number of edges to remove from the function graph")
    parser.add_argument("--seed",               type=int,               default=42,                                help="the random seed to use for the edge removal process")
    args = parser.parse_args() 
    return args

def load_targetome(args): 

    drugs = pd.read_csv(f'{args.meta}/pert_ids.csv').pert_id.values.tolist() 
    clue_mapping = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t')[['inchi_key', 'pert_id']].drop_duplicates() 
    tge = pd.read_csv(f'{args.data}/targetome_extended-01-23-25.csv').merge(clue_mapping, on='inchi_key', how='inner')
    tge = tge[lambda x: x.pert_id.isin(drugs)]

    # map tge uniprot_id to gene symbols (feature name: target) 
    u2s = uniprot2symbol(tge.uniprot_id.values.tolist(), allow='1:m', drop_na=True)
    tge = tge.merge(u2s, on='uniprot_id', how='inner')

    tge = tge[lambda x: x.assay_type.isin(['Kd', 'Ki'])] # direct targets only 
    tge = tge[lambda x: x.assay_relation.isin(['=', '<', '<='])] # exclude ">" relations 
    tge = tge[lambda x: x.assay_value <= args.max_dti_kd] # only targets with affinity <= 1000 nM

    tge = tge[['pert_id', 'gene_symbol']].drop_duplicates()
    tge = tge.assign(dst = ['PROTEIN__' + x for x in tge.gene_symbol.values])
    tge = tge.assign(src = ['DRUG__' + x for x in tge.pert_id.values]) 

    tge = tge[['src', 'dst']].drop_duplicates() 

    return tge


def load_gene_inputs(args): 
    gene_names = pd.read_csv(f'{args.meta}/gene_names.csv')['gene_names'].values.astype(str).tolist() 
    df = pd.DataFrame({'src': ['GENE__' + x for x in gene_names], 'dst': ['RNA__' + x for x in gene_names]})
    return df, ['GENE__' + x for x in gene_names]

def load_gene_outputs(args): 
    gene_names = pd.read_csv(f'{args.meta}/gene_names.csv')['gene_names'].values.astype(str).tolist() 
    df = pd.DataFrame({'src': ['RNA__' + x for x in gene_names], 'dst': ['GENE__' + x for x in gene_names]})
    return df, ['GENE__' + x for x in gene_names]

def load_cell_inputs(args, func_candidates): 
    ''' add an edge from every cell to every function node'''
    lines = pd.read_csv(f'{args.meta}/cell_inames.csv').cell_iname.values.tolist() 
    df = {'src': [], 'dst': []}
    for line in lines: 
        for f in func_candidates: 
            df['src'].append('LINE__' + line)
            df['dst'].append(f)
    df = pd.DataFrame(df)
    return df 



def load_data(args): 
    print('Loading data...')

    dti_input_edges = load_targetome(args) 

    func_names, func_edges = get_bio_interactions(include_extra=args.include_extra, 
                                                include_mirna=args.include_mirna, 
                                                dorothea_levels=list(args.dorothea_levels),
                                                gene_symbol=True)
    
    gene_input_edges, gene_names = load_gene_inputs(args) 
    gene_output_edges, _ = load_gene_outputs(args) 
    cell_input_edges = load_cell_inputs(args, func_names) 

    return dti_input_edges, gene_input_edges, gene_output_edges, func_edges, cell_input_edges, gene_names


if __name__ == '__main__': 

    args = get_args() 
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)
    print() 
    print() 

    dti_input_edges, gene_input_edges, gene_output_edges, func_edges, cell_input_edges, gene_names = load_data(args) 

    input_edges = pd.concat([dti_input_edges, gene_input_edges], axis=0) 
    output_edges = gene_output_edges  
    mediator_edges = cell_input_edges

    print('network pre-processing and construction...')
    constructor = GSNNNetworkConstructor(depth=args.filter_depth, 
                                         verbose=True)

    data = constructor.build(input_edges=input_edges, 
                             output_edges=output_edges, 
                             function_edges=func_edges, 
                             mediator_edges=mediator_edges, 
                             input_names=gene_names,            # force all gene inputs to be included in graph even if there are no edges 
                             output_names=gene_names)           # force all gene outputs to be included in graph even if there are no edges 


    # we want to remove function -> output edges, however, output edges go from RNA__NODE -> GENE__NODE 
    # so we need to remove edges from FUNCTION__NODE -> RNA__NODE --> GENE__NODE 

    if args.remove_output_edges:  

        print('-'*40)
        print('Removing output edges...')
        print('-'*40)
        print() 

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        n_edges_before = data.edge_index_dict['function', 'to', 'function'].shape[1]
        removed_edges = {'src_idx': [], 'dst_idx': [], 'src_name': [], 'dst_name': []}
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

            if len(edge_candidate_ixs)  <= (args.n_edges_to_remove):
                print(f'WARNING: {output_node} has only {len(edge_candidate_ixs)} edges [n_edges_to_remove={args.n_edges_to_remove}], skipping...')
                continue 

            ixs_to_remove = np.random.choice(edge_candidate_ixs, size=args.n_edges_to_remove, replace=False) 

            # remove edges 
            mask_to_keep = np.ones(src.shape[0], dtype=bool)
            mask_to_keep[ixs_to_remove] = False 
            data.edge_index_dict['function', 'to', 'function'] = data.edge_index_dict['function', 'to', 'function'][:, mask_to_keep]

            src_ixs = src[ixs_to_remove].tolist() 
            dst_ixs = dst[ixs_to_remove].tolist() 
            removed_edges['src_idx'] += src_ixs
            removed_edges['dst_idx'] += dst_ixs
            func_names = np.array(data.node_names_dict['function'])
            removed_edges['src_name'] += func_names[src_ixs].tolist()
            removed_edges['dst_name'] += func_names[dst_ixs].tolist()

        removed_edges = pd.DataFrame(removed_edges)
        removed_edges.to_csv(f'{args.out}/removed_edges.csv', index=False)

        n_edges_after = data.edge_index_dict['function', 'to', 'function'].shape[1] 

        n_removed = n_edges_before - n_edges_after 

        assert n_removed == len(removed_edges), 'number of removed edges does not match recorded number of removed edges'

        print(f'removed {n_removed} edges, {n_edges_after} edges remaining')
        
        print('-'*40)
        print('-'*40)


    print('saving data...')
    torch.save(data, f'{args.out}/bionetwork.pt')
    print() 
    print() 


