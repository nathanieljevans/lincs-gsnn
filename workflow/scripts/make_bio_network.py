import argparse 
import pandas as pd 
from lincs_gsnn.proc.get_bio_interactions import get_bio_interactions 
from lincs_gsnn.proc.subset import filter_func_nodes
import torch_geometric as pyg
import numpy as np
import torch




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

    args = parser.parse_args() 
    return args


def load_data(args): 
    print('Loading data...')
    
    gene_names = pd.read_csv(f'{args.meta}//gene_names.csv')['gene_names'].values.astype(str).tolist() 
    drugs = pd.read_csv(f'{args.meta}/pert_ids.csv').pert_id.values.tolist() 
    lines = pd.read_csv(f'{args.meta}/cell_inames.csv').cell_iname.values.tolist()

    clue_mapping = pd.read_csv(f'{args.data}/compoundinfo_beta.txt', sep='\t')[['inchi_key', 'pert_id']].drop_duplicates() 
    tge = pd.read_csv(f'{args.data}/targetome_extended-01-23-25.csv').merge(clue_mapping, on='inchi_key', how='inner')

    landmark_mapping = pd.read_csv(f'{args.extdata}/landmark_gene2uni.tsv', sep='\t')
    landmark_mapping = landmark_mapping.rename({'From': 'gene_symbol', 'Entry': 'uniprot_id'}, axis=1)[['gene_symbol', 'uniprot_id']]

    func_names, func_df = get_bio_interactions(include_extra=args.include_extra, include_mirna=args.include_mirna, dorothea_levels=list(args.dorothea_levels),)

    return gene_names, drugs, lines, tge, landmark_mapping, func_names, func_df


def filter_dtis(args, dtis, protein_candidates, drugs): 
    print('Filtering DTI data...')

    dtis = dtis[lambda x: x.assay_type.isin(['Kd', 'Ki'])] # direct targets only 
    dtis = dtis[lambda x: x.assay_relation.isin(['=', '<', '<='])] # exclude ">" relations 
    dtis = dtis[lambda x: x.assay_value <= args.max_dti_kd] # only targets with affinity <= 1000 nM 
    dtis = dtis[lambda x: x.pert_id.isin(drugs)] # only drugs in our list
    dtis = dtis[lambda x: x.uniprot_id.isin(protein_candidates)]

    dtis = dtis[['pert_id', 'uniprot_id']].drop_duplicates().rename({'uniprot_id': 'target'}, axis=1)
    dtis = dtis.assign(target_name = ['PROTEIN__' + x for x in dtis.target.values]) # add prefix to match func_names

    return dtis

def get_unique_landmark_mapping(landmark_mapping, rna_candidates): 
    print('Getting unique landmark mapping...')

    # are there any duplicate gene->uni mappings?
    #print(landmark_mapping.duplicated(subset=['gene_symbol']).sum())

    landmark_mapping = landmark_mapping[lambda x: x.uniprot_id.isin(rna_candidates)] 
    landmark_mapping = landmark_mapping.drop_duplicates(subset=['gene_symbol'])
    landmark_mapping = landmark_mapping.set_index('gene_symbol')['uniprot_id'].to_dict() 

    return landmark_mapping

def check_all_drugs_have_targets(dtis, drugs): 
    print('Checking if all drugs have targets in the targetome...')

    cnts = dtis.groupby('pert_id').size().sort_values(ascending=False) 
    missing_drugs = [x for x in drugs if x not in cnts.index] 
    if len(missing_drugs) > 0: 
        print(f'WARNING: {len(missing_drugs)} drugs have no targets in the targetome: {missing_drugs}')

def make_input_edge_index(targets, function_names, landmark_map_dict, gene_names, input_name2idx, function_name2idx, lines):
    print('Making input edge index...') 

    input_edge_list = [] 

    # add drug edges 
    for i, dti_row in targets.iterrows(): 
        if dti_row.target_name in function_names: 
            input_edge_list.append(['DRUG__' + dti_row.pert_id, dti_row.target_name]) 
        else:
            print(f'No function node for target {dti_row.target} in drug {dti_row.pert_id}')

    # add cell line edges (one edge to every function node)
    for line in lines: 
        for f in function_names: 
            input_edge_list.append(['LINE__' + line, f])

    for g in gene_names: 
        targ = landmark_map_dict.get(g, None) 

        if (targ is not None) and ('RNA__' + targ in function_names): 
            input_edge_list.append(['GENE__' + g, 'RNA__' + targ]) # gene to uniprot target edge
        else: 
            print(f'No uniprot mapping for gene {g}') 

    row,col = np.array(input_edge_list).T
    row = [input_name2idx[r] for r in row]
    col = [function_name2idx[c] for c in col] 

    input_edge_index = torch.tensor([row, col], dtype=torch.long) 

    return input_edge_index


def make_function_edge_index(func_df): 
    print('Making function edge index...')

    func_edge_index = torch.stack([torch.tensor(func_df.src_idx.values, dtype=torch.long),
                                   torch.tensor(func_df.dst_idx.values, dtype=torch.long)], dim=0) 
    
    return func_edge_index

def make_output_edge_index(gene_names, landmark_map_dict, function_names, function_name2idx, output_name2idx): 
    print('Making output edge index...')

    output_edge_list = [] 

    for g in gene_names:
        targ = landmark_map_dict.get(g, None) 

        if (targ is not None) and ('RNA__' + targ in function_names): 
            output_edge_list.append(['RNA__' + targ, 'GENE__' + g]) # gene to uniprot target edge
        else: 
            print(f'No uniprot mapping for gene {g}') 
            # self loop as placeholder?

    row,col = np.array(output_edge_list).T
    row = [function_name2idx[r] for r in row]
    col = [output_name2idx[c] for c in col]
    output_edge_index = torch.tensor([row, col], dtype=torch.long)

    return output_edge_index

def checks(data, *, require_drug_targets: bool = True) -> None:
    """
    Sanity‑check a `pyg.data.HeteroData` graph produced by the LINCS‑GSNN
    build script.  The routine raises **AssertionError** with descriptive
    messages whenever a structural inconsistency is found.

    Parameters
    ----------
    data : pyg.data.HeteroData
        Must contain:
            • data['node_names_dict'] — {'input':  [...],
                                         'function':[...],
                                         'output': [...]}  
              (every list is unique and ordered)
            • data['edge_index_dict']
              { ('input','to','function')  : LongTensor[[row],[col]],
                ('function','to','function'): …,
                ('function','to','output')  : … }
    require_drug_targets : bool, default=True
        If **True**, assert that every DRUG__ node has ≥ 1 outgoing edge
        to a function node.
    """
    import torch

    # ------------------------------------------------------------------
    # 1.  Node dictionaries are present and consistent
    # ------------------------------------------------------------------
    node_names = data.get('node_names_dict')
    assert isinstance(node_names, dict), "`node_names_dict` missing or not a dict"

    for key in ('input', 'function', 'output'):
        assert key in node_names, f"`node_names_dict` missing '{key}' list"
        names = node_names[key]
        assert len(names) == len(set(names)), f"duplicate node names in '{key}' list"

    inp_names, fun_names, out_names = (
        node_names['input'], node_names['function'], node_names['output']
    )
    n_inp, n_fun, n_out = map(len, (inp_names, fun_names, out_names))

    # ------------------------------------------------------------------
    # 2.  Edge dictionaries are present and shaped correctly
    # ------------------------------------------------------------------
    edge_dict = data.get('edge_index_dict')
    assert isinstance(edge_dict, dict), "`edge_index_dict` missing or not a dict"

    expected_keys = [
        ('input',    'to', 'function'),
        ('function', 'to', 'function'),
        ('function', 'to', 'output'),
    ]
    missing = [k for k in expected_keys if k not in edge_dict]
    assert not missing, f"`edge_index_dict` missing keys: {missing}"

    def _range_check(t: torch.Tensor, lo: int, hi: int, name: str):
        assert t.dtype == torch.long and t.dim() == 2 and t.size(0) == 2, \
            f"{name} must be LongTensor shape (2, E)"
        src, dst = t
        assert (0 <= src.min() < hi) and (0 <= dst.min() < hi), \
            f"{name} contains negative indices"
        assert src.max() < lo, f"{name} src index overflow ({src.max()} ≥ {lo})"
        assert dst.max() < hi, f"{name} dst index overflow ({dst.max()} ≥ {hi})"

    # a) input  → function
    t_if = edge_dict[('input', 'to', 'function')].cpu()
    _range_check(t_if, n_inp, n_fun, "edge[input→function]")

    # b) function → function
    t_ff = edge_dict[('function', 'to', 'function')].cpu()
    _range_check(t_ff, n_fun, n_fun, "edge[function→function]")

    # c) function → output
    t_fo = edge_dict[('function', 'to', 'output')].cpu()
    _range_check(t_fo, n_fun, n_out, "edge[function→output]")

    # ------------------------------------------------------------------
    # 3.  Optional: each DRUG__ node has at least one outgoing edge
    # ------------------------------------------------------------------
    if require_drug_targets:
        drug_prefix = "DRUG__"
        drug_indices = [
            i for i, name in enumerate(inp_names) if name.startswith(drug_prefix)
        ]
        src_if = t_if[0]
        for idx in drug_indices:
            assert (src_if == idx).any(), \
                f"{inp_names[idx]} has no outgoing edge to a function node"

    # ------------------------------------------------------------------
    # 4.  Everything passed
    # ------------------------------------------------------------------
    return  # (explicit for clarity)



if __name__ == '__main__': 

    args = get_args() 
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)

    gene_names, drugs, lines, dtis, landmark_mapping, func_names, func_df = load_data(args)

    rna_candidates = [x.split('__')[1] for x in func_names if x.startswith('RNA__')]  # uniprot ids 
    prot_candidates = [x.split('__')[1] for x in func_names if x.startswith('PROTEIN__')] # uniprot ids 

    landmark_mapping = get_unique_landmark_mapping(landmark_mapping, rna_candidates)

    dtis = filter_dtis(args, dtis, prot_candidates, drugs)

    check_all_drugs_have_targets(dtis, drugs)

    lincs = list(landmark_mapping.values()) # lincs landmark genes with uniprot ids
    func_names, func_df, dtis, drugs, lincs = filter_func_nodes(func_names, func_df, dtis, lincs, drugs, filter_depth=args.filter_depth)

    input_names = ['DRUG__' + p for p in drugs] + ['LINE__' + c for c in lines] + ['GENE__' + g for g in gene_names] 
    output_names = ['GENE__' + g for g in gene_names]

    input_name2idx = {n:i for i,n in enumerate(input_names)}
    output_name2idx = {n:i for i,n in enumerate(output_names)}
    function_name2idx = {n:i for i,n in enumerate(func_names)}

    func_df = func_df.assign(src_idx = [function_name2idx[f] for f in func_df.source.values])
    func_df = func_df.assign(dst_idx = [function_name2idx[f] for f in func_df.target.values])

    input_edge_index = make_input_edge_index(dtis, func_names, landmark_mapping, gene_names, input_name2idx, function_name2idx, lines)
    func_edge_index = make_function_edge_index(func_df) 
    output_edge_index = make_output_edge_index(gene_names, landmark_mapping, func_names, function_name2idx, output_name2idx)

    data = pyg.data.HeteroData() 

    data['edge_index_dict'] = {
        ('input',       'to',           'function')     : input_edge_index, 
        ('function',    'to',           'function')     : func_edge_index, 
        ('function',    'to',           'output')       : output_edge_index, 
    }

    data['node_names_dict'] = {'input':input_names,
                                'function':func_names,
                                'output':output_names}
    
    torch.save(data, f'{args.out}/bionetwork.pt')

    print('--'*40)
    print('Bio-network summary:')
    print(f'\tInput nodes: {len(input_names)}')
    print(f'\tFunction nodes: {len(func_names)}')
    print(f'\t\t # Protein nodes: {len([f for f in func_names if f.startswith("PROTEIN__")])}')
    print(f'\t\t # RNA nodes: {len([f for f in func_names if f.startswith("RNA__")])}')
    print(f'\tOutput nodes: {len(output_names)}')
    print(f'\tInput edges: {data["edge_index_dict"][("input", "to", "function")].shape[1]}')
    print(f'\tFunction edges: {data["edge_index_dict"][("function", "to", "function")].shape[1]}')
    print(f'\tOutput edges: {data["edge_index_dict"][("function", "to", "output")].shape[1]}')
    print('--'*40) 


    checks(data)