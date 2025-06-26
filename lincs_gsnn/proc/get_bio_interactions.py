import omnipath as op 
import pandas as pd 
import copy 
import numpy as np


def get_bio_interactions(undirected=False, include_mirna=False, include_extra=False, dorothea_levels=['A', 'B']): 
    '''
    retrieve and process the omnipath interactions. 
    '''

    dorothea        = op.interactions.Dorothea().get(organism = 'human', dorothea_levels=dorothea_levels)
    omnipath        = op.interactions.OmniPath().get(organism = 'human')

    if include_extra:
        pathways_extra  = op.interactions.PathwayExtra().get(organism = 'human')
        kin_extra       = op.interactions.KinaseExtra().get(organism = 'human')

    if include_mirna: 
        tf_mirna        = op.interactions.TFmiRNA().get(organism = 'human')
        mirna           = op.interactions.miRNA().get(organism = 'human')

    doro = dorothea.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['RNA__' + y for y in x.target], 
                        edge_type = 'dorothea')[['source', 'target', 'edge_type']]

    omni = omnipath.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                        target = lambda x: ['PROTEIN__' + y for y in x.target], 
                        edge_type = 'omnipath')[['source', 'target', 'edge_type']]
    
    interactions = [doro, omni]
    
    if include_extra:
        # interactions without literature reference 
        path = pathways_extra.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                            target = lambda x: ['PROTEIN__' + y for y in x.target], 
                            edge_type = 'pathways_extra')[['source', 'target', 'edge_type']]   
        
        kin = kin_extra.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                            target = lambda x: ['PROTEIN__' + y for y in x.target], 
                            edge_type = 'kinase_extra')[['source', 'target', 'edge_type']]

        interactions += [path, kin]  

    if include_mirna:
        tfmirna = tf_mirna.assign(source = lambda x: ['PROTEIN__' + y for y in x.source],
                            target = lambda x: ['RNA__' + y for y in x.target], 
                            edge_type = 'tf_mirna')[['source', 'target', 'edge_type']]

        mirna_ = mirna.assign(source = lambda x: ['RNA__' + y for y in x.source],
                            target = lambda x: ['RNA__' + y for y in x.target], 
                            edge_type = 'mirna')[['source', 'target', 'edge_type']] 
        
        interactions += [tfmirna, mirna_]
    
    # get translation interactions 
    _fdf = pd.concat(interactions, axis=0, ignore_index=True)

    _fnames = _fdf.source.values.tolist() + _fdf.target.values.tolist()
    rna_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'RNA']
    protein_space = [x.split('__')[1] for x in _fnames if x.split('__')[0] == 'PROTEIN']
    RNA_PROT_OVERLAP = list(set(rna_space).intersection(set(protein_space)))
    trans = pd.DataFrame({'source': ['RNA__' + x for x in RNA_PROT_OVERLAP],
                        'target': ['PROTEIN__' + x for x in RNA_PROT_OVERLAP],
                        'edge_type':'translation'})
    print('# of translation (RNA->PROTEIN) edges:', len(trans))

    # combine all edges 
    func_df = pd.concat(interactions + [trans], axis=0, ignore_index=True)

    if undirected: 
        print('making function graph undirected (adding reverse edges)')
        func_df2 = copy.deepcopy(func_df)
        func_df2 = func_df2.rename({'target':'source', 'source':'target'}, axis=1)
        func_df = pd.concat((func_df, func_df2), ignore_index=True, axis=0)
        func_df = func_df.drop_duplicates()
        func_df = func_df.dropna()

    func_names = np.unique(func_df.source.tolist() + func_df.target.tolist()).tolist()

    return func_names, func_df
