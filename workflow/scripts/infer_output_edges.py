
import torch 
import pandas as pd 
import numpy as np
import os
from torch.utils.data import DataLoader 

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

import networkx as nx 
import pickle as pkl
import argparse
from torchdiffeq import odeint
from lincs_gsnn.models.ODEFunc import ODEFunc
from lincs_gsnn.data.TrajDataset import TrajDataset
from lincs_gsnn.data.DXDTDataset import DXDTDataset
from gsnn.optim.OutputEdgeInferer import OutputEdgeInferer

from pypath.utils import mapping


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--gsnn_root",          type=str,               default='../../workflow_outputs/lincs-gsnn', help="path to gsnn root directory")
    parser.add_argument("--traj_root",          type=str,               default='../../workflow_outputs/lincs-traj', help="path to traj root directory")
    parser.add_argument("--dxdt_dir",           type=str,               default='../../workflow_outputs/lincs-traj/runs/exp/default_v02/output/predict_grid/dxdt', help="path to dxdt directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                     help="path to data directory")
    parser.add_argument("--pretrained",         type=str,               default='../../proc/pretrained',           help="path to pretrained model directory")
    parser.add_argument("--batch_size",         type=int,               default=128,                              help="batch size for training")
    parser.add_argument("--train_prop",         type=float,             default=0.75,                              help="proportion of data to use for training")
    parser.add_argument("--lr",                 type=float,             default=1e-2,                             help="learning rate for optimizer")
    parser.add_argument("--wd",                 type=float,             default=1e-4,                             help="weight decay for optimizer")
    parser.add_argument("--epochs",             type=int,               default=2,                                help="number of epochs to train for")
    parser.add_argument("--agg",                type=str,               default='all',                            help="aggregation type for latent space [all, mean, sum, max, last]")
    parser.add_argument("--use_batchnorm",      action='store_true',     default=True,                             help="whether to use batchnorm in the model")
    parser.add_argument("--sample",             type=str,               required=True,                             help="sample directory name (e.g., sample_0)")
    parser.add_argument("--verbose",            action='store_true',     default=False,                             help="verbose output")
    
    args = parser.parse_args() 

    return args 



def load_data(args): 

    data = torch.load(f'{args.traj_root}/bionetwork.pt', weights_only=False)
    model = torch.load(f'{args.pretrained}/pretrained_model_{args.sample}.pt', weights_only=False).eval()
    dxdt_scale = torch.load(f'{args.pretrained}/dxdt_scale_{args.sample}.pt', weights_only=False).item()
    x_names = pd.read_csv(f'{args.gsnn_root}/gene_names.csv')['gene_names'].values.astype(str)
    dxdt_meta = pd.read_csv(f'{args.gsnn_root}/dxdt_meta.csv')
    x_meta = pd.read_csv(f'{args.gsnn_root}/pred_meta.csv')

    valid_drugs = [x.split('DRUG__')[1] for x in data.node_names_dict['input'] if 'DRUG__' in x]

    dxdt_meta = dxdt_meta[dxdt_meta['pert_id'].isin(valid_drugs)] 
    x_meta = x_meta[x_meta['pert_id'].isin(valid_drugs)] 

    return data, model, dxdt_scale, x_names, dxdt_meta, x_meta 

def freeze_(model): 
    for param in model.parameters(): 
        param.requires_grad = False


if __name__ == '__main__': 

    args = get_args() 

    print('-'*100)
    print(args)
    print('-'*100)

    data, model, dxdt_scale, x_names, dxdt_meta, x_meta = load_data(args) 

    freeze_(model)
    model.eval() 

    train_ids = dxdt_meta.sample(frac=args.train_prop).index 
    test_ids = dxdt_meta.index.difference(train_ids) 
    train_cond = dxdt_meta.loc[train_ids]
    test_cond = dxdt_meta.loc[test_ids]

    # Determine obs and dxdt directories for the sample
    sample_obs_dir = f'{args.gsnn_root}/{args.sample}/obs'
    sample_dxdt_dir = args.dxdt_dir

    train_dataset = DXDTDataset(train_cond, 
                            input_names=data.node_names_dict['input'], 
                            output_names=data.node_names_dict['output'], 
                            src_names=x_names, 
                            obs_dir=sample_obs_dir,
                            dxdt_dir=sample_dxdt_dir,
                            scale=dxdt_scale) 

    test_dataset = DXDTDataset(test_cond, 
                            input_names=data.node_names_dict['input'], 
                            output_names=data.node_names_dict['output'], 
                            src_names=x_names, 
                            obs_dir=sample_obs_dir,
                            dxdt_dir=sample_dxdt_dir,
                            scale=dxdt_scale) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    latent_dim = model.channels*model.layers if args.agg == 'all' else model.channels
    OEI = OutputEdgeInferer(data, 
                            latent_dim, 
                            lr=args.lr, 
                            wd=args.wd, 
                            epochs=args.epochs, 
                            agg=args.agg, 
                            use_batchnorm=args.use_batchnorm)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _ = OEI.fit(train_loader, model, device=device)
    res  = OEI.evaluate(dataloader=test_loader, model=model, device=device, verbose=args.verbose)
    res = res.assign(dxdt_dir=args.dxdt_dir)

    res.to_csv(f'{args.out}/inferred_output_edges_test.csv')
