import argparse 
import pandas as pd 
from lincs_gsnn.proc.get_bio_interactions import get_bio_interactions 
from lincs_gsnn.proc.subset import filter_func_nodes
import torch_geometric as pyg
import numpy as np
import torch

from gsnn.models.GSNN import GSNN 
from torch.utils.data import DataLoader 
from sklearn.metrics import r2_score
from lincs_gsnn.data.TrajDataset import TrajDataset
from lincs_gsnn.data.DXDTDataset import DXDTDataset

import time 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                   help="path to data directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                help="path to data directory")
    parser.add_argument("--bionet",             type=str,               default='../../proc/bionetwork.pt',       help="path to bionetwork file")
    parser.add_argument("--batch_size",         type=int,               default=64,                                help="batch size for training")
    parser.add_argument("--num_workers",        type=int,               default=4,                                 help="number of workers for dataloader")
    parser.add_argument("--epochs",             type=int,               default=100,                               help="number of epochs to train for")
    parser.add_argument("--lr",                 type=float,             default=1e-3,                              help="learning rate for optimizer")
    parser.add_argument("--wd",                 type=float,             default=1e-4,                              help="weight decay for optimizer")
    parser.add_argument("--patience",           type=int,               default=10,                                help="patience for learning rate scheduler")
    parser.add_argument("--channels",           type=int,               default=64,                                help="number of channels in the model")
    parser.add_argument("--layers",             type=int,               default=3,                                 help="number of layers in the model")
    parser.add_argument("--share_layers",       action='store_true',    default=False,                              help="whether to share layers between input and output nodes")
    parser.add_argument("--dropout",            type=float,             default=0.1,                               help="dropout rate for the model")
    parser.add_argument("--norm",               type=str,               default='batch',                           help="normalization type for the model [batch, layer, none]")
    parser.add_argument("--checkpoint",         action='store_true',    default=False,                              help="whether to use checkpointing in the model")

    args = parser.parse_args() 
    return args

def train(args, model, optim, scheduler): 

    print('Training model...')
    print()

    for epoch in range(args.epochs): 

        losses = 0
        r2s = 0
        tic = time.time()
        for i, (X, dxdt) in enumerate(dataloader):
            optim.zero_grad()

            X = X.to(device)
            dxdt = dxdt.to(device)

            dxdt_hat = model(X) 

            loss = crit(dxdt_hat, dxdt)  # compute loss 
            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optim.step()
            losses += loss.item()
            r2 = r2_score(dxdt.cpu().numpy(), dxdt_hat.detach().cpu().numpy(), multioutput='uniform_average')
            r2s += r2
            print(f'[batch {i}/{len(dataloader)} -> loss: {loss.item():.2E}, r2: {r2:.2f}]', end='\r')

        scheduler.step(losses/len(dataloader))  # step the scheduler based on the average loss
        lr = scheduler.get_last_lr()[0]

        print(f'--> epoch {epoch+1} -> loss: {losses/len(dataloader):.4E} | r2: {r2s/len(dataloader):.3f} | lr: {lr:.2E} | time/epoch: {time.time() - tic:.2f}s')

    return model

if __name__ == '__main__': 

    args = get_args() 
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)

    dxdt_meta = pd.read_csv(f'{args.data}/dxdt_meta.csv')

    data = torch.load(f'{args.bionet}/bionetwork.pt', weights_only=False)

    dataset = DXDTDataset(dxdt_meta, 
                      input_names=data.node_names_dict['input'], 
                      obs_dir=f'{args.data}/dxdt/')

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GSNN(edge_index_dict=data.edge_index_dict, 
                node_names_dict=data.node_names_dict, 
                channels = args.channels, 
                layers = args.layers, 
                share_layers = args.share_layers,
                dropout = args.dropout, 
                checkpoint=args.checkpoint,
                norm = args.norm).to(device) 
    
    print('# parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=args.patience, threshold=1e-3)
    crit = torch.nn.MSELoss(reduction='mean')

    model = train(args, model, optim, scheduler)

    torch.save(model, f'{args.out}/pretrained_model.pt')
    torch.save(torch.tensor([dataset._scale]), f'{args.out}/dxdt_scale.pt')



    