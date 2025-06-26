from lincs_gsnn.data.DXDTDataset import DXDTDataset
import torch
from torch.utils.data import DataLoader
from lincs_gsnn.models.SplineWeightEmbedding import SplineWeightEmbedding
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np



def freeze_(model):
    for param in model.parameters():
        param.requires_grad = False 

def infer_edge_weights(model, data, cond, target_genes, obs_dir, dxdt_scale, 
                       n_ctrl_pts=5, degree=3, prior=0, lr=5e-2, beta=5e-4, 
                       epochs=250, batch_size=600, num_workers=2, dropout=0): 
    
    target_ixs = torch.tensor([data.node_names_dict['output'].index('GENE__' + g) for g in target_genes], dtype=torch.long).view(-1)

    freeze_(model) 
    model = model.eval()

    dataset = DXDTDataset(
                meta = cond, 
                obs_dir=obs_dir,
                scale=dxdt_scale,
                input_names=data.node_names_dict['input'],
                return_time=True
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    E = model.edge_index.shape[1]
    SWE = SplineWeightEmbedding(n_outputs = E, 
                                n_ctrl_pts = n_ctrl_pts, 
                                degree = degree, 
                                prior = prior, 
                                dropout=dropout)  # dropout is optional
    
    crit = torch.nn.MSELoss()
    optim = torch.optim.Adam(SWE.parameters(), lr=lr)

    model.to(device)
    SWE.to(device) 

    metrics = {'norm': [], 'mse': [], 'r2': []}
    for  epochs in range(epochs):
        epoch_metrics = {'norm': 0, 'mse': 0, 'r2': 0}
        for X, dxdt, t in loader:
            optim.zero_grad()

            X = X.to(device)
            dxdt = dxdt.to(device)
            t = t.to(device)

            w = SWE(t).sigmoid()  

            out = model(X, edge_mask=w) 

            out = out[:, target_ixs] 
            dxdt = dxdt[:, target_ixs]

            mse = crit(out, dxdt) 
            norm = w.norm()
            loss = mse + beta * norm

            loss.backward()
            optim.step()

            r2 = r2_score(dxdt.cpu().detach().numpy(), out.cpu().detach().numpy())

            epoch_metrics['norm'] += norm.item()
            epoch_metrics['mse'] += mse.item()
            epoch_metrics['r2'] += r2
        epoch_metrics = {k: v / len(loader) for k, v in epoch_metrics.items()}
        metrics['norm'].append(epoch_metrics['norm'])
        metrics['mse'].append(epoch_metrics['mse'])
        metrics['r2'].append(epoch_metrics['r2'])

        print(f'Epoch {epochs}, norm: {epoch_metrics["norm"]:.4f}, mse: {epoch_metrics["mse"]:.4f}, r2: {epoch_metrics["r2"]:.4f}', end='\r')

    t = torch.linspace(0, 1, 100).to(device) 

    SWE.eval()  # set to eval mode to avoid dropout
    wt = SWE(t).sigmoid().cpu().detach().numpy() # shape (100, E)

    return wt, SWE, metrics

