
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

from pypath.utils import mapping


def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",               type=str,               default='../../../data/',                  help="path to data directory")
    parser.add_argument("--out",                type=str,               default='../../proc/',                     help="path to data directory")
    parser.add_argument("--pretrained",         type=str,               default='../../proc/pretrained',           help="path to pretrained model directory")
    parser.add_argument("--bionet",             type=str,               default='../../proc/bionetwork.pt',        help="path to bionetwork file")
    parser.add_argument("--batch_size",         type=int,               default=32,                                help="batch size for training")
    parser.add_argument("--horizon",            type=int,               default=-1,                                help="horizon for the model")
    parser.add_argument("--multiple_shooting",  action='store_true',    default=False,                             help="whether to use multiple shooting")
    parser.add_argument("--lr",                 type=float,             default=1e-3,                              help="learning rate for optimizer")
    parser.add_argument("--beta",               type=float,             default=1e-3,                              help="weight for edge weight regularization")
    parser.add_argument("--prior",              type=float,             default=3,                                 help="initial prior for edge weights (pos values -> prob > 0.5)")
    parser.add_argument("--epochs",             type=int,               default=250,                               help="number of epochs to train for")
    parser.add_argument("--gene_targets",        type=str, nargs='+',   default=['GNPDA1'],                          help="gene symbol target to explain")
    parser.add_argument("--drugs",               type=str, nargs='+',   default=['BRD-K49328571'],                   help="drug to use for the condition")
    parser.add_argument("--cells",               type=str, nargs='+',   default=['HME1'],                            help="cell line to use for the condition")

    args = parser.parse_args() 
    if args.horizon < 0: 
        args.horizon = None

    assert len(args.gene_targets) == len(args.drugs) == len(args.cells), \
        f'Number of gene targets ({len(args.gene_targets)}), drugs ({len(args.drugs)}) and cells ({len(args.cells)}) must be the same.'

    return args 



def load_data(args): 

    data = torch.load(f'{args.bionet}/bionetwork.pt', weights_only=False)
    model = torch.load(f'{args.pretrained}/pretrained_model.pt', weights_only=False)
    dxdt_scale = torch.load(f'{args.pretrained}/dxdt_scale.pt', weights_only=False).item()
    gene_names = pd.read_csv(f'{args.data}/gene_names.csv')['gene_names'].values.astype(str)
    meta = pd.read_csv(f'{args.data}/pred_meta.csv')

    return data, model, dxdt_scale, gene_names, meta 

def freeze_(model): 
    for param in model.parameters(): 
        param.requires_grad = False

def get_t(args, meta):

    min_time = meta.time_min.values[0]
    max_time = meta.time_max.values[0] 
    n_time = meta.n_time_pts.values[0]
    dt = (max_time - min_time) / (n_time - 1) 

    if args.horizon is None: 
        args.horizon = n_time

    t = torch.linspace(0, args.horizon*dt, args.horizon, device=device)
    return t 

def train(args, func, optim, loader, edge_weight, crit, t, ): 
    
    print() 
    metrics ={'mse':[], 'r2':[], 'edge_weight_norm':[]}
    for epoch in range(args.epochs):
        epoch_metrics = {'mse':0, 'r2':0, 'edge_weight_norm':0}
        for i, (xt, _, X) in enumerate(dataloader):
            optim.zero_grad()

            X = X.to(device); xt = xt.to(device)

            # set the edge weight for the GSNN 
            func.set_edge_mask(edge_weight.sigmoid())

            out = odeint(func=func, y0=X, t=t, method='dopri5').transpose(0, 1)  # (n_time, B, n_input_nodes) 
            xt_hat = out[:, :, func.gene_ixs]  # (n_time, B, n_genes)

            mse = crit(xt_hat[:, :, args.target_ix], xt[:, :, args.target_ix]) 
            loss = mse + args.beta*edge_weight.sigmoid().norm()
            loss.backward()
            optim.step()

            # Compute R2 based on change from baseline 
            delta_ = xt.detach().cpu().numpy() - xt[:, [0], :].detach().cpu().numpy()
            delta_hat_ = xt_hat.detach().cpu().numpy() - xt[:, [0], :].detach().cpu().numpy()
            delta_ = delta_[:, :, args.target_ix]
            delta_hat_ = delta_hat_[:, :, args.target_ix]
            r2 = r2_score(delta_.ravel(), delta_hat_.ravel(), multioutput='uniform_average')

            epoch_metrics['mse'] += mse.item()
            epoch_metrics['r2'] += r2
            epoch_metrics['edge_weight_norm'] += edge_weight.sigmoid().norm().item()
        epoch_metrics = {k: v / len(dataloader) for k, v in epoch_metrics.items()}
        mse = epoch_metrics['mse']
        r2 = epoch_metrics['r2']
        edge_weight_norm = epoch_metrics['edge_weight_norm'] 

        metrics['mse'].append(mse) 
        metrics['r2'].append(r2)
        metrics['edge_weight_norm'].append(edge_weight_norm)

        print(f'Epoch {epoch}, mse: {mse:.3f}, Loss: {loss:.3f}, r2: {r2:.3f}, Edge Weight Norm: {edge_weight.sigmoid().norm().item():.3f}', end='\r')
    
    print() 
    return edge_weight, metrics

def eval(args, cond, data, edge_weight, func, device): 

    full_dataset = TrajDataset(cond, 
                      input_names=data.node_names_dict['input'], 
                      obs_dir=f'{args.data}/obs')

    full_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    min_time = full_dataset.meta.time_min.values[0]
    max_time = full_dataset.meta.time_max.values[0]
    n_time = full_dataset.meta.n_time_pts.values[0]
    t = torch.linspace(min_time, max_time, n_time, device=device)

    def eval_():
        xt_hats = [] 
        xt_mus = [] 
        for xt_mu, xt_sigma, X in full_dataloader: 

            xt_mu = xt_mu.to(device)
            xt_sigma = xt_sigma.to(device)
            X = X.to(device)

            with torch.no_grad():
                out = odeint(func=func, y0=X, t=t, method='dopri5').transpose(0, 1)  # (B, n_time, n_input_nodes) 

            xt_hat = out[:, :, func.gene_ixs]  # (n_time, B, n_genes)
            xt_hats.append(xt_hat) 
            xt_mus.append(xt_mu)

        xt_hat = torch.cat(xt_hats, dim=0)  # (n_time, B, n_genes)
        xt_mu = torch.cat(xt_mus, dim=0)  # (n_time, B, n_genes)

        delta_ = xt_mu.detach().cpu().numpy() - xt_mu[:, [0], :].detach().cpu().numpy()
        delta_hat_ = xt_hat.detach().cpu().numpy() - xt_mu[:, [0], :].detach().cpu().numpy()

        # select target ix only 
        delta_ = delta_[:, :, args.target_ix]
        delta_hat_ = delta_hat_[:, :, args.target_ix]

        r2 = r2_score(delta_.ravel(), delta_hat_.ravel(), multioutput='uniform_average')
        mse = ((delta_ - delta_hat_) ** 2).mean()

        metrics = {'mse': mse, 'r2': r2} 

        return xt_mu[:, :, args.target_ix], xt_hat[:, :, args.target_ix], metrics 
    

    func.set_edge_mask(None)
    xt_mu_unmasked, xt_hat_unmasked, metrics_unmasked = eval_()

    func.set_edge_mask(1.*(edge_weight.sigmoid() > 0.5))
    xt_mu_masked, xt_hat_masked, metrics_masked = eval_()

    metrics = {'unmasked': metrics_unmasked, 'masked': metrics_masked} 
    with open(f'{args.out}/edge_masking_metrics.json', 'w') as f:
        f.write(str(metrics))

    f, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for i in range(xt_mu_masked.shape[0]):
        if i == 0: 
            label1 = 'True'
            label2 = 'Predicted'
        else: 
            label1 = None
            label2 = None

        axes[0].plot(t.cpu().numpy(), xt_mu_masked.detach().cpu().numpy()[i, :].ravel(), 'k', label=label1)
        axes[0].plot(t.cpu().numpy(), xt_hat_masked.detach().cpu().numpy()[i, :].ravel(), 'r--', label=label2)
    axes[0].set_title('Masked Edge Weights')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Gene Expression') 
    axes[0].legend() 

    for i in range(xt_mu_unmasked.shape[0]):
        if i == 0: 
            label1 = 'True'
            label2 = 'Predicted'
        else: 
            label1 = None
            label2 = None

        axes[1].plot(t.cpu().numpy(), xt_mu_unmasked.detach().cpu().numpy()[i, :].ravel(), 'k', label=label1)
        axes[1].plot(t.cpu().numpy(), xt_hat_unmasked.detach().cpu().numpy()[i, :].ravel(), 'r--', label=label2)


    axes[1].set_title('Unmasked Edge Weights')
    axes[1].set_xlabel('Time')
    axes[1].legend() 

    plt.tight_layout() 
    plt.savefig(f'{args.out}/edge_masking_performance_comparison.png')
    plt.close() 

def plot_training_metrics(args, metrics): 

    plt.figure(figsize=(10, 6))
    plt.plot(metrics['r2'], label='R2')
    plt.plot(np.array(metrics['edge_weight_norm']) / 100, label='Edge Weight Norm (x100)')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Training Metrics')
    plt.savefig(f'{args.out}/training_metrics.png')
    plt.close()

def plot_edge_weight_hist(args, edge_weight):
    plt.figure(figsize=(10, 6))
    plt.hist(edge_weight.sigmoid().detach().cpu().numpy(), bins=50, alpha=0.7)
    plt.xlabel('Edge Weight')
    plt.ylabel('counts')
    plt.title('Edge Weight Distribution')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.out}/edge_weight_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def annotate_edges(model, edge_weight): 

    res = pd.DataFrame({'src': np.array(model.homo_names)[model.edge_index[:, model.function_edge_mask][0].cpu()],
                     'dst': np.array(model.homo_names)[model.edge_index[:, model.function_edge_mask][1].cpu()],
                     'weight': edge_weight.sigmoid().detach().cpu().numpy()[model.function_edge_mask.cpu()]})

    res = res.assign(src_uniprot = [x.split('__')[1] for x in res.src])
    res = res.assign(dst_uniprot = [x.split('__')[1] for x in res.dst])

    res = res.assign(src_type = [x.split('__')[0] for x in res.src])
    res = res.assign(dst_type = [x.split('__')[0] for x in res.dst])

    res.sort_values('weight', ascending=False)

    return res 

def get_drug_edges(args, data): 

    drug_idx = data.node_names_dict['input'].index('DRUG__' + args.drug)
    row,col = data.edge_index_dict['input','to','function'] 
    drug_edge_mask = row == drug_idx 
    drug_edges = data.edge_index_dict['input','to','function'][:, drug_edge_mask]

    return drug_edges


def make_subgraph(args, res, drug_edges, target_node): 

    G = nx.DiGraph()
    for i,row in res[lambda x: x.weight > 0.5].iterrows(): 
        G.add_edge(row.src, row.dst)

    ii = 0
    for i in range(drug_edges.shape[1]): 
        src,dst = drug_edges[:, i] 
        src_name = data.node_names_dict['input'][src]
        dst_name = data.node_names_dict['function'][dst]

        if dst_name in G: 
            ii+=1
            G.add_edge(src_name, dst_name)

    assert ii > 0, f'No edges found for drug {args.drug} in the graph'

    # remove any nodes that is not an ancestor of the target node 
    ancestors = nx.ancestors(G, target_node)
    G = G.subgraph(list(ancestors) + [target_node]) 

    # add node type 
    for node in G.nodes():
        node_type, uniprot_id = node.split('__')
        nx.set_node_attributes(G, {node: node_type}, 'node_type')

        try: 
            gene_name = list(mapping.map_name(uniprot_id, 'uniprot', 'genesymbol'))[0]
        except: 
            gene_name = uniprot_id

        nx.set_node_attributes(G, {node: gene_name}, 'node_name')

    return G


def plot_graph(G, ns=2500): 


    node_color = []
    for node in G.nodes(): 
        node_type = G.nodes[node]['node_type']
        if node_type == 'RNA':
            node_color.append('r')
        elif node_type == 'PROTEIN':
            node_color.append('g')
        elif node_type == 'DRUG':
            node_color.append('b')
        else: 
            node_color.append('gray')

    H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
    H_layout = nx.nx_pydot.pydot_layout(H, prog="dot")
    pos = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    plt.figure(figsize=(40,15)) 
    nx.draw_networkx_nodes(G, pos, node_size=ns, alpha=1., node_shape='o', node_color=node_color)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=1., node_size=ns)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_family='sans-serif', labels={n: G.nodes[n]['node_name'] for n in G.nodes()})

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='RNA', markerfacecolor='r', markersize=25),
        plt.Line2D([0], [0], marker='o', color='w', label='PROTEIN', markerfacecolor='g', markersize=25),
        plt.Line2D([0], [0], marker='o', color='w', label='DRUG', markerfacecolor='b', markersize=25)
    ], loc='upper left', fontsize=25)
    
    plt.savefig(f'{args.out}/condition_subgraph.png', dpi=300, bbox_inches='tight')

def run(args, data, model, func, t, cond_meta, dataloader): 
    edge_weight = torch.nn.Parameter(args.prior*torch.ones(model.edge_index.shape[1], device=device), requires_grad=True)
    optim = torch.optim.Adam([edge_weight], lr=args.lr) 
    crit = torch.nn.MSELoss() 

    edge_weight, metrics = train(args, func, optim, dataloader, edge_weight, crit, t) 

    eval(args, cond_meta, data, edge_weight, func, device)

    plot_training_metrics(args, metrics)
    plot_edge_weight_hist(args, edge_weight)

    ## create subgraph for the condition
    res = annotate_edges(model, edge_weight)
    drug_edges = get_drug_edges(args, data)

    # target_node is uniprot id of the target gene 
    # get this by the edge from the function node (target: RNA_uniprot_id) to the LINCS node (LINCS_gene_name) 
    src,dst = data.edge_index_dict['function', 'to', 'output']
    target_func_ix = src[dst == args.target_ix] 
    assert target_func_ix.shape[0] == 1, f'Expected exactly one function edge for target gene {args.gene_target}, found {target_func_ix.shape[0]}' 
    target_node = data.node_names_dict['function'][target_func_ix[0]]  # this is the function node for the target gene
    print(f'Target FUNCTION node for {args.gene_target} is {target_node}')

    G = make_subgraph(args, res, drug_edges, target_node)
    print(f'Number of edges in the subgraph: {len(G.edges())}')
    print(f'Number of nodes in the subgraph: {len(G.nodes())}')
    #pkl.dump(G, open(f'{args.out}/condition_nx_subgraph.pkl', 'wb'))

    assert len(G.edges()) < 2500, f'subgraph has too many edges to plot graph'

    plot_graph(G, ns=2500)

if __name__ == '__main__': 

    args = get_args() 
    print('--'*40)
    print('Arguments:')
    print(args)
    print('--'*40)

    # Load data
    data, model, dxdt_scale, gene_names, meta = load_data(args)

    freeze_(model)  # Freeze the model parameters
    model = model.eval() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    func = ODEFunc(model, scale=dxdt_scale, input_names=data.node_names_dict['input']).to(device)
    t = get_t(args, meta)

    out_root = args.out
    for gene_target, drug, cell in zip(args.gene_targets, args.drugs, args.cells): 
        print(f'Running explanation for condition: {gene_target}, drug: {drug}, cell: {cell}')
        cond_out = f'{out_root}/{gene_target}_{drug}_{cell}'
        os.makedirs(cond_out, exist_ok=True)
        args.gene_target = gene_target
        args.drug = drug
        args.cell = cell
        args.out = cond_out 
        args.target_ix = gene_names.tolist().index(args.gene_target) 

        cond_meta = meta[(meta['pert_id'] == args.drug) & (meta['cell_iname'] == args.cell)] 
        dataset = TrajDataset(cond_meta, 
                        input_names=data.node_names_dict['input'], 
                        obs_dir=f'{args.data}/obs',
                        horizon=args.horizon, 
                        multiple_shooting=args.multiple_shooting) 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 

        run(args, data, model, func, t, cond_meta, dataloader)

    







