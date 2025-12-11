'''
Target Gene - The gene of interest that is being explained
Cell line comparisons (cell line 1 vs cell line 2) - The two cell lines that are being compared 
Sample id - The sample id that is being explained 

'''

import argparse
import pandas as pd
import torch
import os

from lincs_gsnn.data.DXDTDataset import DXDTDataset
from torch.utils.data import DataLoader
from lincs_gsnn.models.ODEWrapper import ODEWrapper
from gsnn.interpret.ContrastiveGSNNExplainer import ContrastiveGSNNExplainer
from gsnn.interpret.ContrastiveIGExplainer import ContrastiveIGExplainer
from gsnn.interpret.ContrastiveOcclusionExplainer import ContrastiveOcclusionExplainer



def get_args():
    parser = argparse.ArgumentParser(description='Generate contrastive explanation for a given target gene, cell line comparison, and sample id')

    parser.add_argument('--root_gsnn', type=str, required=True,
                       help='Root directory of the gsnn outputs')
    parser.add_argument('--root_traj', type=str, required=True,
                       help='Root directory of the traj outputs')
    parser.add_argument('--sample_id', type=str, required=True,
                       help='Sample id of the sample to explain')

    parser.add_argument('--target_gene', type=str, required=True,
                       help='Target gene of interest')
    parser.add_argument('--cell_line_1', type=str, required=True,  
                       help='First cell line in the comparison')
    parser.add_argument('--cell_line_2', type=str, required=True,
                       help='Second cell line in the comparison')
    parser.add_argument('--pert_id', type=str, required=True,
                       help='Perturbation id of the perturbation to explain')
    parser.add_argument('--dose', type=float, default=10.0,
                       help='Dose of the perturbation to explain')
    parser.add_argument('--horizon', type=int, default=72,
                       help='Horizon of the perturbation to explain (time in hours) (default: 72)')
    parser.add_argument('--n_time_pts', type=int, default=20,
                       help='Number of time points to explain (default: 20)')
    parser.add_argument('--method', type=str, default='euler',
                       choices=['dopri5', 'euler'],
                       help='Method to use for the ode integration (default: euler)')
    parser.add_argument('--tol', type=float, default=1e-4,
                       help='Tolerance for the ode integration (dorpi5 only) (default: 1e-4)')
    parser.add_argument('--hard', type=bool, default=False,
                       help='Whether to use a hard threshold for the explanation (default: False)')
    parser.add_argument('--beta', type=float, default=5e-4,
                       help='Beta for the explanation (default: 5e-4)')
    parser.add_argument('--lr', type=float, default=5e-2,
                       help='Learning rate for the explanation (default: 5e-2)')
    parser.add_argument('--prior', type=float, default=5,
                       help='Prior for the explanation (default: 5)')
    parser.add_argument('--iters', type=int, default=250,
                       help='Number of iterations for the explanation (default: 250)')
    parser.add_argument('--entropy', type=float, default=1.,
                       help='Entropy for the explanation (default: 1.)')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                       help='Gradient norm clip for the explanation (default: 1.0)')
    parser.add_argument('--free_edges', type=int, default=100,
                       help='Number of free edges for the explanation (default: 100)')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for the explanation results')

    parser.add_argument('--explanation_target', type=str, default='edge',
                       choices=['edge', 'node'],
                       help='Target for the explanation (default: edge)')

    args = parser.parse_args()

    args.dxdt_dir = os.path.join(args.root_traj, f'predict_grid/{args.sample_id}/dxdt/')
    args.obs_dir = os.path.join(args.root_traj, f'predict_grid/{args.sample_id}/obs/')
    
    args.data = torch.load(os.path.join(args.root_gsnn, 'bionetwork/bionetwork.pt'), weights_only=False)
    args.model = torch.load(os.path.join(args.root_gsnn, f'pretrain/pretrained_model_{args.sample_id}.pt'), weights_only=False)
    args.dxdt_scale = torch.load(os.path.join(args.root_gsnn, f'pretrain/dxdt_scale_{args.sample_id}.pt'), weights_only=False).item()
    args.x_names = pd.read_csv(os.path.join(args.root_traj, 'predict_grid/gene_names.csv'))['gene_names'].values.astype(str).tolist()
    args.dxdt_meta = pd.read_csv(os.path.join(args.root_traj, 'predict_grid/dxdt_meta.csv'))

    args.cond1 = args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_1) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)]
    args.cond2 = args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_2) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)] 

    # freeze the model 
    for param in args.model.parameters():
        param.requires_grad = False
    args.model = args.model.eval()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.t = torch.linspace(0, args.horizon, args.n_time_pts, device=args.device)
    args.model = args.model.to(args.device)

    args.target_gene = 'GENE__' + args.target_gene 

    try: 
        args.target_gene_output_ix = args.data.node_names_dict['output'].index(args.target_gene) 
        args.target_gene_input_ix = args.data.node_names_dict['input'].index(args.target_gene) 
    except ValueError:
        raise ValueError(f'Target gene {args.target_gene} not found in the data')

    assert args.cell_line_1 in args.dxdt_meta.cell_iname.unique(), f'Cell line {args.cell_line_1} not found in the data'
    assert args.cell_line_2 in args.dxdt_meta.cell_iname.unique(), f'Cell line {args.cell_line_2} not found in the data'
    assert args.pert_id in args.dxdt_meta.pert_id.unique(), f'Perturbation {args.pert_id} not found in the data'
    assert args.dose in args.dxdt_meta.dose.unique(), f'Dose {args.dose} not found in the data'
    assert args.horizon <= args.dxdt_meta.time.max(), f'Horizon {args.horizon} is greater than the maximum number of time points {args.dxdt_meta.time.max()}' 

    return args 


def retrieve_data(dataloader):

    xs = []; ys = [] 
    for i, (x, y) in enumerate(dataloader):
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    return xs, ys 

def get_x(args): 

    dataset1 = DXDTDataset(meta=args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_1) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)], 
                      input_names=args.data.node_names_dict['input'], 
                      output_names=args.data.node_names_dict['output'],
                      src_names=args.x_names,
                      obs_dir=args.obs_dir,
                      dxdt_dir=args.dxdt_dir,
                      scale=args.dxdt_scale)

    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=False)

    dataset2 = DXDTDataset(meta=args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_2) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)], 
                      input_names=args.data.node_names_dict['input'], 
                      output_names=args.data.node_names_dict['output'],
                      src_names=args.x_names,
                      obs_dir=args.obs_dir,
                      dxdt_dir=args.dxdt_dir,
                      scale=args.dxdt_scale)

    dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=False)

    x1, _ = retrieve_data(dataloader1)
    x2, _ = retrieve_data(dataloader2)

    return x1, x2

def predict_xt(args, x):

    x0 = x[[0]]

    ode_gsnn = ODEWrapper(args.model, 
                        args.data, 
                        args.dxdt_scale, 
                        t=args.t,     
                        method= args.method, 
                        tol=args.tol)

    xt_hat = ode_gsnn(x0.to(args.device))

    # Use x[0] to get shape (N_features,) then expand to (T, N_features)
    xt_hat_input = x[0].clone().unsqueeze(0).expand(xt_hat.shape[0], -1,).clone()
    xt_hat_input[:, :xt_hat.shape[1]] = xt_hat

    return xt_hat_input, xt_hat


def run_contrastive_explanation(args, x1, x2, xt_hat_w_inputs_1, xt_hat_w_inputs_2, target_ix): 


    ####################################################################################################
    # GSNN Explainer
    ####################################################################################################

    ode_gsnn = ODEWrapper(args.model, 
                        args.data, 
                        args.dxdt_scale, 
                        t=args.t,     
                        method= args.method, 
                        tol=args.tol)

    explainer = ContrastiveGSNNExplainer(ode_gsnn, 
                                        args.data,
                                        hard=args.hard,
                                        beta=args.beta,
                                        lr=args.lr,
                                        prior=args.prior,
                                        iters=args.iters,
                                        free_edges=args.free_edges)


    x0_1 = x1[[0]]                                         
    x0_2 = x2[[0]]
    cres_gsnn = explainer.explain(x0_1.to(args.device), 
                                  x0_2.to(args.device), 
                                  target_idx=args.target_gene_output_ix, 
                                  target=args.explanation_target)

    ####################################################################################################
    # IG Explainer
    ####################################################################################################
    
    explainer = ContrastiveIGExplainer(args.model.to(args.device), args.data)

    cres_ig = explainer.explain(xt_hat_w_inputs_1.to(args.device), 
                                xt_hat_w_inputs_2.to(args.device), 
                                target_idx=args.target_gene_output_ix, 
                                element_mask=cres_gsnn.score.values > 0.5,
                                target=args.explanation_target) 

    ####################################################################################################
    # Occlusion Explainer
    ####################################################################################################
    explainer = ContrastiveOcclusionExplainer(args.model.to(args.device), args.data)

    cres_occ = explainer.explain(xt_hat_w_inputs_1.to(args.device), 
                                xt_hat_w_inputs_2.to(args.device), 
                                target_idx=args.target_gene_output_ix, 
                                element_mask=cres_gsnn.score.values > 0.5,
                                target=args.explanation_target)

    ####################################################################################################
    # Combine Results
    ####################################################################################################

    if args.explanation_target == 'edge':
        cres = cres_ig.merge(cres_gsnn, on=['source', 'target'], how='inner').rename(columns={'score_x': 'ig_score', 'score_y': 'gsnn_score'})
        cres = cres.merge(cres_occ, on=['source', 'target'], how='left').rename(columns={'score': 'occlusion_score'})
    elif args.explanation_target == 'node':
        cres = cres_ig.merge(cres_gsnn, on=['node'], how='inner').rename(columns={'score_x': 'ig_score', 'score_y': 'gsnn_score'})
        cres = cres.merge(cres_occ, on=['node'], how='left').rename(columns={'score': 'occlusion_score'})
    else:
        raise ValueError(f'Explanation target {args.explanation_target} not supported')

    return cres


if __name__ == '__main__': 
    
    print()
    args = get_args()
    print('--'*40)

    x1, x2 = get_x(args) 
    xt_hat_w_inputs_1, xt_hat_1 = predict_xt(args, x1)
    xt_hat_w_inputs_2, xt_hat_2 = predict_xt(args, x2) 

    cres = run_contrastive_explanation(args, x1, x2, xt_hat_w_inputs_1, xt_hat_w_inputs_2, args.target_gene_output_ix)

    # Save contrastive results as CSV
    cres.to_csv(os.path.join(args.out, f'contrastive_results_{args.sample_id}.csv'), index=False)

    # Save out_dict with cres, trajectory predictions and metadata
    out_dict = {
        'cres': cres,
        'xt_hat_1': xt_hat_1.detach().cpu().numpy(),
        'xt_hat_2': xt_hat_2.detach().cpu().numpy(),
        'x1_observed': x1.detach().cpu().numpy(),
        'x2_observed': x2.detach().cpu().numpy(),
        'target_gene': args.target_gene,
        'target_gene_output_ix': args.target_gene_output_ix,
        'cell_line_1': args.cell_line_1,
        'cell_line_2': args.cell_line_2,
        'pert_id': args.pert_id,
        'dose': args.dose,
        'horizon': args.horizon,
        'n_time_pts': args.n_time_pts,
        't': args.t.cpu().numpy(),
    }
    torch.save(out_dict, os.path.join(args.out, f'contrastive_results_{args.sample_id}.pt'))

    print(f'Contrastive explanation results saved to {args.out}')
    print('--'*40)
    print() 






    