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
from lincs_gsnn.proc.model_paths import gsnn_model_path
from lincs_gsnn.proc.node_activity import load_node_activity_artifact
from gsnn.interpret.ContrastiveGSNNExplainer import ContrastiveGSNNExplainer
from gsnn.interpret.ContrastiveIGExplainer import ContrastiveIGExplainer
from gsnn.interpret.ContrastiveOcclusionExplainer import ContrastiveOcclusionExplainer



def get_args():
    parser = argparse.ArgumentParser(description='Generate contrastive explanation for a given target gene, cell line comparison, and sample id')

    parser.add_argument('--root_gsnn', type=str, required=True,
                       help='Root directory of the gsnn outputs')
    parser.add_argument('--root_traj', type=str, required=True,
                       help='Root directory of the traj outputs')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Model replicate id (e.g. model_0)')

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

    parser.add_argument('--use_hypernetwork', action='store_true', default=False,
                       help='Use the cell-line-conditioned hypernetwork artifact '
                            '(pretrained_hnet_<sample_id>.pt) to materialize a '
                            'separate GSNN per cell line. Falls back to the legacy '
                            'single-model behavior if the artifact is missing.')

    parser.add_argument('--node_activity_path', type=str, default=None,
                       help='Path to the node_activity.pt artifact. Required when '
                            'the loaded model has node_activity enabled; defaults '
                            'to <root_gsnn>/bionetwork/node_activity.pt.')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to GSNN checkpoint (.pt). Defaults to '
                            'train/trained_model_<sample_id>.pt when present, '
                            'else pretrain/pretrained_model_<sample_id>.pt.')

    args = parser.parse_args()

    args.pred_dir = os.path.join(args.root_traj, 'predict_grid')
    
    args.data = torch.load(os.path.join(args.root_gsnn, 'bionetwork/bionetwork.pt'), weights_only=False)
    model_path = gsnn_model_path(args.root_gsnn, args.model_id, model_path=args.model_path)
    args.model = torch.load(model_path, weights_only=False)
    print(f'Loaded GSNN checkpoint: {model_path}')
    args.dxdt_scale = torch.load(os.path.join(args.root_gsnn, f'pretrain/dxdt_scale_{args.model_id}.pt'), weights_only=False).item()
    args.x_names = pd.read_csv(os.path.join(args.root_traj, 'predict_grid/gene_names.csv'))['gene_names'].values.astype(str).tolist()
    args.dxdt_meta = pd.read_csv(os.path.join(args.root_traj, 'predict_grid/dxdt_meta.csv'))

    args.cond1 = args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_1) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)]
    args.cond2 = args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_2) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)] 

    # Optionally load the hypernetwork artifact and materialize per-cell-line
    # vanilla GSNNs. We keep args.model as the legacy "mean cell line" model
    # (also produced by pretrain) for any code path that may still reference it.
    args.hnet_artifact_path = os.path.join(args.root_gsnn, f'pretrain/pretrained_hnet_{args.model_id}.pt')
    args.use_hypernetwork = bool(args.use_hypernetwork) and os.path.exists(args.hnet_artifact_path)
    args.model_1 = None
    args.model_2 = None

    # freeze the model 
    for param in args.model.parameters():
        param.requires_grad = False
    args.model = args.model.eval()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The GSNN field is parameterized in NORMALIZED time: the full predict_grid
    # trajectory spans tau in [0, 1] over time_max hours, and dx/dt is
    # d(z)/d(tau). --horizon is given in HOURS, so map it onto normalized time
    # (tau = hours / time_max) before integrating. Integrating on the raw hour
    # axis drives the rollout to |z|~1e2-1e3 (same failure fixed in
    # train_gsnn_with_odeint).
    _time_max = float(pd.read_csv(os.path.join(args.pred_dir, 'pred_meta.csv'))['time_max'].iloc[0])
    args.t = torch.linspace(0, args.horizon / _time_max, args.n_time_pts, device=args.device)
    args.model = args.model.to(args.device)

    # ------------------------------------------------------------------
    # Optional per-function-node activity feature.  When the loaded GSNN
    # was trained with `node_activity=True`, look up the per-cell-line
    # x_fn tensors for both arms of the contrast and stash them on args.
    # Mutually exclusive with --use_hypernetwork (cell-line conditioning
    # already lives in args.model_1/args.model_2 in that case).
    # ------------------------------------------------------------------
    args.x_fn_1 = None
    args.x_fn_2 = None
    if getattr(args.model, 'node_activity', False):
        if args.use_hypernetwork:
            raise ValueError(
                "Loaded model has node_activity=True but --use_hypernetwork was set; "
                "the two cell-line conditioning mechanisms are mutually exclusive."
            )
        na_path = args.node_activity_path or os.path.join(args.root_gsnn, 'bionetwork/node_activity.pt')
        if not os.path.exists(na_path):
            raise FileNotFoundError(
                f"Loaded model has node_activity=True but artifact not found at {na_path}. "
                "Pass --node_activity_path explicitly or rebuild with `make_bio_network.py --node_activity`."
            )
        na_payload = load_node_activity_artifact(na_path, node_names_dict=args.data.node_names_dict)
        for cl in (args.cell_line_1, args.cell_line_2):
            if cl not in na_payload['x_fn_by_ciname']:
                raise KeyError(
                    f"node_activity artifact does not contain cell_iname={cl!r}. "
                    f"Available: {sorted(na_payload['x_fn_by_ciname'])[:5]}..."
                )
        args.x_fn_1 = na_payload['x_fn_by_ciname'][args.cell_line_1].unsqueeze(0).to(args.device)
        args.x_fn_2 = na_payload['x_fn_by_ciname'][args.cell_line_2].unsqueeze(0).to(args.device)
        print(f'node_activity: loaded x_fn for {args.cell_line_1} and {args.cell_line_2} '
              f'(shape per cell={tuple(args.x_fn_1.shape)}) from {na_path}')

    if args.use_hypernetwork:
        # Local import keeps the legacy code path free of any hnet dependency.
        from lincs_gsnn.models.HnetGSNN import (
            cell_onehot,
            load_hnet_artifact,
            materialize_gsnn,
        )

        loaded = load_hnet_artifact(args.hnet_artifact_path, args.data)
        hnet = loaded['hnet'].to(args.device).eval()
        cell_lines = loaded['cell_lines']

        C1 = cell_onehot(args.cell_line_1, cell_lines, device=args.device)
        C2 = cell_onehot(args.cell_line_2, cell_lines, device=args.device)
        with torch.no_grad():
            args.model_1 = materialize_gsnn(hnet, C1).to(args.device).eval()
            args.model_2 = materialize_gsnn(hnet, C2).to(args.device).eval()
        for p in args.model_1.parameters():
            p.requires_grad = False
        for p in args.model_2.parameters():
            p.requires_grad = False

        print(f'Hypernetwork mode: materialized cell-line-specific GSNNs for '
              f'{args.cell_line_1} and {args.cell_line_2}.')

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
    xs = []
    for batch in dataloader:
        xs.append(batch[0])
    return torch.cat(xs, dim=0)

def get_x(args): 

    dataset1 = DXDTDataset(meta=args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_1) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)], 
                      input_names=args.data.node_names_dict['input'], 
                      output_names=args.data.node_names_dict['output'],
                      src_names=args.x_names,
                      pred_dir=args.pred_dir,
                      scale=args.dxdt_scale)

    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=False)

    dataset2 = DXDTDataset(meta=args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line_2) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)], 
                      input_names=args.data.node_names_dict['input'], 
                      output_names=args.data.node_names_dict['output'],
                      src_names=args.x_names,
                      pred_dir=args.pred_dir,
                      scale=args.dxdt_scale)

    dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=False)

    x1 = retrieve_data(dataloader1)
    x2 = retrieve_data(dataloader2)

    return x1, x2

def predict_xt(args, x, model=None, x_fn=None):

    x0 = x[[0]]

    if model is None:
        model = args.model

    ode_gsnn = ODEWrapper(model,
                        args.data, 
                        args.dxdt_scale, 
                        t=args.t,     
                        method= args.method, 
                        tol=args.tol)

    # Pass x_fn through ODEWrapper -> ODEFunc.set_x_fn when node_activity
    # is enabled; otherwise the call is byte-identical to the legacy form.
    if x_fn is not None:
        xt_hat = ode_gsnn(x0.to(args.device), x_fn=x_fn)
    else:
        xt_hat = ode_gsnn(x0.to(args.device))

    # Use x[0] to get shape (N_features,) then expand to (T, N_features)
    xt_hat_input = x[0].clone().unsqueeze(0).expand(xt_hat.shape[0], -1,).clone()
    xt_hat_input[:, :xt_hat.shape[1]] = xt_hat

    return xt_hat_input, xt_hat


def _build_ode_router(args, x0_1, x0_2):
    """For ContrastiveGSNNExplainer (which operates on the ODE rollout):
    build two separate ODEWrappers (one per cell line) and route by the
    data_ptr of the *initial* state ``x0`` so the integrator's internal
    fresh state tensors never need re-routing."""
    ode_1 = ODEWrapper(args.model_1, args.data, args.dxdt_scale,
                       t=args.t, method=args.method, tol=args.tol)
    ode_2 = ODEWrapper(args.model_2, args.data, args.dxdt_scale,
                       t=args.t, method=args.method, tol=args.tol)
    from lincs_gsnn.models.HnetGSNN import CellLineRouter
    return CellLineRouter(ode_1, ode_2, x0_1, x0_2)


def _build_static_router(args, *pairs):
    """For ContrastiveIG/Occlusion (which operate on trajectory rollouts and
    use cat-batch and split-baseline patterns): a router around the bare
    GSNN clones. Pre-registers all (x1_ref, x2_ref) pairs the explainers
    will pass."""
    from lincs_gsnn.models.HnetGSNN import CellLineRouter
    if not pairs:
        raise ValueError("_build_static_router requires at least one (x1, x2) pair")
    x1_first, x2_first = pairs[0]
    router = CellLineRouter(args.model_1, args.model_2, x1_first, x2_first)
    for x1_ref, x2_ref in pairs[1:]:
        router.register_pair(x1_ref, x2_ref)
    return router


def run_contrastive_explanation(args, x1, x2, xt_hat_w_inputs_1, xt_hat_w_inputs_2, target_ix):

    # Pre-move inputs to device so the explainers' internal .to(device) is a
    # no-op (preserves data_ptr-based dispatch in CellLineRouter).
    x0_1 = x1[[0]].to(args.device)
    x0_2 = x2[[0]].to(args.device)
    xt_hat_w_inputs_1 = xt_hat_w_inputs_1.to(args.device)
    xt_hat_w_inputs_2 = xt_hat_w_inputs_2.to(args.device)

    # Per-side model kwargs for node_activity-enabled models.  Each tensor
    # has leading dim 1 so the explainers' slice/repeat helpers broadcast
    # cleanly across whatever batch shape they build internally.
    mk1 = {'x_fn': args.x_fn_1} if args.x_fn_1 is not None else None
    mk2 = {'x_fn': args.x_fn_2} if args.x_fn_2 is not None else None

    ####################################################################################################
    # GSNN Explainer
    ####################################################################################################
    if args.use_hypernetwork:
        ode_gsnn = _build_ode_router(args, x0_1, x0_2)
    else:
        ode_gsnn = ODEWrapper(args.model,
                            args.data,
                            args.dxdt_scale,
                            t=args.t,
                            method=args.method,
                            tol=args.tol)

    explainer = ContrastiveGSNNExplainer(ode_gsnn, 
                                        args.data,
                                        hard=args.hard,
                                        beta=args.beta,
                                        lr=args.lr,
                                        prior=args.prior,
                                        iters=args.iters,
                                        free_edges=args.free_edges)


    cres_gsnn = explainer.explain(x0_1,
                                  x0_2,
                                  target_idx=args.target_gene_output_ix, 
                                  target=args.explanation_target,
                                  model_kwargs1=mk1,
                                  model_kwargs2=mk2)

    ####################################################################################################
    # IG Explainer
    ####################################################################################################
    if args.use_hypernetwork:
        static_model = _build_static_router(
            args,
            (xt_hat_w_inputs_1, xt_hat_w_inputs_2),
            (x0_1, x0_2),
        )
    else:
        static_model = args.model

    explainer = ContrastiveIGExplainer(static_model, args.data)

    cres_ig = explainer.explain(xt_hat_w_inputs_1,
                                xt_hat_w_inputs_2,
                                target_idx=args.target_gene_output_ix, 
                                element_mask=cres_gsnn.score.values > 0.5,
                                target=args.explanation_target,
                                model_kwargs1=mk1,
                                model_kwargs2=mk2)

    ####################################################################################################
    # Occlusion Explainer
    ####################################################################################################
    explainer = ContrastiveOcclusionExplainer(static_model, args.data)

    cres_occ = explainer.explain(xt_hat_w_inputs_1,
                                xt_hat_w_inputs_2,
                                target_idx=args.target_gene_output_ix, 
                                element_mask=cres_gsnn.score.values > 0.5,
                                target=args.explanation_target,
                                model_kwargs1=mk1,
                                model_kwargs2=mk2)

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
    # Use cell-line-specific models for trajectory rollout when in
    # hypernetwork mode; otherwise fall back to the single legacy model.
    # When node_activity is enabled, the (single) model is conditioned on
    # the cell line via x_fn instead.
    xt_hat_w_inputs_1, xt_hat_1 = predict_xt(args, x1,
                                              model=args.model_1 if args.use_hypernetwork else None,
                                              x_fn=args.x_fn_1)
    xt_hat_w_inputs_2, xt_hat_2 = predict_xt(args, x2,
                                              model=args.model_2 if args.use_hypernetwork else None,
                                              x_fn=args.x_fn_2)

    cres = run_contrastive_explanation(args, x1, x2, xt_hat_w_inputs_1, xt_hat_w_inputs_2, args.target_gene_output_ix)

    # Save contrastive results as CSV
    cres.to_csv(os.path.join(args.out, f'contrastive_results_{args.model_id}.csv'), index=False)

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
    torch.save(out_dict, os.path.join(args.out, f'contrastive_results_{args.model_id}.pt'))

    print(f'Contrastive explanation results saved to {args.out}')
    print('--'*40)
    print() 






    