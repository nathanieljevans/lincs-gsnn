'''
Non-contrastive explanation for a single (target_gene, cell_line, sample) condition.

Mirrors `contrastive_explanation.py` but operates on a single cell line: it
runs the three non-contrastive explainers from `gsnn.interpret` (GSNNExplainer,
IGExplainer, OcclusionExplainer) and merges their per-edge / per-node scores
into a single result frame.

Target Gene  - The gene of interest that is being explained
Cell line    - The single cell line the explanation is computed for
Sample id    - The sample id that is being explained
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
from gsnn.interpret.GSNNExplainer import GSNNExplainer
from gsnn.interpret.IGExplainer import IGExplainer
from gsnn.interpret.OcclusionExplainer import OcclusionExplainer



def get_args():
    parser = argparse.ArgumentParser(description='Generate non-contrastive explanation for a given target gene, cell line, and sample id')

    parser.add_argument('--root_gsnn', type=str, required=True,
                       help='Root directory of the gsnn outputs')
    parser.add_argument('--root_traj', type=str, required=True,
                       help='Root directory of the traj outputs')
    parser.add_argument('--model_id', type=str, required=True,
                       help='Model replicate id (e.g. model_0)')

    parser.add_argument('--target_gene', type=str, required=True,
                       help='Target gene of interest')
    parser.add_argument('--cell_line', type=str, required=True,
                       help='Cell line to compute the explanation for')
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
                            'cell-line-specific GSNN. Falls back to the legacy '
                            'single-model behavior if the artifact is missing.')

    parser.add_argument('--node_activity_path', type=str, default=None,
                       help='Path to the node_activity.pt artifact. Required when the '
                            'loaded model has node_activity enabled; defaults to '
                            '<root_gsnn>/bionetwork/node_activity.pt.')
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

    args.cond = args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)]

    # Optionally swap args.model for the cell-line-specific GSNN materialized
    # from the hypernetwork artifact. We do this BEFORE freezing/.to(device)
    # below so the rest of the script remains identical to the legacy path.
    args.hnet_artifact_path = os.path.join(args.root_gsnn, f'pretrain/pretrained_hnet_{args.model_id}.pt')
    args.use_hypernetwork = bool(args.use_hypernetwork) and os.path.exists(args.hnet_artifact_path)
    if args.use_hypernetwork:
        from lincs_gsnn.models.HnetGSNN import (
            cell_onehot,
            load_hnet_artifact,
            materialize_gsnn,
        )
        loaded = load_hnet_artifact(args.hnet_artifact_path, args.data)
        hnet = loaded['hnet'].eval()
        cell_lines = loaded['cell_lines']
        C = cell_onehot(args.cell_line, cell_lines)
        with torch.no_grad():
            args.model = materialize_gsnn(hnet, C)
        print(f'Hypernetwork mode: materialized cell-line-specific GSNN for {args.cell_line}.')

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
    # Optional per-function-node activity feature.  If the loaded GSNN was
    # trained with `node_activity=True`, every forward call must include
    # `x_fn=...`; here we look it up once for this cell line and stash it
    # on args so `predict_xt` and the explainers can broadcast it across
    # samples / IG path / occlusion grid.  When node_activity is disabled
    # `args.x_fn` is None and downstream code paths are untouched.
    # ------------------------------------------------------------------
    args.x_fn = None
    if getattr(args.model, 'node_activity', False):
        na_path = args.node_activity_path or os.path.join(args.root_gsnn, 'bionetwork/node_activity.pt')
        if not os.path.exists(na_path):
            raise FileNotFoundError(
                f"Loaded model has node_activity=True but artifact not found at {na_path}. "
                "Pass --node_activity_path explicitly or rebuild with `make_bio_network.py --node_activity`."
            )
        na_payload = load_node_activity_artifact(na_path, node_names_dict=args.data.node_names_dict)
        if args.cell_line not in na_payload['x_fn_by_ciname']:
            raise KeyError(
                f"node_activity artifact does not contain cell_iname={args.cell_line!r}. "
                f"Available: {sorted(na_payload['x_fn_by_ciname'])[:5]}..."
            )
        # Shape (1, Nf, activity_dim); broadcasts across all explainer calls.
        args.x_fn = na_payload['x_fn_by_ciname'][args.cell_line].unsqueeze(0).to(args.device)
        print(f'node_activity: loaded x_fn for {args.cell_line} (shape={tuple(args.x_fn.shape)}) from {na_path}')

    args.target_gene = 'GENE__' + args.target_gene

    try:
        args.target_gene_output_ix = args.data.node_names_dict['output'].index(args.target_gene)
        args.target_gene_input_ix = args.data.node_names_dict['input'].index(args.target_gene)
    except ValueError:
        raise ValueError(f'Target gene {args.target_gene} not found in the data')

    assert args.cell_line in args.dxdt_meta.cell_iname.unique(), f'Cell line {args.cell_line} not found in the data'
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

    dataset = DXDTDataset(meta=args.dxdt_meta[(args.dxdt_meta['cell_iname'] == args.cell_line) & (args.dxdt_meta['pert_id'] == args.pert_id) & (args.dxdt_meta['dose'] == args.dose)],
                      input_names=args.data.node_names_dict['input'],
                      output_names=args.data.node_names_dict['output'],
                      src_names=args.x_names,
                      pred_dir=args.pred_dir,
                      scale=args.dxdt_scale)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    return retrieve_data(dataloader)


def predict_xt(args, x):

    x0 = x[[0]]

    ode_gsnn = ODEWrapper(args.model,
                        args.data,
                        args.dxdt_scale,
                        t=args.t,
                        method= args.method,
                        tol=args.tol)

    # When node_activity is enabled, the ODE func needs x_fn at every
    # integration step. ODEWrapper forwards the kwarg to ODEFunc.set_x_fn.
    if args.x_fn is not None:
        xt_hat = ode_gsnn(x0.to(args.device), x_fn=args.x_fn)
    else:
        xt_hat = ode_gsnn(x0.to(args.device))

    # Use x[0] to get shape (N_features,) then expand to (T, N_features)
    xt_hat_input = x[0].clone().unsqueeze(0).expand(xt_hat.shape[0], -1,).clone()
    xt_hat_input[:, :xt_hat.shape[1]] = xt_hat

    return xt_hat_input, xt_hat


def run_non_contrastive_explanation(args, x, xt_hat_w_inputs, target_ix):

    # When node_activity is enabled, every explainer needs to know the
    # per-cell-line x_fn so its internal model() calls match training. We
    # pass a leading-dim-1 tensor so the explainers' slice/repeat helpers
    # broadcast it across whatever batch shape they construct.
    model_kwargs = {'x_fn': args.x_fn} if args.x_fn is not None else None

    ####################################################################################################
    # GSNN Explainer
    ####################################################################################################

    explainer = GSNNExplainer(args.model.to(args.device),
                              args.data,
                              hard=args.hard,
                              beta=args.beta,
                              lr=args.lr,
                              prior=args.prior,
                              iters=args.iters,
                              free_edges=args.free_edges,
                              grad_norm_clip=args.grad_norm_clip,
                              entropy=args.entropy)

    x0 = x[[0]]
    cres_gsnn = explainer.explain(x0.to(args.device),
                                  target_idx=args.target_gene_output_ix,
                                  target=args.explanation_target,
                                  model_kwargs=model_kwargs)

    ####################################################################################################
    # IG Explainer
    ####################################################################################################

    explainer = IGExplainer(args.model.to(args.device), args.data)

    cres_ig = explainer.explain(xt_hat_w_inputs.to(args.device),
                                target_idx=args.target_gene_output_ix,
                                element_mask=cres_gsnn.score.values > 0.5,
                                target=args.explanation_target,
                                model_kwargs=model_kwargs)

    ####################################################################################################
    # Occlusion Explainer
    ####################################################################################################
    explainer = OcclusionExplainer(args.model.to(args.device), args.data)

    cres_occ = explainer.explain(xt_hat_w_inputs.to(args.device),
                                 target_idx=args.target_gene_output_ix,
                                 element_mask=cres_gsnn.score.values > 0.5,
                                 target=args.explanation_target,
                                 model_kwargs=model_kwargs)

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

    x = get_x(args)
    xt_hat_w_inputs, xt_hat = predict_xt(args, x)

    cres = run_non_contrastive_explanation(args, x, xt_hat_w_inputs, args.target_gene_output_ix)

    # Save non-contrastive results as CSV
    cres.to_csv(os.path.join(args.out, f'non_contrastive_results_{args.model_id}.csv'), index=False)

    # Save out_dict with cres, trajectory predictions and metadata
    out_dict = {
        'cres': cres,
        'xt_hat': xt_hat.detach().cpu().numpy(),
        'x_observed': x.detach().cpu().numpy(),
        'target_gene': args.target_gene,
        'target_gene_output_ix': args.target_gene_output_ix,
        'cell_line': args.cell_line,
        'pert_id': args.pert_id,
        'dose': args.dose,
        'horizon': args.horizon,
        'n_time_pts': args.n_time_pts,
        't': args.t.cpu().numpy(),
    }
    torch.save(out_dict, os.path.join(args.out, f'non_contrastive_results_{args.model_id}.pt'))

    print(f'Non-contrastive explanation results saved to {args.out}')
    print('--'*40)
    print()
