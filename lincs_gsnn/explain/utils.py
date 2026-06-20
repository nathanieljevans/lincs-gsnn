import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from lincs_gsnn.data.DXDTDataset import DXDTDataset
from lincs_gsnn.models.ODEWrapper import ODEWrapper
from lincs_gsnn.proc.model_paths import gsnn_model_path
from lincs_gsnn.proc.node_activity import load_node_activity_artifact

def generate_masked_trajectories(root_gsnn, root_traj, model_id, cell, drug, dose, 
                        target_gene, remove_name=None, t=torch.linspace(0, 72, 100), method='euler', tol=None,
                        model_path=None): 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pred_dir = os.path.join(root_traj, 'predict_grid')

    # load data 
    data = torch.load(os.path.join(root_gsnn, 'bionetwork/bionetwork.pt'), weights_only=False)
    resolved_model_path = gsnn_model_path(root_gsnn, model_id, model_path=model_path)
    model = torch.load(resolved_model_path, weights_only=False, map_location=device)
    dxdt_scale = torch.load(os.path.join(root_gsnn, f'pretrain/dxdt_scale_{model_id}.pt'), weights_only=False).item()
    x_names = pd.read_csv(os.path.join(root_traj, 'predict_grid/gene_names.csv'))['gene_names'].values.astype(str).tolist()
    dxdt_meta = pd.read_csv(os.path.join(root_traj, 'predict_grid/dxdt_meta.csv'))

    # freeze the model 
    for param in model.parameters():
        param.requires_grad = False
    
    # set model to evaluation mode 
    model = model.eval() 

    # When the loaded model was trained with node_activity=True, every forward
    # call must include x_fn=<per-cell function-node activity>. Load the
    # artifact once and look up the cell-line tensor so we can pass it to
    # ODEWrapper for both the unmasked and edge-masked forward calls.
    x_fn_1 = None
    if getattr(model, 'node_activity', False):
        na_path = os.path.join(root_gsnn, 'bionetwork/node_activity.pt')
        na_payload = load_node_activity_artifact(na_path, node_names_dict=data.node_names_dict)
        x_fn_by_ciname = na_payload['x_fn_by_ciname']
        if cell not in x_fn_by_ciname:
            raise KeyError(
                f"node_activity artifact missing cell_iname={cell!r}; "
                f"available (sample): {sorted(x_fn_by_ciname)[:5]}..."
            )
        x_fn_1 = x_fn_by_ciname[cell].unsqueeze(0).to(device)

    t = t.to(device)

    target_ix = data.node_names_dict['output'].index(target_gene)       #

    assert drug in dxdt_meta.pert_id.unique(), f'{drug} not found in dxdt_meta'
    assert cell in dxdt_meta.cell_iname.unique(), f'{cell} not found in dxdt_meta'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cond1 = dxdt_meta[(dxdt_meta['pert_id'] == drug) 
                        & (dxdt_meta['cell_iname'] == cell) 
                        & (dxdt_meta['dose'] == dose)].reset_index(drop=True)

    dataset1 = DXDTDataset(meta=cond1, 
                      input_names=data.node_names_dict['input'], 
                      output_names=data.node_names_dict['output'],
                      src_names=x_names,
                      pred_dir=pred_dir,
                      scale=dxdt_scale)

    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=False)

    def retrieve_data(dataloader):

        xs = []; ys = [] 
        for i, (x, y) in enumerate(dataloader):
            xs.append(x)
            ys.append(y)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)

        return xs, ys 

    x1, y1 = retrieve_data(dataloader1)

    x0_1 = x1[[0]] 

    ode_gsnn = ODEWrapper(model, 
                        data, 
                        dxdt_scale, 
                        t=t, 
                        method=method, 
                        tol=tol)

    x1_t_hat = ode_gsnn(x0_1.to(device), x_fn=x_fn_1)

    x1_t_hat_input = x1[0].clone().unsqueeze(0).expand(x1_t_hat.shape[0], -1,).clone()

    x1_t_hat_input[:, :x1_t_hat.shape[1]] = x1_t_hat

    edge_mask_ = torch.ones(model.edge_index.shape[1], device=device) 

    if (remove_name is not None): 
        src, dst = model.edge_index.detach().cpu()
        remove_ix = model.homo_names.index(remove_name)

        edge_mask_[src == remove_ix] = 0 
        edge_mask_[dst == remove_ix] = 0 

        # confirm we removed the right edges 
        homo_names = np.array(model.homo_names) 
        src_name = homo_names[src]
        dst_name = homo_names[dst]

        remove_src = src_name[edge_mask_.cpu() == 0]
        remove_dst = dst_name[edge_mask_.cpu() == 0]

        assert all([(a == remove_name) or (b == remove_name) for a, b in zip(remove_src, remove_dst)]), 'we removed the wrong edges'

        # generate trajectories with edge mask. Pass x_fn so node_activity-trained
        # models receive the per-cell function-node activity on every forward call.
        x1_t_hat_nm = ode_gsnn(x0_1.to(device), edge_mask=edge_mask_, x_fn=x_fn_1)

        return x1_t_hat[:, target_ix], x1_t_hat_nm[:, target_ix]
    
    else: 
        return x1_t_hat[:, target_ix], None



def predict_node_activity(root_gsnn, root_traj, model_id, cell, model_path=None):
    """Compute per-function-node activity gate scores for one cell line.

    Loads the GSNN trained with ``node_activity=True``, looks up the per-cell
    function-node feature tensor ``x_fn`` from ``node_activity.pt``, runs it
    through the model's ``NodeActivity`` gate, and returns the resulting
    sigmoid gate value per function node. The returned score is in ``[0, 1]``:
    values near 1 indicate the model lets signal flow through that function
    node for this cell line; values near 0 indicate the node is gated off.

    Parameters
    ----------
    root_gsnn : str
        Root containing ``bionetwork/bionetwork.pt``,
        ``bionetwork/node_activity.pt`` and the GSNN checkpoint
        (``train/trained_model_*`` or ``pretrain/pretrained_model_*``).
    root_traj : str
        Unused. Kept for signature parity with the other helpers in this
        module.
    model_id : str
        Model replicate id (e.g. ``model_0``).
    cell : str
        ``cell_iname`` to score. Must be present in the node_activity
        artifact's per-cell lookup.

    Returns
    -------
    pandas.DataFrame
        Columns: ``node_name`` (bionet function-node name, e.g.
        ``PROTEIN__TP53``) and ``node_activity_score`` (sigmoid gate value
        in ``[0, 1]``), one row per function node in bionet order.
    """
    del root_traj  # not needed; artifact lives under root_gsnn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load(
        os.path.join(root_gsnn, 'bionetwork/bionetwork.pt'),
        weights_only=False,
    )
    model = torch.load(
        gsnn_model_path(root_gsnn, model_id, model_path=model_path),
        weights_only=False,
        map_location=device,
    )

    na_module = getattr(model, 'node_activity_model', None)
    if not getattr(model, 'node_activity', False) or na_module is None:
        raise ValueError(
            "predict_node_activity requires a GSNN trained with "
            "node_activity=True (model.node_activity_model must be present)."
        )

    na_path = os.path.join(root_gsnn, 'bionetwork/node_activity.pt')
    na_payload = load_node_activity_artifact(na_path, node_names_dict=data.node_names_dict)
    x_fn_by_ciname = na_payload['x_fn_by_ciname']
    if cell not in x_fn_by_ciname:
        raise KeyError(
            f"node_activity artifact missing cell_iname={cell!r}; "
            f"available (sample): {sorted(x_fn_by_ciname)[:5]}..."
        )

    # Per-cell function-node features: shape (Nf, activity_dim). Kept here
    # before adding the batch dim so we can also emit it column-wise below.
    x_fn_cell = torch.as_tensor(x_fn_by_ciname[cell], dtype=torch.float32)
    if x_fn_cell.dim() == 1:
        x_fn_cell = x_fn_cell.unsqueeze(-1)  # (Nf, 1) when activity_dim == 1

    # (1, Nf, activity_dim) — batch dim of 1; NodeActivity's MLP is shared
    # across nodes and outputs a per-node logit that we sigmoid into a gate
    # value (with the same temperature scaling as NodeActivity.forward). We
    # call the MLP directly rather than NodeActivity.forward to avoid the
    # side-effects (store_alpha_mean is for the training-time sparsity
    # penalty, and the forward output is expanded across channels).
    x_fn = x_fn_cell.unsqueeze(0).to(device)

    MODE = na_module.mode

    model = model.eval()
    with torch.no_grad():
        alpha = (na_module.mlp(x_fn) / na_module.temperature).sigmoid()

    alpha = alpha.squeeze(0) # (Nf, 1)

    function_names = list(data.node_names_dict['function'])
    alpha_np = alpha.detach().cpu().numpy()

    if alpha_np.shape[0] != len(function_names):
        raise RuntimeError(
            f"NodeActivity alpha length {alpha_np.shape[0]} does not match the "
            f"bionet function-node count {len(function_names)}."
        )

    # Channel names for the per-feature columns. Falls back to positional
    # f"feature_{i}" names if the artifact predates `activity_features`.
    activity_features = list(na_payload.get('activity_features') or [])
    if len(activity_features) != x_fn_cell.shape[-1]:
        activity_features = [f'feature_{i}' for i in range(x_fn_cell.shape[-1])]

    x_fn_np = x_fn_cell.detach().cpu().numpy()  # (Nf, activity_dim)

    df_cols = {
        'node_name': function_names,
        'mode': MODE,
    }

    for i, alpha_i in enumerate(alpha_np.T):
        df_cols[f'node_activity_score_{i}'] = alpha_i

    for i, fname in enumerate(activity_features):
        df_cols[fname] = x_fn_np[:, i]

    node_activity_df = pd.DataFrame(df_cols)

    return node_activity_df  # columns: node_name, node_activity_score, <activity_features...>
