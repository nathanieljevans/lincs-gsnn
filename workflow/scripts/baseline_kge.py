'''
Knowledge-graph embedding baseline (ComplEx2) for function->function edge prediction.

When ``--removed_edges`` is provided, validation uses the same held-out val edges
as the MEI workflow (``split == 'val'`` in removed_edges.csv). Training uses all
edges present in bionetwork.pt (train edges only; val/test are already removed).
'''

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='KGE (ComplEx2) edge prediction baseline')
    parser.add_argument('--bionet', type=str, required=True,
                        help='Directory containing bionetwork.pt')
    parser.add_argument('--removed_edges', type=str, default=None,
                        help='Path to removed_edges.csv; val split used for validation')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory (writes predictions.csv)')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--val_frac', type=float, default=0.05,
                        help='Legacy random train split when --removed_edges is omitted')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def _load_val_triples(removed_edges_path, node_names):
    '''Load held-out val edges (same split as MEI evaluation).'''
    holdout = pd.read_csv(removed_edges_path, low_memory=False)
    if 'split' not in holdout.columns:
        raise ValueError(f'{removed_edges_path} has no split column; cannot load MEI val edges')

    val = holdout.loc[holdout['split'].eq('val')].copy()
    if val.empty:
        raise ValueError(f'No val edges found in {removed_edges_path}')

    if {'src_idx', 'dst_idx'}.issubset(val.columns):
        heads = torch.tensor(val['src_idx'].astype(int).values, dtype=torch.long)
        tails = torch.tensor(val['dst_idx'].astype(int).values, dtype=torch.long)
    else:
        name_to_idx = {n: i for i, n in enumerate(node_names)}
        src_col = 'src_name' if 'src_name' in val.columns else 'source'
        dst_col = 'dst_name' if 'dst_name' in val.columns else 'target'
        heads = torch.tensor([name_to_idx[s] for s in val[src_col].astype(str)], dtype=torch.long)
        tails = torch.tensor([name_to_idx[t] for t in val[dst_col].astype(str)], dtype=torch.long)

    relations = torch.zeros(len(heads), dtype=torch.long)
    return heads, tails, relations


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        from complex2.models.ComplEx2 import ComplEx2
        from complex2.data.TriplesDataset import TriplesDataset
    except ImportError as exc:
        raise ImportError(
            'baseline_kge requires the complex2 package. '
            'Install it or set baselines.kge.enabled=false in the workflow config.'
        ) from exc

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = torch.load(os.path.join(args.bionet, 'bionetwork.pt'), weights_only=False)
    edge_index = data.edge_index_dict['function', 'to', 'function'].long()
    n_nodes = len(data.node_names_dict['function'])
    node_names = np.array(data.node_names_dict['function'])

    val_heads = val_tails = val_relations = None
    if args.removed_edges:
        val_heads, val_tails, val_relations = _load_val_triples(args.removed_edges, node_names)
        train_heads, train_tails = edge_index[0], edge_index[1]
        print(
            f'KGE train/val split from removed_edges: '
            f'train={train_heads.shape[0]} val={val_heads.shape[0]}',
            flush=True,
        )
    else:
        val_mask = torch.randint(0, 100, (edge_index.shape[1],)) > int(100 * (1 - args.val_frac))
        train_mask = ~val_mask
        train_heads, train_tails = edge_index[:, train_mask]
        print(
            f'KGE legacy random train split: train={train_heads.shape[0]} '
            f'(held out {int(val_mask.sum())} unused edges)',
            flush=True,
        )

    train_relations = torch.zeros(train_heads.shape[0], dtype=torch.long)
    train_dataset = TriplesDataset({
        'head': train_heads, 'tail': train_tails, 'relation': train_relations,
    })
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    data2 = pyg.data.Data()
    data2.edge_index_dict = {('function', 'to', 'function'): edge_index}
    data2['num_nodes_dict'] = {'function': n_nodes}
    data2['edge_reltype'] = {('function', 'to', 'function'): np.array(0)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ComplEx2(data2, hidden_channels=args.hidden_channels).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_metric = float('inf')
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for h, t, r in train_loader:
            h, t, r = h.to(device), t.to(device), r.to(device)
            optim.zero_grad()
            loss = -model(h, r, t).mean()
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        val_loss = None
        if val_heads is not None:
            model.eval()
            with torch.no_grad():
                vh = val_heads.to(device)
                vt = val_tails.to(device)
                vr = val_relations.to(device)
                val_loss = float(-model(vh, vr, vt).mean().item())

        metric = val_loss if val_loss is not None else train_loss
        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 100 == 0:
            if val_loss is not None:
                print(
                    f'epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}',
                    flush=True,
                )
            else:
                print(f'epoch {epoch}: train_loss={train_loss:.4f}', flush=True)

    model.load_state_dict(best_state)
    model.eval()

    heads, tails = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                heads.append(i)
                tails.append(j)
    heads = torch.tensor(heads, dtype=torch.long)
    tails = torch.tensor(tails, dtype=torch.long)
    relations = torch.zeros(len(heads), dtype=torch.long)
    dataset = TriplesDataset({'head': heads, 'tail': tails, 'relation': relations})
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for h, t, r in loader:
            logprobs = model(h.to(device), r.to(device), t.to(device))
            predictions.append(logprobs.detach().cpu().numpy())
    predictions = np.concatenate(predictions)

    res = pd.DataFrame({
        'source': node_names[heads.numpy()],
        'target': node_names[tails.numpy()],
        'score': predictions,
    })
    out_path = os.path.join(args.out, 'predictions.csv')
    res.to_csv(out_path, index=False)
    print(f'Wrote {len(res)} predictions to {out_path}', flush=True)


if __name__ == '__main__':
    main()
