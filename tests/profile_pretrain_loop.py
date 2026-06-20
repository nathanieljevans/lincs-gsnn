#!/usr/bin/env python3
"""Profile the pretrain_gsnn_with_dxdt training loop (exp_37 defaults).

Measures wall time per training-step section (dataloader, H2D, forward, loss,
backward, grad clip, optimizer, metrics) and optionally runs cProfile / the
PyTorch CUDA profiler on a short warmup + profile window.

Usage (defaults point at exp_37 artifacts on gscratch)::

    python tests/profile_pretrain_loop.py
    python tests/profile_pretrain_loop.py --batches 20 --num-workers 0
    python tests/profile_pretrain_loop.py --torch-profiler  # CUDA only
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import os
import pstats
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
PRETRAIN_SCRIPT = REPO / "workflow" / "scripts" / "pretrain_gsnn_with_dxdt.py"

DEFAULT_PREDS = (
    "/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/"
    "lincs-traj/runs/tune/tune_37/output/predict_grid"
)
DEFAULT_BIONET = (
    "/home/exacloud/gscratch/mcweeney_lab/evans/lincs-modeling/outputs/"
    "lincs-gsnn/exp_37/bionetwork"
)


def _load_pretrain_module():
    spec = importlib.util.spec_from_file_location("pretrain_gsnn_with_dxdt", PRETRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PRETRAIN_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class SectionStats:
    total_s: float = 0.0
    n: int = 0

    def add(self, dt: float) -> None:
        self.total_s += dt
        self.n += 1

    @property
    def mean_s(self) -> float:
        return self.total_s / max(self.n, 1)


@dataclass
class ProfileRun:
    sections: dict[str, SectionStats] = field(default_factory=lambda: defaultdict(SectionStats))
    batch_wall_s: list[float] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default=DEFAULT_PREDS, help="predict_grid root")
    p.add_argument("--bionet", type=str, default=DEFAULT_BIONET, help="bionetwork directory")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3, help="untracked warmup batches")
    p.add_argument("--batches", type=int, default=15, help="profiled batches after warmup")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", type=str, default="muon", choices=["adam", "adamw", "muon", "rmsprop"])
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", action="store_false", dest="tf32")
    p.add_argument("--objective", type=str, default="mse", choices=["nll", "mse"])
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--cprofile", action="store_true", help="run cProfile on profiled batches")
    p.add_argument("--torch-profiler", action="store_true", help="CUDA torch.profiler on last batch")
    # exp_37 BIOGSNN architecture
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--norm", type=str, default="none")
    p.add_argument("--dxdt-nonlin", type=str, default="leaky_relu")
    p.add_argument("--init-rna-half-life", type=float, default=9.0)
    p.add_argument("--gamma-prior-weight", type=float, default=1e-2)
    return p.parse_args()


def _build_pretrain_args(cli: argparse.Namespace, pretrain_mod) -> argparse.Namespace:
    """Namespace compatible with pretrain helpers."""
    return argparse.Namespace(
        data=cli.data,
        bionet=cli.bionet,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        seed=cli.seed,
        sigma_floor=1e-4,
        objective=cli.objective,
        clip_grad_norm=cli.clip_grad_norm,
        optimizer=cli.optimizer,
        amp=cli.amp,
        tf32=cli.tf32,
        lr=1e-2,
        wd=1e-6,
        channels=cli.channels,
        layers=cli.layers,
        share_layers=False,
        dropout=cli.dropout,
        norm=cli.norm,
        checkpoint=False,
        init="degree_normalized",
        add_function_self_edges=True,
        bias=True,
        residual=True,
        node_mlp=False,
        node_mlp_dim=16,
        node_attn=False,
        attn_mlp_hidden=16,
        node_activity=False,
        alpha_decay=1e-2,
        BIOGSNN=True,
        GSNN=False,
        dxdt_nonlin=cli.dxdt_nonlin,
        init_rna_half_life=cli.init_rna_half_life,
        gamma_prior_weight=cli.gamma_prior_weight,
        gene_norm_path=os.path.join(cli.bionet, "gene_norm.pt"),
        val_cells_per_drug=1,
    )


def _setup_training(cli: argparse.Namespace):
    from gsnn.models.GSNN import GSNN
    from lincs_gsnn.data.DXDTDataset import DXDTDataset
    from lincs_gsnn.models.BIOGSNN import BIOGSNN
    from lincs_gsnn.proc.cell_drug_split import build_cell_drug_split, filter_meta_by_partition
    from lincs_gsnn.proc.drug_accessibility import accessible_indices, get_or_compute_drug_accessible_mask
    from lincs_gsnn.proc.gene_norm import load_gene_norm_artifact
    from lincs_gsnn.train.console import configure_cuda_performance
    from lincs_gsnn.train.optim import build_optimizer
    from lincs_gsnn.utils.GaussianNLL import GaussianNLL

    pretrain_mod = _load_pretrain_module()

    if not os.path.isfile(os.path.join(cli.data, "dxdt_meta.csv")):
        raise FileNotFoundError(f"predict_grid missing under {cli.data}")
    if not os.path.isfile(os.path.join(cli.bionet, "bionetwork.pt")):
        raise FileNotFoundError(f"bionetwork.pt missing under {cli.bionet}")

    args = _build_pretrain_args(cli, pretrain_mod)
    dxdt_meta = pd.read_csv(os.path.join(cli.data, "dxdt_meta.csv"))
    src_gene_names = pd.read_csv(os.path.join(cli.data, "gene_names.csv"))["gene_names"].tolist()
    data = torch.load(os.path.join(cli.bionet, "bionetwork.pt"), weights_only=False)
    gene_norm = load_gene_norm_artifact(args.gene_norm_path, output_names=data.node_names_dict["output"])

    pert_ids_net = [x.split("__")[1] for x in data.node_names_dict["input"] if "DRUG__" in x]
    dxdt_meta = dxdt_meta[dxdt_meta["pert_id"].isin(pert_ids_net)]
    split_df = build_cell_drug_split(
        pert_ids=sorted(dxdt_meta["pert_id"].unique().tolist()),
        cell_inames=sorted(dxdt_meta["cell_iname"].unique().tolist()),
        n_val=1,
        seed=cli.seed,
    )
    train_meta = filter_meta_by_partition(dxdt_meta, split_df, "train")

    scale_ds = DXDTDataset(
        train_meta,
        input_names=data.node_names_dict["input"],
        output_names=data.node_names_dict["output"],
        src_names=src_gene_names,
        pred_dir=cli.data,
        sigma_floor=args.sigma_floor,
    )
    dxdt_scale = scale_ds._scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    configure_cuda_performance(args.tf32 and device.type == "cuda")

    accessible_out_ix = accessible_indices(get_or_compute_drug_accessible_mask(data)).to(device)
    gsnn_kwargs = pretrain_mod._gsnn_kwargs_from_args(args, data)
    model = BIOGSNN(
        gsnn_kwargs=gsnn_kwargs,
        gene_norm=gene_norm,
        dxdt_nonlin=args.dxdt_nonlin,
        init_rna_half_life=args.init_rna_half_life,
        dxdt_scale=dxdt_scale if args.init_rna_half_life is not None else None,
    ).to(device)
    model.train()

    train_loader = DataLoader(
        DXDTDataset(
            train_meta,
            input_names=data.node_names_dict["input"],
            output_names=data.node_names_dict["output"],
            src_names=src_gene_names,
            pred_dir=cli.data,
            scale=dxdt_scale,
            sigma_floor=args.sigma_floor,
        ),
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=cli.num_workers,
        persistent_workers=(cli.num_workers > 0),
    )

    optim = build_optimizer(model, args.optimizer, lr=args.lr, wd=args.wd)
    crit = GaussianNLL()
    mse_crit = torch.nn.MSELoss(reduction="mean")
    alpha_decay, na_module, use_alpha_decay = pretrain_mod._node_activity_penalty_args(args, model)
    gamma_prior_weight, use_gamma_prior = pretrain_mod._gamma_prior_penalty_args(args, model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "args": args,
        "device": device,
        "use_amp": use_amp,
        "model": model,
        "optim": optim,
        "crit": crit,
        "mse_crit": mse_crit,
        "train_loader": train_loader,
        "accessible_out_ix": accessible_out_ix,
        "alpha_decay": alpha_decay,
        "na_module": na_module,
        "use_alpha_decay": use_alpha_decay,
        "gamma_prior_weight": gamma_prior_weight,
        "use_gamma_prior": use_gamma_prior,
        "n_train_rows": len(train_meta),
        "n_params": n_params,
        "n_fn_nodes": len(data.node_names_dict["function"]),
        "n_out": len(data.node_names_dict["output"]),
        "n_acc_out": int(accessible_out_ix.numel()),
    }


def _train_step(ctx, batch, stats: ProfileRun | None) -> float:
    """One instrumented training step; returns batch wall time in seconds."""
    args = ctx["args"]
    device = ctx["device"]
    model = ctx["model"]
    optim = ctx["optim"]
    crit = ctx["crit"]
    mse_crit = ctx["mse_crit"]
    accessible_out_ix = ctx["accessible_out_ix"]
    use_amp = ctx["use_amp"]
    device_type = device.type

    t_batch = time.perf_counter()

    def record(name: str, t0: float) -> None:
        if stats is not None:
            stats.sections[name].add(time.perf_counter() - t0)

    if len(batch) == 3:
        X, dxdt_mu, dxdt_sigma = batch
        x_fn = None
    else:
        X, dxdt_mu, dxdt_sigma, x_fn = batch
        t0 = time.perf_counter()
        x_fn = x_fn.to(device, non_blocking=True)
        record("x_fn_h2d", t0)

    t0 = time.perf_counter()
    X = X.to(device, non_blocking=True)
    dxdt_mu = dxdt_mu.to(device, non_blocking=True)
    dxdt_sigma = dxdt_sigma.to(device, non_blocking=True)
    _sync(device)
    record("batch_h2d", t0)

    optim.zero_grad(set_to_none=True)

    t0 = time.perf_counter()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
        dxdt_hat = model(X) if x_fn is None else model(X, x_fn=x_fn)
    _sync(device)
    record("forward", t0)

    t0 = time.perf_counter()
    mu_sub = dxdt_mu[:, accessible_out_ix]
    sigma_sub = dxdt_sigma[:, accessible_out_ix]
    hat_sub = dxdt_hat[:, accessible_out_ix]
    nll_loss = crit(hat_sub, mu_sub, sigma_sub)
    mse_loss = mse_crit(hat_sub, mu_sub)
    loss = mse_loss if args.objective == "mse" else nll_loss
    if ctx["use_alpha_decay"]:
        loss = loss + ctx["alpha_decay"] * ctx["na_module"].get_alpha_mean().mean()
    if ctx["use_gamma_prior"]:
        loss = loss + ctx["gamma_prior_weight"] * model.gamma_prior_loss()
    _sync(device)
    record("loss", t0)

    t0 = time.perf_counter()
    loss.backward()
    _sync(device)
    record("backward", t0)

    t0 = time.perf_counter()
    if args.clip_grad_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    _sync(device)
    record("grad_clip", t0)

    t0 = time.perf_counter()
    optim.step()
    _sync(device)
    record("optim_step", t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        r2_score(
            mu_sub.detach().cpu().numpy(),
            hat_sub.detach().cpu().numpy(),
            multioutput="uniform_average",
        )
    record("metrics_r2", t0)

    dt = time.perf_counter() - t_batch
    if stats is not None:
        stats.batch_wall_s.append(dt)
    return dt


def _profile_loop(ctx, *, warmup: int, batches: int) -> ProfileRun:
    loader = ctx["train_loader"]
    it = iter(loader)
    stats = ProfileRun()

    for _ in range(warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _train_step(ctx, batch, stats=None)

    for _ in range(batches):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        stats.sections["dataloader"].add(time.perf_counter() - t0)
        _train_step(ctx, batch, stats)

    return stats


def _print_report(ctx, stats: ProfileRun, *, warmup: int, batches: int) -> None:
    total_tracked = sum(s.total_s for s in stats.sections.values())
    print("\n" + "=" * 72)
    print("PRETRAIN LOOP PROFILE")
    print("=" * 72)
    print(f"  device:           {ctx['device']}")
    print(f"  amp:              {ctx['use_amp']}")
    print(f"  optimizer:        {ctx['args'].optimizer}")
    print(f"  objective:        {ctx['args'].objective}")
    print(f"  batch_size:       {ctx['args'].batch_size}")
    print(f"  num_workers:      {ctx['train_loader'].num_workers}")
    print(f"  train rows:       {ctx['n_train_rows']}")
    print(f"  params:           {ctx['n_params']:,}")
    print(f"  function nodes:   {ctx['n_fn_nodes']:,}")
    print(f"  outputs (acc):    {ctx['n_acc_out']} / {ctx['n_out']}")
    print(f"  warmup / profile: {warmup} / {batches} batches")
    if stats.batch_wall_s:
        mean_batch = sum(stats.batch_wall_s) / len(stats.batch_wall_s)
        print(f"  mean batch wall:  {mean_batch:.3f}s  ({1.0 / mean_batch:.2f} batch/s)")

    rows = []
    for name, sec in sorted(stats.sections.items(), key=lambda kv: -kv[1].total_s):
        pct = 100.0 * sec.total_s / max(total_tracked, 1e-9)
        rows.append((name, sec.n, sec.mean_s, sec.total_s, pct))

    print("\n  Section            N      mean(s)    total(s)   share")
    print("  " + "-" * 58)
    for name, n, mean_s, total_s, pct in rows:
        print(f"  {name:<18} {n:5d}  {mean_s:9.4f}  {total_s:9.3f}  {pct:5.1f}%")
    print("  " + "-" * 58)
    print(f"  {'tracked_sum':<18} {'':5s}  {'':9s}  {total_tracked:9.3f}  100.0%")
    untracked = sum(stats.batch_wall_s) - total_tracked
    if stats.batch_wall_s:
        print(f"  {'untracked/overhead':<18} {'':5s}  {untracked / len(stats.batch_wall_s):9.4f}  {untracked:9.3f}")
    print("=" * 72)


def _run_cprofile(ctx, *, warmup: int, batches: int) -> None:
    prof = cProfile.Profile()

    def _target():
        loader = ctx["train_loader"]
        it = iter(loader)
        for _ in range(warmup + batches):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            _train_step(ctx, batch, stats=None)

    prof.runcall(_target)
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumtime").print_stats(30)
    print("\nCPROFILE (top 30 by cumtime)\n")
    print(stream.getvalue())


def _run_torch_profiler(ctx, *, warmup: int) -> None:
    if ctx["device"].type != "cuda":
        print("\nSkipping torch.profiler (CUDA not available).")
        return

    from torch.profiler import ProfilerActivity, profile, record_function

    loader = ctx["train_loader"]
    it = iter(loader)
    for _ in range(warmup):
        batch = next(it)

    batch = next(it)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("train_step"):
            _train_step(ctx, batch, stats=None)

    print("\nTORCH PROFILER (top CUDA kernels by self CUDA time)\n")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    print("\nTORCH PROFILER (top ops by self CPU time)\n")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


def main() -> None:
    cli = _parse_args()
    print("Building pretrain context...")
    ctx = _setup_training(cli)
    print(
        f"Context ready: {ctx['n_params']:,} params, "
        f"{len(ctx['train_loader'])} batches/epoch, device={ctx['device']}"
    )

    stats = _profile_loop(ctx, warmup=cli.warmup, batches=cli.batches)
    _print_report(ctx, stats, warmup=cli.warmup, batches=cli.batches)

    if cli.cprofile:
        _run_cprofile(ctx, warmup=cli.warmup, batches=cli.batches)
    if cli.torch_profiler:
        _run_torch_profiler(ctx, warmup=cli.warmup)


if __name__ == "__main__":
    main()
