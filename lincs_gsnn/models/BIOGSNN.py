'''BIOGSNN: a GSNN wrapper that imposes mass-action-style mRNA dynamics
on the output-gene layer.

    dx/dt = tf_activity(x) - gamma * level3(x_gene)

where

  * `tf_activity = GSNN(x_z)` is the raw GSNN output, interpreted as a
    net transcription contribution to dx/dt. It is NOT constrained to be
    non-negative: the wrapped GSNN can emit negative values, so the
    "production rate" framing is a loose interpretation rather than a
    hard guarantee. In practice this gives the network the freedom to
    represent net repressive effects without having to fight the
    degradation term.
  * `gamma = softplus(self.gamma)` is a learnable per-output-gene
    first-order mRNA degradation coefficient (non-negative by
    construction).
  * `level3 = relu(mu_g + sigma_g * x_z)` back-transforms the z-scored
    gene input to an abundance-like proxy in Level-3 expression units.
    Because `level3 >= 0` always, the term `- gamma * level3` itself is
    guaranteed non-positive; the sign of the full `dx/dt` depends on
    whether `tf_activity` overcomes it.

`mu_g, sigma_g` are the per-gene control-population statistics produced
by lincs-traj's `make_proc.py` and copied into `gene_norm.pt` by
`make_bio_network.py`; they ride along as registered buffers so saved
BIOGSNN checkpoints carry them inside their state_dict.

Note on units: `tf_activity` lives in scaled-dxdt units (i.e.
`d(z_g)/dt` divided by the global `dxdt_scale` used by `DXDTDataset`),
while `level3` lives in Level-3 (log1p) expression units. The learned
`gamma` therefore absorbs both the per-gene `1/sigma_g` and the global
`1/dxdt_scale` factor needed to reconcile the two sides. The physical
mRNA half-life for gene g is
`ln(2) * dxdt_scale / (gamma[g] * sigma_g[g])` in whatever time units
the trajectory grid uses (hours in the current lincs-traj pipeline);
gamma itself is NOT directly interpretable as a physical decay rate
without that conversion.

Optional derivative bound: pass ``dxdt_clip`` (>0) to softly saturate the net
``dx/dt`` at ``+/- dxdt_clip`` (in the model's native scaled-dxdt units) via a
``tanh``. Because it is set at construction and stored on the module, the bound
is identical at pretrain time and during the ``ODEFunc`` rollout, which keeps the
integrated dynamics from running away (and reduces ODE stiffness/step count).

Optional physical initialization: pass ``init_rna_half_life`` (hours) and
``dxdt_scale`` to seed ``gamma`` from a literature mRNA half-life prior and
register a frozen ``gamma_prior`` buffer for a soft log-rate L2 anchor during
training (see :func:`init_gamma` and :meth:`gamma_prior_loss`). When
``init_rna_half_life`` is omitted, the legacy scalar ``gamma=1.0`` init is
used and no prior buffer is registered (byte-identical to earlier checkpoints).
'''

import math

import torch
import torch.nn.functional as F
from gsnn.models.GSNN import GSNN

from lincs_gsnn.proc.gene_norm import mu_sigma_for_outputs


# Default mRNA half-life prior is set from the median mammalian mRNA half-life
# reported in Schwanhäusser et al., "Global quantification of mammalian gene
# expression control", Nature 473, 337-342 (2011); doi:10.1038/nature10098.
# IQR ~3-15h across the proteome; 9h is the population median.
def init_gamma(
    sigma_g: torch.Tensor,
    dxdt_scale: float,
    t_half_hours: float,
) -> torch.Tensor:
    """Per-gene physical-decay initialization for BIOGSNN.gamma.

    Returns ``gamma_model[g] = ln(2)/t_half * dxdt_scale / sigma_g[g]`` in
    the post-softplus units consumed by :meth:`BIOGSNN.get_gamma`.
    """
    k = math.log(2.0) / float(t_half_hours)  # 1/hour
    return k * float(dxdt_scale) / sigma_g


def _inverse_softplus(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Map a non-negative target through the inverse of ``softplus``."""
    return torch.log(torch.expm1(y.clamp(min=eps)))


def get_nonlin(nonlin: str):
    if nonlin == 'relu':
        return torch.nn.ReLU()
    elif nonlin == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif nonlin == 'sigmoid':
        return torch.nn.Sigmoid()
    elif nonlin == 'softplus':
        return torch.nn.Softplus()
    elif nonlin == 'elu':
        return torch.nn.ELU()
    elif nonlin == 'selu':
        return torch.nn.SELU()
    elif nonlin == 'gelu':
        return torch.nn.GELU()
    elif nonlin == 'swish':
        return torch.nn.Swish()
    else:
        raise ValueError(f'Invalid nonlin: {nonlin}')

class BIOGSNN(torch.nn.Module):
    def __init__(
        self,
        gamma=1.0,
        gsnn_kwargs=None,
        gene_norm=None,
        dxdt_nonlin=None,
        init_rna_half_life=None,
        dxdt_scale=None,
        dxdt_clip=0.0,
    ):
        super().__init__()

        # Intrinsic soft bound on the net dx/dt magnitude, in the model's
        # native (scaled-dxdt) output units. <=0 disables it. Stored on the
        # module so it is fixed at construction (pretrain) and travels with the
        # checkpoint into the odeint fine-tune, guaranteeing identical bounded
        # dynamics in both stages rather than a train-only afterthought.
        self.dxdt_clip = float(dxdt_clip or 0.0)

        if gene_norm is None:
            raise ValueError(
                "BIOGSNN requires `gene_norm` (per-gene mu/sigma payload). Build "
                "gene_norm.pt via `make_bio_network.py --gene_stats_path ...` and "
                "load it with `lincs_gsnn.proc.gene_norm.load_gene_norm_artifact`."
            )

        self.gsnn = GSNN(**gsnn_kwargs)

        if dxdt_nonlin is not None:
            self.dxdt_nonlin = get_nonlin(dxdt_nonlin)
        else:
            self.dxdt_nonlin = torch.nn.Identity()

        input_names = gsnn_kwargs['node_names_dict']['input']
        output_names = gsnn_kwargs['node_names_dict']['output']

        n_outputs = len(output_names)

        # Explicit per-output-gene -> input-column index map. Guarantees that
        # column i of `xg`, `gamma`, and `dxdt` all refer to the same gene,
        # regardless of how the bionetwork constructor orders
        # `node_names_dict['input']` after pruning.
        gene_in_idx = {n: i for i, n in enumerate(input_names) if n.startswith('GENE__')}
        out_to_inp = torch.tensor(
            [gene_in_idx.get(n, -1) for n in output_names], dtype=torch.long
        )
        missing = int((out_to_inp < 0).sum().item())
        if missing:
            raise ValueError(
                f"BIOGSNN: {missing} of {len(output_names)} output genes have no "
                "matching GENE__ entry in node_names_dict['input']. Rebuild the "
                "bionetwork so every output gene also appears as an input."
            )
        self.register_buffer('out_to_inp_idx', out_to_inp)

        # Per-output-gene control-population mu/sigma from gene_norm.pt,
        # reordered to match `output_names`. Hard-fails on any missing
        # output gene. Buffers travel with state_dict so saved BIOGSNNs are
        # self-contained at explain-time.
        mu_g, sigma_g = mu_sigma_for_outputs(gene_norm, output_names)
        self.register_buffer('mu_g', mu_g)
        self.register_buffer('sigma_g', sigma_g)

        if init_rna_half_life is not None:
            if dxdt_scale is None:
                raise ValueError(
                    "BIOGSNN: `init_rna_half_life` requires `dxdt_scale` "
                    "(the global dx/dt normalization factor from DXDTDataset)."
                )
            gamma_phys = init_gamma(sigma_g, dxdt_scale, init_rna_half_life)
            self.gamma = torch.nn.Parameter(_inverse_softplus(gamma_phys))
            self.register_buffer('gamma_prior', gamma_phys.detach())
            self.register_buffer(
                'gamma_prior_t_half',
                torch.tensor(float(init_rna_half_life), dtype=torch.float32),
            )
        else:
            self.gamma = torch.nn.Parameter(torch.ones(n_outputs) * gamma)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # nn.Module's __getattr__
        except AttributeError:
            # Forward unknown attribute access to the wrapped GSNN so that
            # explainers and ODE wrappers can read e.g. model.edge_index,
            # model.homo_names, model.num_nodes. Guard against infinite
            # recursion when `gsnn` itself hasn't been registered yet
            # (during __init__ or unpickling).
            modules = self.__dict__.get('_modules')
            if not modules or 'gsnn' not in modules:
                raise AttributeError(name)
            return getattr(modules['gsnn'], name)

    def forward(self, x, node_mask=None, edge_mask=None, ret_edge_out=False, e0=None, node_errs=None, x_fn=None):

        # Edge-level features are unaffected by the BIO production/degradation
        # transform; pass them straight through so edge-attribution explainers
        # see raw GSNN edge features rather than malformed broadcasts.
        if ret_edge_out:
            return self.gsnn(x, node_mask=node_mask, edge_mask=edge_mask,
                             ret_edge_out=True, e0=e0, node_errs=node_errs,
                             x_fn=x_fn)

        tf_activity = self.predict_dxdt(x, node_mask=node_mask, edge_mask=edge_mask,
                                               e0=e0, node_errs=node_errs, x_fn=x_fn)
        level3 = self.predict_level3(x)
        gamma = self.get_gamma()

        # The degredation rate is in terms of level3 units, while tf_activity is zscored and dxdt scaled
        # 1) need to scale by dxdt_scale ?
        # 2) need to convert back to zscore after degredation computation ?
        dxdt = tf_activity - gamma.unsqueeze(0) * level3

        # Smoothly saturate the net derivative at +/- dxdt_clip when enabled.
        # tanh keeps gradients non-zero in the saturating regime. getattr
        # fallback keeps older checkpoints (no dxdt_clip attribute) working.
        c = getattr(self, 'dxdt_clip', 0.0)
        if c and c > 0.0:
            dxdt = c * torch.tanh(dxdt / c)

        return dxdt

    def get_gamma(self):
        # Per-output-gene first-order mRNA degradation rate, constrained
        # non-negative via softplus on the raw learnable parameter.
        return F.softplus(self.gamma)

    def gamma_prior_loss(self):
        """Soft log-rate L2 anchor toward :attr:`gamma_prior` when registered."""
        if not hasattr(self, 'gamma_prior'):
            return torch.zeros((), device=self.gamma.device)
        eps = 1e-8
        log_gamma = self.get_gamma().clamp(min=eps).log()
        log_prior = self.gamma_prior.clamp(min=eps).log()
        return ((log_gamma - log_prior) ** 2).mean()

    def predict_dxdt(self, x, node_mask=None, edge_mask=None, e0=None, node_errs=None, x_fn=None):
        # Raw GSNN output interpreted as a net transcription contribution
        # to dx/dt. Sign is unconstrained: the wrapped GSNN may emit
        # negative values to represent net repressive regulation.
        out = self.gsnn(x,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
                        ret_edge_out=False,
                        e0=e0,
                        node_errs=node_errs,
                        x_fn=x_fn)

        out = self.dxdt_nonlin(out)

        return out

    def predict_level3(self, x):
        # Back-transform the z-scored gene state into a non-negative
        # Level-3 expression proxy aligned to output-gene order.
        x_z = x[:, self.out_to_inp_idx]
        return F.relu(self.mu_g + self.sigma_g * x_z)
