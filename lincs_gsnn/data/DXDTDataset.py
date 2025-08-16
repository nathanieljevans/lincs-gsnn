from torch.utils.data import Dataset
import torch
from typing import Optional, List
from DeepTraj.models.DeepTraj import dose_transform


class DXDTDataset(Dataset):
    """
    Dataset that returns (X, dxdt, [t]) tensors for training the GSNN on dX/dt targets.

    Parameters
    ----------
    meta : pandas.DataFrame
        Table with at least the columns: `file_name`, `pert_id`, `dose`, `cell_iname`, `time`.
    input_names : list[str]
        Complete list of node names (genes, drugs, cell lines, …) used by the GSNN.
    obs_dir : str
        Directory that contains the `.pt` observation files referenced in `meta['file_name']`.
    scale : float | None
        If None, the global std-dev of dx/dt is estimated on the fly and the returned
        dx/dt is divided by it. Otherwise the provided value is used.
    return_time : bool
        If True, each sample additionally returns the time scalar `t`.
    src_names : list[str] | None
        Optional explicit list with the order of genes inside each observation file.
        When omitted, the order is inferred from the first observation and assumed
        to match the order of `GENE__` entries in `input_names`.
    """

    def __init__(self, meta, input_names, output_names, src_names, obs_dir: str = "", scale: Optional[float] = None,
                 return_time: bool = False):

        self.meta = meta
        self.obs_dir = obs_dir
        self.input_names = input_names  # GSNN input order
        self.output_names = output_names  # GSNN output order

        # ------------------------------------------------------------------ #
        # Build mapping between the gene slice of x (trajectory order) and
        # the corresponding slots in X (GSNN input order).
        # ------------------------------------------------------------------ #
    
        self.src_names = src_names
        src_ixs, dst_ixs, dst2_ixs = [], [], []
        for i_src, name in enumerate(src_names):
            # src entries may be plain symbols ("TP53") or already prefixed ("GENE__TP53")
            gene_sym = name.split("__")[1] if name.startswith("GENE__") else name
            gene_node = f"GENE__{gene_sym}"

            try:
                i_dst = self.input_names.index(gene_node)
                i_dst2 = self.output_names.index(gene_node)
            except ValueError:
                # Gene missing in either input or output list → skip
                continue

            src_ixs.append(i_src)
            dst_ixs.append(i_dst)
            dst2_ixs.append(i_dst2)
        self.src_ixs = torch.tensor(src_ixs, dtype=torch.long)
        self.dst_ixs = torch.tensor(dst_ixs, dtype=torch.long)
        self.dst2_ixs = torch.tensor(dst2_ixs, dtype=torch.long)

        self.return_time = return_time

        # Estimate or store normalization factor
        if scale is None:
            self._scale = self.estimate_dxdt_std(n_samples=10000)
        else:
            self._scale = float(scale)

    # ---------------------------------------------------------------------- #
    # Helper                                                                 #
    # ---------------------------------------------------------------------- #
    def estimate_dxdt_std(self, n_samples: int = 250) -> float:
        """Rough global std-dev of dx/dt for normalisation."""
        dxdt_samples = []
        for _ in range(n_samples):
            dxdt_samples.append(self.get(torch.randint(0, len(self), (1,)).item())[1].view(-1))
        dxdt_samples = torch.stack(dxdt_samples, dim=0)
        scale = dxdt_samples.std().item() 

        assert scale > 0, f'scale is {scale}'
        return scale 

    # ---------------------------------------------------------------------- #
    # Core Dataset API                                                       #
    # ---------------------------------------------------------------------- #
    def get(self, idx: int):
        row = self.meta.iloc[idx]
        obs_path = f"{self.obs_dir}/{row['file_name']}"
        obs = torch.load(obs_path, map_location="cpu", weights_only=False).type(torch.float32)

        x = obs[0]    # gene expression (trajectory order)
        dxdt = obs[1] # ground-truth derivative (trajectory order)

        assert not torch.isnan(x).any(), f'x has nans at index {idx}'
        assert not torch.isnan(dxdt).any(), f'dxdt has nans at index {idx}'

        # Allocate full input vector
        X = torch.zeros(len(self.input_names), dtype=torch.float32)

        # Drug and cell-line channels
        drug_ix = self.input_names.index("DRUG__" + row["pert_id"])
        line_ix = self.input_names.index("LINE__" + row["cell_iname"])

        X[drug_ix] = dose_transform(torch.tensor([row["dose"]], dtype=torch.float32))
        X[line_ix] = 1.0

        # Gene channels – map from trajectory order into GSNN input order
        X[self.dst_ixs] = x[self.src_ixs]

        # Re-order dxdt accordingly so that its gene slice matches GSNN outputs
        DXDT = torch.zeros(len(self.output_names), dtype=torch.float32)
        DXDT[self.dst2_ixs] = dxdt[self.src_ixs].contiguous().detach()

        # Time (optional)
        t = torch.tensor([row["time"]], dtype=torch.float32)

        return X.clone().detach(), DXDT.clone().detach(), t.clone().detach()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int):
        X, dxdt, t = self.get(idx)
        dxdt = dxdt / self._scale

        # check for nans 
        assert not torch.isnan(X).any(), f'X has nans at index {idx}'
        assert not torch.isnan(dxdt).any(), f'DXDT has nans at index {idx}'
        assert not torch.isnan(t).any(), f't has nans at index {idx}'

        return (X, dxdt, t) if self.return_time else (X, dxdt)
