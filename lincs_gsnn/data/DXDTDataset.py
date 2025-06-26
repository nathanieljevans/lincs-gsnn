from torch.utils.data import Dataset
import torch
from DeepTraj.models.DeepTraj import dose_transform


class DXDTDataset(Dataset):
    def __init__(self, meta, input_names, obs_dir='', scale=None, return_time=False):

        self.meta = meta
        self.obs_dir = obs_dir 
        self.input_names = input_names
        self.gene_ixs = [i for i, name in enumerate(input_names) if name.startswith('GENE__')]
        self.return_time = return_time

        if scale is None: 
            self._scale = self.estimate_dxdt_std(n_samples=10000)  # Estimate the standard deviation of dx/dt
        else: 
            self._scale = scale

    def estimate_dxdt_std(self, n_samples=250):

        dxdt_samples = []
        for _ in range(n_samples):
            dxdt_samples.append(self.get(torch.randint(0, len(self), (1,)).item())[1].view(-1))
        dxdt_samples = torch.stack(dxdt_samples, dim=0)  # (n_samples, n_genes)
        dxdt_std = dxdt_samples.std()
        return dxdt_std.item()

    def get(self, idx): 
        row = self.meta.iloc[idx]
        obs = torch.load(f"{self.obs_dir}/{row['file_name']}", weights_only=False, map_location='cpu').type(torch.float32)
        x = obs[0]
        dxdt = obs[1]

        X = torch.zeros(len(self.input_names), dtype=torch.float32) 
        X[self.input_names.index('DRUG__' + row['pert_id'])] = dose_transform(torch.tensor([row['dose']], dtype=torch.float32)) 
        X[self.input_names.index('LINE__' + row['cell_iname'])] = 1.0 
        
        X[self.gene_ixs] = x

        dxdt = dxdt.contiguous().detach() 

        # time:
        t = torch.tensor([row['time']], dtype=torch.float32)

        return X, dxdt, t

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        
        X, dxdt, t = self.get(idx)

        if self.return_time:
            return X, dxdt/self._scale, t
        else:
            return X, dxdt/self._scale