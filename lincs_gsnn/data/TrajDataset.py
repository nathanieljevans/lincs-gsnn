from torch.utils.data import Dataset
import torch
from DeepTraj.models.DeepTraj import dose_transform


class TrajDataset(Dataset):
    def __init__(self, meta, input_names, obs_dir='', horizon=None, multiple_shooting=False):

        self.meta = meta
        self.obs_dir = obs_dir 
        self.input_names = input_names
        self.gene_ixs = [i for i, name in enumerate(input_names) if name.startswith('GENE__')]
        self.horizon = horizon
        self.multiple_shooting = multiple_shooting

    def set_horizon(self, horizon): 
        self.horizon = horizon

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        obs_dict = torch.load(f"{self.obs_dir}/{row['file_name']}", weights_only=False)

        obs_mu = obs_dict['mean'].type(torch.float32)  # (n_time, n_genes)
        obs_sigma = obs_dict['std'].type(torch.float32)  # (n_time, n_genes))

        if (self.horizon is not None): 
            if self.multiple_shooting: 
                # multiple shooting: https://docs.sciml.ai/DiffEqFlux/stable/examples/multiple_shooting/#:~:text=In%20Multiple%20Shooting%2C%20the%20training,without%20splitting
                t0 = torch.randint(0, obs_mu.shape[0] - self.horizon + 1, size=(1,)).item()  # random start time
            else: 
                t0 = 0 
                
            tT = t0 + self.horizon 
            obs_mu = obs_mu[t0:tT, :]
            obs_sigma = obs_sigma[t0:tT, :]

        t0_mu = obs_mu[0,:] # (978,)

        x = torch.zeros(len(self.input_names), dtype=torch.float32) 
        x[self.input_names.index('DRUG__' + row['pert_id'])] = dose_transform(torch.tensor([row['dose']], dtype=torch.float32)) 
        x[self.input_names.index('LINE__' + row['cell_iname'])] = 1.0 
        x[self.gene_ixs] = t0_mu

        return obs_mu, obs_sigma, x.contiguous().detach()