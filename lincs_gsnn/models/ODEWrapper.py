import torch 
from .ODEFunc import ODEFunc

class ODEWrapper(torch.nn.Module): 

    def __init__(self, model, data, dxdt_scale, method='dopri5', 
                    tol=1e-4, t=torch.linspace(0, 72, 50), return_last=False):
        
        super().__init__()

        self.model_ = model
        self.ode_func = ODEFunc(model, data.node_names_dict['input'], scale=dxdt_scale)
        self.data_ = data
        self.dxdt_scale_ = dxdt_scale
        self.method_ = method
        self.tol_ = tol
        self.t_ = t
        self.edge_index = model.edge_index
        self.homo_names = model.homo_names
        self.num_nodes = model.num_nodes
        self.return_last_ = return_last
    
    def forward(self, x0, edge_mask=None, node_mask=None):

        B, N = x0.shape

        xt = self.ode_func.integrate(x = x0, 
                                     time = self.t_,
                                     node_mask=node_mask, 
                                     edge_mask=edge_mask, 
                                     method=self.method_, 
                                     tol=self.tol_) # (T, B, N)

        if self.return_last_:
            return xt[-1, :, :] # (B, N)
        else:
            return xt.squeeze(1)       # (T, B, N)
    