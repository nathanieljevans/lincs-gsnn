import torch 
from torchdiffeq import odeint

class ODEFunc(torch.nn.Module): 
    def __init__(self, model, input_names, scale=1.0):
        super().__init__() 
        self.model = model
        self.input_names = input_names
        self.gene_ixs = torch.tensor([i for i, name in enumerate(input_names) if name.startswith('GENE__')] , dtype=torch.long) 
        self.scale = scale
        self.edge_mask = None
        self.node_mask = None
        # Optional per-batch function-node-activity features. Stays None for
        # GSNNs trained without node_activity so the model() call below is
        # byte-identical to the legacy invocation.
        self.x_fn = None
        self.edge_index = model.edge_index

    def set_edge_mask(self, edge_mask):
        self.edge_mask = edge_mask

    def set_node_mask(self, node_mask):
        self.node_mask = node_mask

    def set_x_fn(self, x_fn):
        self.x_fn = x_fn

    def forward(self, t, x):
        # x shape: (B, n_input_nodes)

        # Only forward x_fn when it was explicitly set so legacy GSNNs (no
        # node_activity) keep their original call signature untouched.
        # dx/dt saturation (dxdt_clip) lives inside the model (e.g. BIOGSNN)
        # so the bounded dynamics are intrinsic and identical at pretrain and
        # integrate time, rather than applied as a post-hoc wrapper here.
        extra_kwargs = {} if self.x_fn is None else {'x_fn': self.x_fn}
        out = self.model(x, edge_mask=self.edge_mask, node_mask=self.node_mask, **extra_kwargs) # (B, n_output_nodes) 

        out = out*self.scale

        # need to return dxdt in the same shape as input nodes 
        dxdt = torch.zeros_like(x)
        # only fill in the gene derivatives
        dxdt[:, self.gene_ixs] = out

        return dxdt

    def integrate(self, x, time, node_mask=None, edge_mask=None, x_fn=None, method='dopri5', tol=1e-4):

        self.set_node_mask(node_mask)
        self.set_edge_mask(edge_mask)
        self.set_x_fn(x_fn)

        out = odeint(func=self, y0=x, t=time, method=method, atol=tol, rtol=tol) # shape: (n_time, B, n_input_nodes)

        return out[:, :, self.gene_ixs] # ordered as self.input_names[self.gene_ixs]