import torch 

class ODEFunc(torch.nn.Module): 
    def __init__(self, model, input_names, scale=1.0, SWE=None):
        super().__init__() 
        self.model = model
        self.input_names = input_names
        self.gene_ixs = torch.tensor([i for i, name in enumerate(input_names) if name.startswith('GENE__')] , dtype=torch.long) 
        self.scale = scale
        self.SWE = SWE

    def forward(self, t, x):
        # x shape: (B, n_input_nodes)

        if self.SWE is not None:
            t = t.view(1,1).expand(x.shape[0], -1)
            edge_mask = self.SWE(t).sigmoid() > 0.5 
        else:
            edge_mask = None
        
        out = self.model(x, edge_mask=edge_mask) # (B, n_output_nodes) 

        out = out*self.scale

        # need to return dxdt in the same shape as input nodes 
        dxdt = torch.zeros_like(x)
        # only fill in the gene derivatives
        dxdt[:, self.gene_ixs] = out

        return dxdt