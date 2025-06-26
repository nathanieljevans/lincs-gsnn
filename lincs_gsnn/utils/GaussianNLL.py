import torch 

class GaussianNLL(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target_mu, target_sigma):
        '''
        unlike `torch.nn.GaussianNLLLoss`, this loss assumes the target is a distribution with mean `target_mu` and standard deviation `target_sigma`.
        the input is a single predicted sample and the loss the is the NLL of the target distribution given the predicted sample.

        Args: 
            target_mu (torch.Tensor): target mean, shape (B, n_genes)
            target_sigma (torch.Tensor): target standard deviation, shape (B, n_genes)
            input (torch.Tensor): predicted sample, shape (B, n_genes)
        Returns:
            torch.Tensor: mean negative log likelihood loss, shape (1,)
        '''
        P = torch.distributions.Normal(target_mu, target_sigma)
        loss = -P.log_prob(input).mean() 
        return loss 
        