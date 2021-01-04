
import sys
import torch
from torch import autograd

epsilon = 10e-8

class PathRegularization():

    def __init__(self):
        self.pl_mean_var = 0.0
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
    def __call__(self, G, minibatch_size, pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
        
        pl_minibatch_size = minibatch_size // pl_minibatch_shrink
        latent_z = torch.randn(pl_minibatch_size, G.dim_latent, device=self.device)
        G.save_dlatents = True
        fake_image_out = G(latent_z)
        G.save_dlatents = False
        dlatents = G.dlatents
        pl_noise = torch.randn_like(fake_image_out,device=self.device) / torch.sqrt(torch.Tensor([fake_image_out.size()[2:].numel()],device=self.device))
        grad, = autograd.grad(outputs=torch.sum(fake_image_out*pl_noise),inputs=dlatents,create_graph=True,retain_graph=True)
        path_lengths = torch.sqrt(grad.pow(2).mean(1) + epsilon)
        pl_mean = self.pl_mean_var + pl_decay*(path_lengths.mean() - self.pl_mean_var)
        self.pl_mean_var = pl_mean.detach()
        pl_penalty = (path_lengths - pl_mean).pow(2).mean() * pl_weight
        
        return pl_penalty, pl_mean.detach()


sys.modules[__name__] = PathRegularization()