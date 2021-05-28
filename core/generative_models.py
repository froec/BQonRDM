import torch
import torch.nn as nn
import numpy as np


# ---------------------------- #
#   Variational Auto Encoder   #
# ---------------------------- #
class VariationalAutoEncoder(nn.Module):

    def __init__(self, D=None, d=None, H=None, activFun=None):
        super(VariationalAutoEncoder, self).__init__()

        # ---------------------- #
        # The encoder definition #
        # ---------------------- #

        # The encoder mean function: D -> d
        self.mu_enc = nn.Sequential(
            nn.Linear(D, 2 * H),
            activFun,
            nn.Linear(2 * H, H),
            activFun,
            nn.Linear(H, d)
        )

        # The encoder variance function: D -> d
        self.var_enc = nn.Sequential(
            nn.Linear(D, 2 * H),
            activFun,
            nn.Linear(2 * H, H),
            activFun,
            nn.Linear(H, d),
            nn.Softplus()
        )

        # ---------------------- #
        # The decoder definition #
        # ---------------------- #

        # The decoder function: d -> D
        self.mu_dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, 2 * H),
            activFun,
            nn.Linear(2 * H, D)
        )

        # The variance of the decoder: d -> D
        self.var_dec = nn.Sequential(
            nn.Linear(d, H),
            activFun,
            nn.Linear(H, 2 * H),
            activFun,
            nn.Linear(2 * H, D),
            nn.Softplus()
        )

    @staticmethod
    # The reparametrization trick
    def reparametrization_trick(mu_enc, var_enc):
        epsilon = torch.randn_like(mu_enc)  # the Gaussian random noise
        return mu_enc + torch.sqrt(var_enc) * epsilon

    def encode(self, x):
        return self.mu_enc(x), self.var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.var_dec(z)

    def forward(self, x):
        mu_z, var_z = self.encode(x)  # Points in the latent space
        z = self.reparametrization_trick(mu_z, var_z)  # Samples from z ~ q(z | x)
        mu_x, var_x = self.decode(z)  # The decoded samples
        return mu_x, var_x, mu_z, var_z


# Computes the objective function of the VAE
def VAE_loss(x, mu_x, var_x, mu_z, var_z, anneal_param):
    DATA_FIT = 0.5 * torch.mean(torch.sum(torch.log(var_x + 1e-7) + ((x - mu_x)**2) / var_x, dim=1))
    KLD = -0.5 * torch.mean(torch.sum(torch.log(var_z + 1e-7) - (mu_z ** 2) - var_z, dim=1))

    # ELBO = DATA_FIT - KLD, to minimize (- ELBO)
    return DATA_FIT + anneal_param * KLD, DATA_FIT, KLD