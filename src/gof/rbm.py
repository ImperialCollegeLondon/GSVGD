'''Copied from https://github.com/wgrathwohl/LSD/
'''

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import os
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm import tqdm


# def try_make_dirs(d):
#     if not os.path.exists(d):
#         os.makedirs(d)

def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()


class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in
        self.event_shape = (B.shape[0],)

    def score(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, shape, verbose=False):
        '''Shape must be an iterable
        '''
        n = shape[0]
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        if verbose:
          for t in tqdm(range(self.burn_in)):
              x, h = self._blocked_gibbs_next(x, h)
        else:
          for t in range(self.burn_in):
              x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h