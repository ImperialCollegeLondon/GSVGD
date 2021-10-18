import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch import einsum
from tqdm import tqdm, trange
import autograd.numpy as np


class GSVGD:
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, T=1, device="cpu", noise=True):
        self.target = target
        self.k = kernel
        self.optim = optimizer
        self.manifold = manifold
        self.delta = delta
        self.sigma_list = []
        self.device = device
        self.noise = noise
        self.T = T
        self.langevin = torch.distributions.Normal(
            torch.zeros(1).to(device), torch.tensor(1).to(device)
        )

    def alpha(self, X, A):
        """A here has orthonormal cols, so is actually P^T in the notes"""
        # PX
        y0 = einsum("ij, jk -> ik", X.detach(), A)

        # score
        # num_particles
        log_prob = self.target.log_prob(X)
        # num_particles x dim
        score = autograd.grad(log_prob.sum(), X)[0]
        # num_particles x proj_dim
        A_score = einsum("ij, jk -> ik", score, A)

        # Gram matrix
        # num_particles x num_particles
        K_AxAx = self.k(y0, y0.detach())
        # proj_dim x num_particles x num_particles
        grad_first_K_Ax = self.k.grad_first(y0, y0)
        # proj_dim x proj_dim x num_particles x num_particles
        gradgrad_K_AxAx = self.k.gradgrad(y0, y0)

        # term 1
        # num_particles x num_particles
        prod = einsum("ij, ij -> ij", A_score @ A_score.T, K_AxAx)
        term1 = prod.sum()

        # term 2
        term2 = einsum("ij, jki -> ki", A_score, grad_first_K_Ax).sum()

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = term2

        # term 4
        # ## compute grad_grad_K directly in matrix form
        term4 = einsum("iijk -> jk", gradgrad_K_AxAx).sum()

        return (term1 + term2 + term3 + term4) / X.shape[0] ** 2

    def phi(self, X, A, projection_epochs):
        # detach from current graph and create a new graph
        X = X.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)
        # TODO check later whether need detach
        AX = (A.t() @ X.t()).t()

        log_prob = self.target.log_prob(X)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X)[0]

        # num_particles x num_particles
        K_XX = self.k(AX, AX.detach())
        # num_particles x proj_dim
        grad_K = -autograd.grad(K_XX.sum(), AX)[0]


        # compute alpha(P)
        # dim x proj_dim
        if self.noise:
            for _ in range(projection_epochs):
                grad_A = autograd.grad(self.alpha(X, A), A)[0]
                # Add noise
                A = self.manifold.retr(
                    A.detach(),
                    self.manifold.egrad2rgrad(
                        A.detach(),
                        self.delta * grad_A.detach()
                        + np.sqrt(2.0 * self.T * self.delta)
                        * self.langevin.sample((self.manifold._n, self.manifold._p)).squeeze(
                            -1
                        ),
                    ),
                )
        else:
            for _ in range(projection_epochs):
                grad_A = autograd.grad(self.alpha(X, A), A)[0]
                # No noise
                A = self.manifold.retr(
                    A.detach(),
                    self.manifold.egrad2rgrad(
                        A.detach(),
                        self.delta * grad_A.detach()
                    ),
                )
        # compute phi
        # num_particles x dim
        phi = (
            K_XX.detach().matmul(torch.einsum("ij, jk -> ik", score_func, A @ A.t()))
            + torch.einsum("ij, jk -> ik", grad_K, A.t())
        ) / X.size(0)

        return phi, A

    def step(self, X, A, projection_epochs):
        self.optim.zero_grad()
        phi, A = self.phi(X, A, projection_epochs)
        X.grad = -phi
        self.optim.step()
        return A

    def fit(self, X, A, epochs=100, projection_epochs=1, verbose=True, metric=None, save_every=100):
        # self.A_list = [0] * epochs
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = X.clone().detach().cpu()
        if verbose:
            pbar = trange(epochs)
            # for i in tqdm(range(epochs)):
            for i in pbar:
                A = self.step(X, A, projection_epochs)
                # self.A_list[i] = A.clone().detach().cpu().numpy()
                if metric and (i+1)%save_every==0:
                    self.metrics[i//save_every] = metric(X.detach())
                    self.particles[1 + i//save_every] = X.clone().detach().cpu()
        else:
            for i in range(epochs):
                A = self.step(X, A, projection_epochs)
                # self.A_list[i] = A.clone().detach().cpu().numpy()
                if metric and (i+1)%save_every==0:
                    self.metrics[i//save_every] = metric(X.detach())
                    self.particles[1 + i//save_every] = X.clone().detach().cpu()
        return A, self.metrics

    def pamrf(self, X, A):
        X, A = X.detach(), A.detach()
        y0 = einsum("ij, jk -> ik", X, A)
        grad_first_K_Ax = self.k.grad_first(y0, y0)
        repulsion = torch.mean(einsum("ijk, li -> jkl", grad_first_K_Ax, A), dim=0)
        res = torch.max(repulsion, dim=0).mean()
        return res

