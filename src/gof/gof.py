"""Adapted f/om 
https://github.com/karlnapf/kernel_goodness_of_fit/blob/master/goodness_of_fit/test.py
"""
import torch
from torch import einsum
import torch.autograd as autograd
import numpy as np
from tqdm import tqdm
from src.Sliced_KSD_Clean.Divergence.Def_Divergence import compute_max_DSSD_eff


class KSD:
    def __init__(self, kernel, log_prob=None, score=None):
        self.log_prob = log_prob
        self.k = kernel
        self.score = score

    def __call__(self, x, y):
        if not self.score:
            log_px = self.log_prob(x)
            score_x = torch.autograd.grad(log_px.sum(), x)[0]
            log_py = self.log_prob(y)
            score_y = torch.autograd.grad(log_py.sum(), y)[0]
        else:
            score_x = self.score(x)
            score_y = self.score(y)

        Kxy = self.k(x, y)
        grad_x_Kxy = self.k.grad_first(x, y)
        grad_y_Kxy = self.k.grad_second(x, y)
        gradgrad_K = self.k.gradgrad(x, y)

        term1 = einsum("ij, kj -> ik", score_x, score_y) * Kxy
        term2 = einsum("ij, jik -> ik", score_x, grad_y_Kxy)
        term3 = einsum("ij, jki -> ki", score_y, grad_x_Kxy)
        term4 = einsum("iijk -> jk", gradgrad_K)
        res = term1 + term2 + term3 + term4
        return res.detach().cpu().numpy()


class PKSD:
    def __init__(self, kernel, P, manifold, log_prob=None, score=None):
        self.log_prob = log_prob
        self.k = kernel
        self.P = P
        self.manifold = manifold
        self.score = score

    def __call__(self, x, y, P=None, numpy=True):
        """P must have orthonormal rows"""
        P = self.P if P is None else P
        assert (
            P.shape[1] == x.shape[1]
        ), "Number of columns of P and dimension of x must match"

        Pt = P.T
        Px = einsum("ij, jk -> ik", x.detach(), Pt)
        Py = einsum("ij, jk -> ik", y.detach(), Pt)

        if not self.score:
            log_probx = self.log_prob(x)
            log_proby = self.log_prob(y)
            score_x = autograd.grad(log_probx.sum(), x)[0]
            score_y = autograd.grad(log_proby.sum(), y)[0]
        else:
            score_x = self.score(x)
            score_y = self.score(y)

        P_score_x = einsum("ij, jk -> ik", score_x, Pt)
        P_score_y = einsum("ij, jk -> ik", score_y, Pt)

        # Gram matrices
        K_PxPy = self.k(Px.detach(), Py.detach())

        # grad and jacobian
        grad_first_K_PxPy = self.k.grad_first(Px, Py)
        grad_second_K_PxPy = self.k.grad_second(Px, Py)
        gradgrad_K_PxPy = self.k.gradgrad(Px, Py)

        # term 1
        term1 = einsum("ij, ij -> ij", P_score_x @ P_score_y.T, K_PxPy)
        term2 = einsum("ij, jki -> ki", P_score_y, grad_first_K_PxPy)
        term3 = einsum("ij, jik -> ik", P_score_x, grad_second_K_PxPy)
        term4 = einsum("iijk -> jk", gradgrad_K_PxPy)
        res = term1 + term2 + term3 + term4
        if numpy:
            return res.detach().cpu().numpy()
        else:
            return res

    def step(self, P, x, y, delta):
        # compute grad_PKSD(P)
        P = torch.Tensor(P.cpu()).detach().requires_grad_(True).to(P.device)
        grad_P = autograd.grad(self.__call__(x, y, P, numpy=False).sum(), P)[0]
        # gradient descent along manifold
        P = self.manifold.retr(
            P.detach().t(),
            self.manifold.egrad2rgrad(P.detach().t(), delta * grad_P.detach().t()),
        )
        return P.t()

    def train_projection(self, x, y, epochs=100, delta=1e-4):
        P_ls = [0] * epochs
        stats = [0] * epochs
        for i in range(epochs):
            P = torch.Tensor(self.P.cpu()).requires_grad_(True).to(self.P.device)
            P = self.step(P, x, y, delta)
            P_ls[i] = P
            stat_matrix = self.__call__(
                x.detach().requires_grad_(True),
                y.detach().requires_grad_(True),
                P.detach().requires_grad_(True),
            )
            stats[i] = (stat_matrix - np.diag(np.diag(stat_matrix))).sum() / (
                x.shape[0] ** 2 - x.shape[0]
            )
            self.P = torch.Tensor(P.cpu()).to(P.device)
        self.P_history = P_ls
        return P_ls, stats


class MaxSKSD:
    def __init__(
        self,
        kernel,
        d_kernel,
        dd_kernel,
        optimizer,
        log_prob=None,
        score=None,
        r=None,
        g=None,
    ):
        self.log_prob = log_prob
        self.score = score
        self.kernel = kernel
        self.d_kernel = d_kernel
        self.dd_kernel = dd_kernel
        self.r = r
        self.g = g
        self.optimizer = optimizer
        self.kernel_hyper = {"bandwidth": None}

    def __call__(self, x, y, r=None, g=None):
        """Compute U-Statistic of maxSKSD

        Args:
            x ([type]): [description]
            y ([type]): [description]
            r ([type], optional): [description]. Defaults to None.
            g ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        r = self.r if r is None else r
        g = self.g if g is None else g
        
        if not self.score:
            log_prob_x = self.log_prob(x)
            log_prob_y = self.log_prob(y)
            score_x = torch.autograd.grad(log_prob_x.sum(), x)[0]
            score_y = torch.autograd.grad(log_prob_y.sum(), y)[0]
        else:
            score_x = self.score(x)
            score_y = self.score(y)

        return (
            compute_max_DSSD_eff(
                x,
                y,
                None,
                self.kernel,
                self.d_kernel,
                self.dd_kernel,
                r=r,
                g=g,
                score_samples1=score_x,
                score_samples2=score_y,
                flag_U=True,
                flag_median=True,
                median_power=0.5,
                bandwidth_scale=1,
                kernel_hyper=self.kernel_hyper,
            )[1]
            .sum(0)
            .detach()
            .cpu()
            .numpy()
        )

    def step(self, g, x, y):
        self.optimizer.zero_grad()
        g = g / (torch.norm(g, 2, dim=-1, keepdim=True) + 1e-10)

        # compute scores
        if not self.score:
            log_prob_x = self.log_prob(x)
            log_prob_y = self.log_prob(y)
            score_x = torch.autograd.grad(log_prob_x.sum(), x)[0]
            score_y = torch.autograd.grad(log_prob_y.sum(), y)[0]
        else:
            x = x.detach().requires_grad_()
            y = y.detach().requires_grad_()
            score_x = self.score(x)
            score_y = self.score(y)

        diver, _ = compute_max_DSSD_eff(
            x,
            y,
            None,
            kernel=self.kernel,
            d_kernel=self.d_kernel,
            dd_kernel=self.dd_kernel,
            r=self.r,
            g=g,
            kernel_hyper=self.kernel_hyper,
            score_samples1=score_x,
            score_samples2=score_y,
            flag_median=True,
            flag_U=False,
            median_power=0.5,
            bandwidth_scale=1,
        )
        (-diver).backward()
        self.optimizer.step()
        return diver

    def train_projection(self, x, y, g, epochs=100):
        g_ls = [0] * epochs
        stats = [0] * epochs
        for i in range(epochs):
            # for i in range(epochs):
            diver = self.step(g.requires_grad_(True), x, y)
            self.g = g
            g_ls[i] = g
            stats[i] = diver
        return g_ls, stats


class GoodnessOfFitTest:
    def __init__(self, discrepancy, x):
        """
        Args:
            discrepancy: A callable that returns a matrix of summands of the
                Stein discrepancy. Must take in exactly two arguments for
                the samples: SD(x, y).
            x: Observed samples.
        """
        self.d = discrepancy
        self.x = x
        self.n = x.shape[0]

    def bootsrap_stat(self, stat_matrix):
        u_matrix = stat_matrix - np.diag(np.diag(stat_matrix))
        weights = np.random.multinomial(n=self.n, pvals=np.ones(self.n) / self.n)
        W = (weights - 1.0) / self.n
        res = W @ u_matrix @ W
        return res

    def bootstrap(self, nbootstrap, stat_matrix):
        bootstrap_stats = np.zeros(nbootstrap)
        for i in range(nbootstrap):
            bootstrap_stats[i] = self.bootsrap_stat(stat_matrix)
        return bootstrap_stats

    def compute_pvalue(self, nbootstrap):
        """
        Arg:
            nbootstrap: Number of bootstrap samples.
        Return:
            bootstrap_stats: bootstrap samples
            stat: U-statistic computeted from the given discrepancy
            pvalue: p-value based on the bootstrap samples
        """
        stat_matrix = self.d(self.x, self.x)
        bootstrap_stats = self.bootstrap(nbootstrap, stat_matrix)
        stat = (stat_matrix - np.diag(np.diag(stat_matrix))).sum() / (
            self.n * (self.n - 1)
        )
        pvalue = (bootstrap_stats > stat).mean()
        return (bootstrap_stats, stat, pvalue)

