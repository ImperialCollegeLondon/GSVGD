import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
import numpy as np
from geomloss import SamplesLoss
import sys
sys.path.append(".")
from src.kernel import RBF
from src.kernel import l2norm

def kldivergence(x, y):
    ## Adapted from https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    term1 = -np.sum(np.log(r[s!=0]/s[s!=0] + 1e-14)) * d / n
    term2 = np.log(m / (n - 1.))
    return  term1 + term2


def mmd_rbf(X, Y, sigma=1):
    '''MMD with RBF kernel
    '''
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    inv_sigma2 = 1 / (sigma + 1e-16)**2
    K_XY = (-0.5 * inv_sigma2 * (-2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0))).exp()
    K_XX = (-0.5 * inv_sigma2 * (-2 * XX + XX.diag().unsqueeze(1) + XX.diag().unsqueeze(0))).exp()
    K_YY = (-0.5 * inv_sigma2 * (-2 * YY + YY.diag().unsqueeze(1) + YY.diag().unsqueeze(0))).exp()

    # unbiased MMD estimator
    # subtracting m and n as diagonals of K_XX and K_YY are ones
    m = K_XX.shape[0]
    n = K_YY.shape[0]
    mmd = (K_XX.sum() - m) / (m * (m - 1)) + (K_YY.sum() - n) / (n * (n - 1)) - 2 * K_XY.mean()
    return mmd.item()
 
def energy_dist(X, Y):
    '''MMD with RBF kernel
    '''
    K_XY = torch.cdist(X, Y)**2
    K_XX = torch.cdist(X, X.clone())**2
    K_YY = torch.cdist(Y, Y.clone())**2

    # unbiased MMD estimator
    # subtracting m and n as diagonals of K_XX and K_YY are ones
    m = K_XX.shape[0]
    n = K_YY.shape[0]
    energy = - (K_XX.sum() - m) / (m * (m - 1)) - (K_YY.sum() - n) / (n * (n - 1)) + 2 * K_XY.mean()
    return energy.item()


def mmd_average(X, Y, sigmas=[1e-3, 1e-2, 1e-1, 1, 10]):
    '''MMD with RBF kernel averaged over a range of lengthscales.
    '''
    mmds = np.array([mmd_rbf(X, Y, s) for s in sigmas])
    return mmds.mean()


class Metric:
    def __init__(self, metric, x_init, x_target, target_dist=None, w=None, b=None, device="cpu"):
        self.metric = metric
        self.target_dist = target_dist
        self.w = w
        self.b = b
        self.device = device

        if metric == "mmd":
            # use median heuristic for mmd
            k_mmd = RBF(method="med_heuristic")
            _ = k_mmd(x_init.detach(), x_target.detach())
            sigma = k_mmd.sigma
            self.metric_fn = lambda y: mmd_rbf(y.detach(), x_target.detach(), sigma=sigma)

        elif metric == "kldiv":
            self.metric_fn = lambda y: kldivergence(y.detach(), x_target.detach())

        elif metric == "mmd_both":
            # use median heuristic for mmd
            k_mmd = RBF(method="med_heuristic")
            _ = k_mmd(x_init.detach(), x_target.detach())
            # median heuristic for first two dims
            k_mmd_sub = RBF(method="med_heuristic")
            _ = k_mmd_sub(x_init[:, :2].detach(), x_target[:, :2].detach())
            def metric_fn(y):
                res = [
                    mmd_rbf(y.detach(), x_target.detach(), sigma=k_mmd.sigma),
                    mmd_rbf(
                        y.detach()[:, :2], 
                        x_target.detach()[:, :2], 
                        sigma=k_mmd_sub.sigma)
                ]
                return res
            self.metric_fn = metric_fn

        elif metric == "mmd_ave":
            # evaluate mmd with a range of lenthscales and average
            self.metric_fn = lambda y: mmd_average(y.detach(), x_target.detach())

        elif metric == "wass":
            # wasserstein-1 distance
            wass = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.8)
            self.metric_fn = lambda y: wass(y.detach(), x_target.detach()).item()

        elif metric == "wass_sub":
            # wasserstein-1 distance
            wass = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.8)
            self.metric_fn = lambda y: wass(y[:, :2].detach(), x_target[:, :2].detach()).item()

        elif metric == "energy":
            # energy distance
            energy = SamplesLoss("energy")
            self.metric_fn = lambda y: energy(y.detach(), x_target.detach()).item()

        elif metric == "energy2":
            # energy distance calculated manually
            self.metric_fn = lambda y: energy_dist(y.detach(), x_target.detach())

        elif metric == "energy_sub":
            # energy distance
            energy = SamplesLoss("energy")
            self.metric_fn = lambda y: energy(y[:, :2].detach(), x_target[:, :2].detach()).item()

        elif metric == "energy_both":
            # energy distance
            energy = SamplesLoss("energy")
            def metric_fn(y):
                res = [
                    energy(y.detach(), x_target.detach()).item(),
                    energy(y[:, :2].detach(), x_target[:, :2].detach()).item()
                ]
                return res
            self.metric_fn = metric_fn

        elif metric == "mean":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            ground_truth = true_samples.mean(axis=0)
            self.metric_fn = lambda y: ((y.mean(axis=0) - ground_truth)**2).mean().item()
            def metric_fn(y):
                return ((y.mean(axis=0) - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "squared":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            ground_truth = (true_samples**2).mean(axis=0)
            def metric_fn(y):
                return ((torch.mean(y**2, axis=0) - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "cos":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            def metric_fn(y):
                ground_truth = torch.cos(self.w*true_samples + self.b).mean(axis=0)
                cos_y = torch.cos(self.w*y + self.b).mean(axis=0)
                return ((cos_y - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "cov_mat":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            true_samples = true_samples - true_samples.mean(axis=0)
            ground_truth = (true_samples.t() @ true_samples) / true_samples.shape[0]
            def metric_fn(y):
                y = y - y.mean(axis=0)
                est_cov = y.t() @ y / y.shape[0]
                return ((est_cov - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn  

        elif metric == "cov_mat_sub":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            true_samples = true_samples[:, :2]
            true_samples = true_samples - true_samples.mean(axis=0)
            ground_truth = (true_samples.t() @ true_samples) / true_samples.shape[0]
            def metric_fn(y):
                y = y[:, :2]
                y = y - y.mean(axis=0)
                est_cov = y.t() @ y / y.shape[0]
                return ((est_cov - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn  

        elif metric == "cov_error":
            # \| cov - cov_true \|_2
            true_samples = self.target_dist.sample((200000, )).cpu()
            cov_true = np.cov(true_samples.T)
            def metric_fn(y):
                y = y.cpu()
                cov_mat = np.cov(y.T)
                return np.sqrt(np.sum((cov_mat - cov_true)**2))
            self.metric_fn = metric_fn

        elif metric == "var":
            # dim-averaged margnial vars
            def metric_fn(y):
                return torch.var(y, axis=0).mean().item()
            self.metric_fn = metric_fn

        elif metric == "var_sub":
            # dim-averaged margnial vars for all but first two dims
            def metric_fn(y):
                return torch.var(y[:, 2:], axis=0).mean().item()
            self.metric_fn = metric_fn
 
    def __call__(self, x, **kwargs):
        return self.metric_fn(x, **kwargs)
