import torch
import numpy as np
import numpy as np
from geomloss import SamplesLoss
from src.kernel import RBF

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


class Metric:
    def __init__(self, metric, x_init, x_target, target_dist=None, w=None, b=None, device="cpu"):
        self.metric = metric
        self.target_dist = target_dist
        self.w = w
        self.b = b
        self.device = device

        if metric == "wass":
            # wasserstein-1 distance
            wass = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.8)
            self.metric_fn = lambda y: wass(y.detach(), x_target.detach()).item()

        elif metric == "wass_sub":
            # wasserstein-1 distance for first 2-dimensional marginal
            wass = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.8)
            self.metric_fn = lambda y: wass(y[:, :2].detach(), x_target[:, :2].detach()).item()

        elif metric == "energy":
            # energy distance
            energy = SamplesLoss("energy")
            self.metric_fn = lambda y: energy(y.detach(), x_target.detach()).item()

        elif metric == "energy_sub":
            # energy distance for first 2-dimensional marginal
            energy = SamplesLoss("energy")
            self.metric_fn = lambda y: energy(y[:, :2].detach(), x_target[:, :2].detach()).item()

        elif metric == "mean":
            # mse for test function E[x]
            true_samples = self.target_dist.sample((100000, ))
            ground_truth = true_samples.mean(axis=0)
            self.metric_fn = lambda y: ((y.mean(axis=0) - ground_truth)**2).mean().item()
            def metric_fn(y):
                return ((y.mean(axis=0) - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "squared":
            # mse for test function E[x**2]
            true_samples = self.target_dist.sample((100000, ))
            ground_truth = (true_samples**2).mean(axis=0)
            def metric_fn(y):
                return ((torch.mean(y**2, axis=0) - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "cos":
            # mse for test function E[cos(w*x + b)]
            true_samples = self.target_dist.sample((100000, ))
            def metric_fn(y):
                ground_truth = torch.cos(self.w*true_samples + self.b).mean(axis=0)
                cos_y = torch.cos(self.w*y + self.b).mean(axis=0)
                return ((cos_y - ground_truth)**2).mean().item()
            self.metric_fn = metric_fn

        elif metric == "cov_error":
            # cov estimation error: \| sample_cov_matrix - true_cov_matrix \|_2^2
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
