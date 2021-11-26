import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class Diffusion():
    def __init__(self, dimension, beta, device, obs=None, loc=None):
        self.device = device
        self.dimension = dimension
        self.t = torch.linspace(0, 1, dimension+1)[1:].to(device)
        self.dt = torch.diff(torch.linspace(0, 1, dimension+1)).to(device)
        self.beta = beta
        self.obs = obs
        self.loc = loc

    def set_noise_covariance(self, noise_covariance):
        self.Gamma_noise_inv = torch.linalg.inv(noise_covariance)

    def set_C(self):
        d = len(self.dt)
        self.C = torch.zeros((d, d), device=self.device)
        for i in range(d):
            for j in range(d):
                self.C[i, j] = torch.minimum(self.t[i], self.t[j])
        self.C_inv = torch.linalg.inv(self.C)

    def solution(self, x):
        x = torch.cat(
            (torch.zeros((x.shape[0], 1), device=self.device), x), 
            axis=1
        ).to(self.device)
        dx = torch.diff(x, dim=1)
        u = torch.zeros_like(x, device=self.device)
        for i in range(1, x.shape[1]):
            u[:, i] = u[:, i - 1] + self.beta * u[:, i - 1] * (1 - u[:, i - 1] ** 2) / (1 + u[:, i - 1] ** 2) * self.dt[i - 1] + dx[:, i - 1]

        return u

    def forward(self, x):
        x = torch.cat(
            (torch.zeros((x.shape[0], 1), device=self.device), x), 
            axis=1
        ).to(self.device) # n x (dim+1)
        dx = torch.diff(x, dim=1) # n x dim
        u = 0.
        d = x.shape[1]
        f = torch.zeros((x.shape[0], d), device=self.device)
        for i in range(1, d):
            u = u + self.beta * u * (1 - u ** 2) / (1 + u ** 2) * self.dt[i - 1] + dx[:, i - 1]
            f[:, i] = u
        y = f[:, self.loc] # n x nobs

        return y

    def log_prob(self, x):
        u = self.forward(x)
        diff = self.obs - u # n x nobs
        log_lik = - 0.5 * torch.diagonal(diff @ self.Gamma_noise_inv @ diff.T)  # n
        log_p = - 0.5 * torch.diagonal(x @ self.C_inv @ x.T) # n
        res = (log_lik + log_p).reshape((-1, 1)) # n x 1

        return res

    def brownian_motion(self, size):
        mean = torch.zeros(size[1], device=self.device)
        x = MultivariateNormal(loc=mean, covariance_matrix=self.C).sample((size[0],))

        return x
