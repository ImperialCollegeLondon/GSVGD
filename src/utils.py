import torch
import numpy as np
import matplotlib.pyplot as plt

class GaussianMixture(torch.distributions.Distribution):
    def __init__(
        self, mean: np.array, covariance_matrix: np.array, prob: np.array
    ) -> None:
        self.num_components = mean.size(0)
        self.mu = mean
        self.covariance_matrix = covariance_matrix
        self.prob = prob

        self.dists = [
            torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
            for mu, sigma in zip(mean, covariance_matrix)
        ]

        super().__init__(
            torch.Size([]), torch.Size([mean.size(-1)])
        )

    def log_prob(self, value: np.ndarray):
        return torch.cat(
            [
                p * d.log_prob(value).unsqueeze(-1)
                for p, d in zip(self.prob, self.dists)
            ],
            dim=-1,
        ).logsumexp(dim=-1)

    def enumerate_support(self):
        return self.dists[0].enumerate_support()


def plot_particles(x_init, x_final, P, d=7.0, step=0.1, concat=None, savedir=None, figsize=(12, 6)):
    '''
    Args:
        pad: If in dim > 2, then concat the 2d grid with specified tensor (of 
            shape (1, dim - 2)) when calculating the log likelihood.
    '''
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    xv, yv = xv.to(x_init.device), yv.to(x_init.device)
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1).to(x_init.device)
    if P.event_shape[0] > 2:
        # assert concat is not None, "Need to specify concat in dim larger than 2."
        if concat is None:
            print(("Concatenating higher dims with vector of zeros. Customize by "
                "using the concat argument."))
            concat = torch.zeros((1, P.event_shape[0] - 2)).to(x_init.device)

        pos_xy = torch.cat(
            (pos_xy, concat.repeat(pos_xy.shape[0], pos_xy.shape[1], 1)), 
            dim=2).to(x_init.device)
    p_xy = P.log_prob(pos_xy).exp().unsqueeze(-1)
    xv, yv, x_init, x_final, p_xy = xv.to("cpu"), yv.to("cpu"), x_init.to("cpu"), x_final.to("cpu"), p_xy.to("cpu")
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.contourf(xv, yv, p_xy.squeeze(-1), levels=20)
    plt.scatter(x_init[:, 0], x_init[:, 1], color="r", alpha=0.5, s=6)
    plt.title("Initial")
    plt.subplot(1, 2, 2)
    plt.contourf(xv, yv, p_xy.squeeze(-1), levels=20)
    plt.scatter(x_final[:, 0], x_final[:, 1], color="r", alpha=0.5, s=6)
    plt.title("Final")
    if savedir:
        fig.savefig(savedir)
    return fig
