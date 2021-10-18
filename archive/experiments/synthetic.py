import torch
import numpy as np
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

