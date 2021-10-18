### Code modified from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
import torch
import torch.nn.functional as F
import sys

import torch.autograd as autograd
from tqdm import tqdm
class SVGD:
    def __init__(
        self,
        target: torch.distributions.Distribution,
        kernel: torch.nn.Module,
    ):
        """
        Args:
            target (torch.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
            kernel (torch.nn.Module): [description]
        """
        self.p = target
        self.k = kernel

    def phi(self, X: torch.Tensor, y):
        """
        Args:
            X (torch.Tensor): Particles being transported to the target distribution
            Size: batch x num_particles x dim
        Returns:
            phi (torch.Tensor): Functional gradient
            Size: batch x num_particles x dim
        """
        # copy the data for X into X
        X = X.detach().requires_grad_(True)

        log_prob = self.p.log_prob(y, X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.k(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        return phi

    def step(self, X: torch.Tensor, y):
        """Gradient descent step
        Args:
            X (torch.Tensor): Particles to transport to the target distribution
        """
        X -= 0.1*self.phi(X, y)
        return X

    def fit(self, x0: torch.Tensor, y, epochs: torch.int64):
        """
        Args:
            x0 (torch.Tensor): Initial set of particles to be updated
            epochs (torch.int64): Number of gradient descent iterations
        """
        y = y.detach().requires_grad_(False)
        for i in range(epochs):
            x0 = self.step(x0, y)
        return x0

class MLP(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.output_loss = torch.nn.BCELoss()

    def encode(self, x, noise):
        x = torch.cat([x, noise], axis=1)
        z = self.encoder(x)
        return z

    def sample_latent(self, x, num_samples=1):
        z = torch.empty(size=(num_samples, x.shape[0], self.latent_dim))
        for i in range(num_samples): 
            noise = torch.normal(0, 1, (x.shape[0], self.latent_dim))
            z[i] = self.encode(x, noise)
        return z

    def log_prob(self, y, z):
        """[summary]

        Args:
            z ([type]): [description]
        """
        # TODO: ??? what's the prior? Uniform??
        # how do you calculate the logprob then?

        # z is of dimension num_samples x num_data x latent_dim
        # repeat x along the 0th dimension
        # x is now of dimension num_samples x num_data x xdim
        # we now output the log_prob of the particles
        # num_data
        return self.output_loss(self.forward(z), y[None, :].repeat(z.shape[0], 1))

    def forward(self, z):
        return self.decoder(z).view(z.shape[0], -1)

if __name__ == '__main__':
    num_batch = 100
    num_latent = 50
    latent_dim = 32
    x_dim = 1024
    D_in = x_dim + latent_dim
    H = 300
    encoder = MLP(D_in, H, latent_dim)
    decoder = MLP(latent_dim, H, 1)
    vae = VAE(encoder, decoder, latent_dim)

    x = torch.ones((num_batch, x_dim))
    y = torch.randn((num_batch))
    noise = torch.ones((num_batch, latent_dim))
    import sys
    sys.path.append("/home/hbz/Documents/work/code/M-SVGD/")
    from src.kernel import RBF
    import torch.optim as optim

    k = RBF()
    svgd = SVGD(vae, k)
    
    z = vae.sample_latent(x, num_samples=num_latent)
    for i in range(num_latent):
        z[:, i, :] = svgd.fit(z[:, i, :], y[None, i], epochs=50)
    
    adam_encoder = optim.Adam(encoder.parameters())
    adam_decoder = optim.Adam(decoder.parameters())

    # from experiments.synthetic import GaussianMixture
    # from src.synthetic import get_density_chart, get_particles_chart
    ## Setup target parameters
    # num_particles = 10
    # mean = torch.Tensor([[-2.0, 0.0], [4.0, 0.0]])
    # covariance = torch.Tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(2, 1, 1)
    # prob = torch.Tensor([[1 / 3], [2 / 3]])
    # gaussian_mix = GaussianMixture(mean=mean, covariance_matrix=covariance, prob=prob)
    # svgd = SVGD(gaussian_mix, k)
    # x_init = torch.randn(num_particles, *gaussian_mix.event_shape)
    # print(x_init.shape)
    # x = svgd.fit(x_init, epochs=50)
    # fig = get_density_chart(gaussian_mix, d=12.0, step=0.1)

    # fig = (fig + get_particles_chart(x_init.cpu().numpy())) | (
    #     fig + get_particles_chart(x.cpu().numpy())
    # )
    # fig.save("test.html")


