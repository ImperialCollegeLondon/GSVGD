import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD
from src.diffusion import Diffusion
import pickle
import argparse
import time

parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--dim", type=int, default=100, help="dimension")
parser.add_argument("--effdim", type=int, default=-1, help="dimension")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--lr_g", type=float, default=0.1, help="learning rate for S-SVGD")
parser.add_argument(
    "--delta", type=float, default=0.1, help="stepsize for projections"
)
parser.add_argument(
    "--T", type=float, default=1e-4, help="noise multiplier for projections"
)
parser.add_argument("--nparticles", type=int, default=50, help="no. of particles")
parser.add_argument("--epochs", type=int, default=2000, help="no. of epochs")
parser.add_argument("--noise", type=str, default="True", help="whether to add noise")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--suffix", type=str, default="", help="suffix for res folder")
parser.add_argument("--method", type=str, default="svgd", help="svgd, gsvgd or s-svgd")
parser.add_argument("--save_every", type=int, default=100, help="batch size")

args = parser.parse_args()
dim = args.dim
lr = args.lr
lr_g = args.lr_g
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs
seed = args.seed
eff_dims = [args.effdim] if args.effdim > 0 else [1, 2, 5]
add_noise = True if args.noise == "True" else False
save_every = args.save_every  # save metric values every 100 epochs
print(f"Running for lr: {lr}, nparticles: {nparticles}")

device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

results_folder = f"./res/diffusion{args.suffix}/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}_dim{dim}"
results_folder = f"{results_folder}/seed{seed}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF

if __name__ == "__main__":
    print(f"Device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## initialize conditional diffusion
    beta = 10
    sigma = 0.1

    distribution = Diffusion(dim, beta, device=device)
 
    loc = torch.arange(0, dim+1, 5)[1:]
    distribution.loc = loc
    noise_covariance = torch.diag(sigma**2 * torch.ones(len(loc), device=device))
    distribution.set_noise_covariance(noise_covariance)
    distribution.set_C()

    x_true = distribution.brownian_motion((1, dim))
    u_true = distribution.solution(x_true)

    obs_noise = torch.normal(0, sigma, size=(1, len(loc))).to(device)
    obs = u_true[:, loc] + obs_noise
    distribution.obs = obs

    # initialize particles
    x0 = distribution.brownian_motion((nparticles, dim))

    if args.method == "svgd":
        ## SVGD
        print("Running SVGD")
        x = x0.clone().requires_grad_()
        # sample from variational density
        kernel = Kernel(method="med_heuristic")
        svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)

        start = time.time()
        svgd.fit(x0=x, epochs=epochs, save_every=save_every)
        elapsed_time = time.time() - start

        fitted_method = svgd
        particles = fitted_method.particles
        method_name = "svgd"

    elif args.method == "gsvgd":
        eff_dim = args.effdim
        print(f"Running GSVGD with eff dim = {eff_dim}")
        
        m = min(20, dim // eff_dim)
        print("number of projections:", m)

        # sample from variational density
        x_gsvgd = x0.clone().requires_grad_()

        kernel_gsvgd = BatchKernel(method="med_heuristic")
        optimizer = optim.Adam([x_gsvgd], lr=lr)
        manifold = Grassmann(dim, eff_dim)
        U = torch.eye(dim).requires_grad_(True).to(device)
        U = U[:, :(m*eff_dim)]

        gsvgd = FullGSVGDBatch(
            target=distribution,
            kernel=kernel_gsvgd,
            manifold=manifold,
            optimizer=optimizer,
            delta=delta,
            T=T,
            device=device,
            noise=add_noise
        )

        start = time.time()
        U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
            verbose=True, save_every=save_every, threshold=1e-2)
        elapsed_time = time.time() - start

        fitted_method = gsvgd
        particles = fitted_method.particles
        method_name = f"gsvgd_effdim{eff_dim}"

    elif args.method == "s-svgd":
        ## S-SVGD
        # sample from variational density
        print("Running S-SVGD")
        x_s_svgd = x0.clone().requires_grad_()
        s_svgd = SlicedSVGD(distribution, device=device)

        start = time.time()
        x_s_svgd, metric_s_svgd = s_svgd.fit(
            samples=x_s_svgd, 
            n_epoch=epochs, 
            lr=lr_g,
            eps=lr,
            save_every=save_every
        )
        elapsed_time = time.time() - start

        fitted_method = s_svgd
        particles = fitted_method.particles
        method_name = f"s-svgd_lrg{lr_g}"

    elif args.method == "hmc":

        import numpyro
        from numpyro.infer import MCMC, NUTS
        import numpyro.distributions as npr_dist
        import jax.random as random
        import jax.numpy as jnp
        import jax

        C = distribution.C.cpu().numpy()
        dt = distribution.dt.cpu().numpy()
        loc = distribution.loc.cpu().numpy()
        beta = distribution.beta
        def model(Y, sigma, dim):
            mean = jnp.zeros(dim)
            x = numpyro.sample(
                "x", npr_dist.MultivariateNormal(loc=mean, covariance_matrix=C)
            ) # nparticles x dim
            u = forward(x) # nparticles x nobs
            numpyro.sample("Y", npr_dist.Normal(u, sigma), obs=Y)

        def forward(x):
            x = jnp.concatenate(
                (jnp.zeros(1), x)
            ) # n x (dim+1)
            dx = jnp.diff(x) # n x dim
            u = 0.
            d = x.shape[0]
            f = jnp.zeros(d)
            for i in range(1, d):
                u = u + beta * u * (1 - u ** 2) / (1 + u ** 2) * dt[i - 1] + dx[i - 1]
                f = jax.ops.index_update(f, i, u)
            y = f[loc] # n x nobs
            return y

        def run_inference(model, rng_key, Y, sigma, dim):
            start = time.time()
            kernel = NUTS(model)
            mcmc = MCMC(
                kernel,
                num_warmup=2000,
                num_samples=2000,
                num_chains=1,
                progress_bar=True,
            )
            mcmc.run(rng_key, Y, sigma, dim)
            mcmc.print_summary()
            elapsed_time = time.time() - start
            print("\nMCMC elapsed time:", elapsed_time)
            return mcmc, mcmc.get_samples(), elapsed_time

        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        Y = distribution.obs.cpu().numpy()
        mcmc, particles_dict, elapsed_time = run_inference(model, rng_key, Y, sigma, dim)
        particles = particles_dict["x"]
        method_name = "hmc"


    ## save results
    pickle.dump(
        {
            **{"x_true": x_true},
            **{"u_true": u_true},
            **{"particles": particles},
            **{"time": elapsed_time}
        },
        open(results_folder + f"/particles_{method_name}.p", "wb")
    )

    # target distribution
    torch.save(distribution, results_folder + '/target_dist.p')
    
    