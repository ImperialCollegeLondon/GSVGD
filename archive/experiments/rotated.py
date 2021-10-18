import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
import torch.distributions as D
import seaborn as sns
from tqdm import tqdm
from geomloss import SamplesLoss
import sys
sys.path.append(".")

print(torch.cuda.device_count())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.svgd import SVGD
from src.full_gsvgd import FullGSVGD
from src.kernel import RBF, IMQ, SumKernel, BatchRBF
from src.utils import plot_particles, plot_metrics, plot_particles_hist
from src.metrics import Metric
from src.manifold import Grassmann
from src.maxsvgd import MaxSVGD

import pickle
import argparse


def xshaped_gauss_experiment(mixture_dist, means, correlation, R):
    '''Mixture of Multivariate gaussian with cov matrices being the identity.
    Args:
        mixture_dist: torch.distributions.Categorical-like instance for the 
            probability of each component in the mixture.
        means: Tensor of shape (nmix, d), where nmix is the number of components 
            and d is the dimension of each component.
        correlation: Float between 0 and 1 for the magnitude of correlation between
            the first two dims.
    '''
    nmix, dim = means.shape
    
    # create multibatch multivariate Gaussian
    cov1 = torch.eye(dim, device=device)
    cov1[:2, :2] = torch.Tensor([[1, correlation], [correlation, 1]])
    # cov2 = torch.eye(dim, device=device)
    # cov2[:2, :2] = torch.Tensor([[1, -correlation], [-correlation, 1]])
    # mix_cov = torch.stack((R @ cov1 @ R.t(), R @ cov2 @ R.t()))
    # comp = D.MultivariateNormal(means.to(device) @ R.t(), mix_cov)

    # distribution = D.mixture_same_family.MixtureSameFamily(mixture_dist, comp)   
    distribution = D.MultivariateNormal(means[0, :].to(device) @ R.t(), R @ cov1 @ R.t())
    return(distribution)


parser = argparse.ArgumentParser(description='Running xshaped experiment.')
parser.add_argument('--dim', type=int, help='dimension')
parser.add_argument('--effdim', type=int, default=-1, help='dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--nparticles', type=int, help='no. of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()
dim = args.dim
lr = args.lr
delta = args.delta
nparticles = args.nparticles
epochs = args.epochs
eff_dims = [args.effdim] if args.effdim > 0 else [1, 2, 5]
add_noise = True if args.noise == "True" else False
correlation = 0.8
save_every = 100# 250 # save metric values
print(f"Running for dim: {dim}, lr: {lr}, nparticles: {nparticles}")

device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')

metric = args.metric

results_folder = f"./res/rotated_xshaped/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}_dim{dim}"
if add_noise:
    results_folder += "_noise"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF
elif args.kernel == "imq":
    Kernel = IMQ

if __name__ == "__main__":
    print(f"Device: {device}")
    torch.manual_seed(1)

    rotation = torch.nn.init.orthogonal_(
        torch.empty(dim, dim, device=device)
    )

    ## target density
    mix_means = torch.zeros((2, dim), device=device)
    mix_means[:, :2] = 1
    # mix_means[:, :] = 2

    distribution = xshaped_gauss_experiment(
        mixture_dist=D.Categorical(torch.ones(mix_means.shape[0], device=device)),
        means=mix_means,
        correlation=correlation,
        R=rotation
    )

    # sample from target (for computing metric)
    x_target = distribution.sample((nparticles, ))
    # sample from variational density
    x_init = torch.randn(nparticles, *distribution.event_shape, device=device)
    # x_init[:, 2:] = (x_init[2:, 2:] - 2) / np.sqrt(2)
    # initialize metric
    metric_fn = Metric(metric=metric, x_init=x_init.clone, x_target=x_target)


    ## SVGD
    print("Running SVGD")
    # sample from variational density
    x = x_init.clone().to(device)
    kernel = Kernel(method="med_heuristic")
    svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)
    metric_svgd = svgd.fit(x, epochs, verbose=True, metric=metric_fn, save_every=save_every)

    # plot particles
    fig_svgd = plot_particles(
        # x_init.detach(), 
        x_target @ rotation,
        x.detach() @ rotation, 
        distribution, 
        d=6.0, 
        step=0.1, 
        concat=mix_means[0, 2:],
        savedir=results_folder + f"/svgd.png"
    )


    ## GSVGD
    res_gsvgd = [0] * len(eff_dims)
    # #! hard coded
    # lr_ls = [0.01, 0.1, 0.5]
    def run_gsvgd(eff_dims):
        for i, eff_dim in enumerate(eff_dims):
            # lr = lr_ls[i] #! hard coded
            print(f"Running GSVGD with eff dim = {eff_dim}")
            # m = dim // eff_dim
            m = 5
            # m = 5
            # m = (dim // eff_dim) if dim <= 30 and eff_dim == 5 else 5

            # sample from variational density
            x_init_gsvgd = x_init.clone()
            x_gsvgd = x_init_gsvgd.clone()

            kernel_gsvgd = BatchKernel(method="med_heuristic")
            optimizer = optim.Adam([x_gsvgd], lr=lr)
            manifold = Grassmann(dim, eff_dim)
            # U = torch.eye(dim).requires_grad_(True).to(device)
            # U = U[:, 2:(2+m*eff_dim)]
            U = torch.nn.init.orthogonal_(
                torch.empty(dim, m*eff_dim)
            ).requires_grad_(True).to(device)

            gsvgd = FullGSVGD(
                target=distribution,
                kernel=kernel_gsvgd,
                manifold=manifold,
                optimizer=optimizer,
                delta=delta,
                device=device,
                noise=add_noise
            )
            U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
                verbose=True, metric=metric_fn, save_every=save_every)

            # plot particles
            fig_gsvgd = plot_particles(
                # x_init_gsvgd.detach(),
                x_target @ rotation, 
                x_gsvgd.detach() @ rotation, 
                distribution, 
                d=6.0, 
                step=0.1, 
                concat=mix_means[0, 2:].to(device),
                savedir=results_folder + f"/fullgsvgd_effdim{eff_dim}_lr{lr}_delta{delta}.png"
            )

            # store results
            res_gsvgd[i] = {"init":x_init_gsvgd, "final":x_gsvgd, "metric":metric_gsvgd, 
                "fig":fig_gsvgd, "particles":gsvgd.particles, "pam":gsvgd.pam, "fro":gsvgd.U_diff_fro, "res": gsvgd}
        return res_gsvgd

    res_gsvgd = run_gsvgd(eff_dims)


    ## MaxSVGD
    print("Running maxSVGD")
    # sample from variational density
    x_init_maxsvgd = x_init.clone()
    x_maxsvgd = x_init_maxsvgd.clone().requires_grad_()
    maxsvgd = MaxSVGD(distribution, device=device)

    # #! hard coded
    # lr = 0.01
    lr_g = 0.1
    x_maxsvgd, metric_maxsvgd = maxsvgd.fit(
        samples=x_maxsvgd, 
        n_epoch=epochs, 
        lr=lr_g, # 0.1
        eps=lr,
        metric=metric_fn,
        save_every=save_every
    )

    # plot particles
    fig_maxsvgd = plot_particles(
        # x_init_maxsvgd.detach(), 
        x_target @ rotation,
        x_maxsvgd.detach() @ rotation, 
        distribution, 
        d=6.0, 
        step=0.1, 
        concat=mix_means[0, 2:],
        savedir=results_folder + f"/ssvgd_lr{lr}_lrg{lr_g}.png"
    )


    ## save results and figs
    # particle-averaged magnitudes
    pam = {
        **{"svgd": svgd.pam},
        **{f"gsvgd_effdim{d}": r["pam"] for d, r in zip(eff_dims, res_gsvgd)},
        **{"maxsvgd": maxsvgd.pam}
    }
    fro = {
        **{f"gsvgd_effdim{d}": r["fro"] for d, r in zip(eff_dims, res_gsvgd)},
    }

    # particles
    particles_epochs = np.arange(0, epochs+save_every, save_every)
    pickle.dump(
        {
            **{"epochs": particles_epochs},
            **{"effdims": eff_dims},
            **{"target": x_target.cpu()},
            **{"svgd": svgd.particles},
            **{f"gsvgd_effdim{d}": r["particles"] for d, r in zip(eff_dims, res_gsvgd)},
            **{"maxsvgd": maxsvgd.particles},
            **{"pam": pam},
            **{"fro": fro},
        },
        open(results_folder + "/particles.p", "wb")
    )

    # target distribution
    torch.save(distribution, results_folder + '/target_dist.p')
    torch.save(rotation, results_folder + '/rotation.p')

    # gsvgd
    torch.save(
        {f"gsvgd_effdim{d}": r["res"] for d, r in zip(eff_dims, res_gsvgd)},
        results_folder + '/gsvgd.p'
    )
