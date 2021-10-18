import os
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
device = "cuda:2"

from src.svgd import SVGD
from src.gsvgd import GSVGD
from src.kernel import RBF, IMQ, SumKernel
from src.utils import plot_particles, plot_metrics
from src.metrics import Metric
from src.manifold import Grassmann
from src.maxsvgd import MaxSVGD

import pickle
import argparse


def xshaped_gauss_experiment(mixture_dist, means, correlation):
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
    cov2 = torch.eye(dim, device=device)
    cov2[:2, :2] = torch.Tensor([[1, -correlation], [-correlation, 1]])
    mix_cov = torch.stack((cov1, cov2))
    comp = D.MultivariateNormal(means.to(device), mix_cov)

    distribution = D.mixture_same_family.MixtureSameFamily(mixture_dist, comp)   
    return(distribution)


parser = argparse.ArgumentParser(description='Running xshaped experiment.')
parser.add_argument('--dim', type=int, help='dimension')
parser.add_argument('--effdim', type=int, default=2, help='effective dimension for GSVGD')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--nparticles', type=int, help='no. of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')

args = parser.parse_args()
dim = args.dim
lr = args.lr
nparticles = args.nparticles
epochs = args.epochs
eff_dims = [args.effdim] if args.effdim > 0 else [1, 2, 5, 10]
add_noise = True if args.noise == "True" else False
correlation = 0.95
save_every = 500 # save metric values
print(f"Running for dim: {dim}, lr: {lr}, nparticles: {nparticles}")

metric = args.metric

results_folder = f"./res/xshaped/{args.kernel}_epochs{epochs}_lr{lr}_samples{nparticles}_dim{dim}"
if add_noise:
    results_folder += "_noise"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
elif args.kernel == "imq":
    Kernel = IMQ
elif args.kernel == "sum":
    Kernel = SumKernel

if __name__ == "__main__":
    print(f"Device: {device}")
    torch.manual_seed(0)

    ## target density
    mix_means = torch.zeros((2, dim)).to(device)
    mix_means[:, :2] = 1

    distribution = xshaped_gauss_experiment(
        mixture_dist=D.Categorical(torch.ones(mix_means.shape[0],).to(device)),
        means=mix_means,
        correlation=correlation
    )

    # sample from target (for computing metric)
    x_target = distribution.sample((nparticles, )).to(device)
    # sample from variational density
    x_init = torch.randn(nparticles, *distribution.event_shape).to(device)
    # initialize metric
    metric_fn = Metric(metric=metric, x_init=x_init.clone, x_target=x_target)

    # ## SVGD
    # print("Running SVGD")
    # x = x_init.clone().to(device)
    # kernel = Kernel(method="med_heuristic")
    # svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)
    # metric_svgd = svgd.fit(x, epochs, verbose=True, metric=metric_fn, save_every=save_every)

    # # plot particles
    # fig_svgd = plot_particles(
    #     x_init.detach(), 
    #     x.detach(), 
    #     distribution, 
    #     d=6.0, 
    #     step=0.1, 
    #     concat=mix_means[0, 2:],
    #     savedir=results_folder + f"/svgd.pdf"
    # )


    ## GSVGD
    # effective dim cannot exceed dim
    eff_dims = [d for d in eff_dims if d <= dim]
    res_gsvgd = [0] * len(eff_dims)
    def run_gsvgd(eff_dims):
        # effective dim cannot exceed dim
        eff_dim = [d for d in eff_dims if d <= dim]
        for i, eff_dim in enumerate(eff_dims):
            print(f"Running GSVGD with eff dim = {eff_dim}")
            # sample from variational density
            x_init_gsvgd = x_init.clone()
            x_gsvgd = x_init_gsvgd.clone()

            kernel_gsvgd = Kernel(method="med_heuristic")
            optimizer = optim.Adam([x_gsvgd], lr=lr)
            manifold = Grassmann(dim, eff_dim)
            A = torch.eye(dim)[:, :eff_dim].requires_grad_(True).to(device)

            gsvgd = GSVGD(
                target=distribution,
                kernel=kernel_gsvgd,
                manifold=manifold,
                optimizer=optimizer,
                delta=1e-5,
                device=device,
                noise=add_noise
            )
            A, metric_gsvgd = gsvgd.fit(x_gsvgd, A, epochs, projection_epochs=1, 
                verbose=True, metric=metric_fn, save_every=save_every)

            # plot particles
            fig_gsvgd = plot_particles(
                x_init_gsvgd.detach(), 
                x_gsvgd.detach(), 
                distribution, 
                d=6.0, 
                step=0.1, 
                concat=mix_means[0, 2:].to(device),
                savedir=results_folder + f"/gsvgd_effdim{eff_dim}.png"
            )

            # store results
            res_gsvgd[i] = {"effdim":eff_dim, "init":x_init_gsvgd, "final":x_gsvgd, "metric":metric_gsvgd, 
                "fig":fig_gsvgd, "particles":gsvgd.particles}
        return res_gsvgd

    res_gsvgd = run_gsvgd(eff_dims)


    ## MaxSVGD
    # sample from variational density
    print("Running maxSVGD")
    x_init_maxsvgd = x_init.clone()
    x_maxsvgd = x_init_maxsvgd.clone().requires_grad_()
    maxsvgd = MaxSVGD(distribution, device=device)

    x_maxsvgd, metric_maxsvgd = maxsvgd.fit(
        samples=x_maxsvgd, 
        n_epoch=epochs, 
        lr=0.1,
        eps=lr,
        metric=metric_fn,
        save_every=save_every
    )

    # plot particles
    fig_maxsvgd = plot_particles(
        x_init_maxsvgd.detach(), 
        x_maxsvgd.detach(), 
        distribution, 
        d=6.0, 
        step=0.1, 
        concat=mix_means[0, 2:],
        savedir=results_folder + f"/maxsvgd.png"
    )


    ## save results and figs
    # particles
    particles_epochs = np.arange(0, epochs+save_every, save_every)
    pickle.dump(
        {
            **{"target_dist": distribution},
            **{"epochs": particles_epochs},
            **{"effdims": eff_dims},
            **{"target": x_target.cpu()},
            **{"svgd": svgd.particles},
            **{f"gsvgd_effdim{d}": r["particles"] for d, r in zip(eff_dims, res_gsvgd)},
            **{"maxsvgd": maxsvgd.particles}
        },
        open(results_folder + "/particles.p", "wb")
    )

    # save and plot metrics
    if metric in ["mmd_both", "energy_both"]:
        metrics_df = pd.DataFrame(
            {
                **{"epochs": particles_epochs[1:]},
                **{"svgd_full": [r[0] for r in metric_svgd]}, 
                **{"svgd_sub": [r[1] for r in metric_svgd]}, 
                **{f"gsvgd_full_{d}":np.array(r["metric"])[:, 0] for d, r in zip(eff_dims, res_gsvgd)},
                **{f"gsvgd_sub_{d}":np.array(r["metric"])[:, 1] for d, r in zip(eff_dims, res_gsvgd)},
                **{"maxsvgd_full": [r[0] for r in metric_maxsvgd]},
                **{"maxsvgd_sub": [r[1] for r in metric_maxsvgd]}
            }
        )
        metrics_df.to_csv(
            results_folder + f'/{metric}.csv',
            index=False
        )
            
        plot_metrics(
            epochs=metrics_df.epochs,
            metric_svgd=[s[0] for s in metric_svgd],
            metric_gsvgd=np.array([np.array(r["metric"])[:, 0] for r in res_gsvgd]).T, 
            eff_dims=eff_dims,
            metric_maxsvgd=[s[0] for s in metric_maxsvgd],
            name=metric.split("_")[0] + "_full",
            savedir=results_folder
        )
        plot_metrics(
            epochs=metrics_df.epochs,
            metric_svgd=[s[1] for s in metric_svgd],
            metric_gsvgd=np.array([np.array(r["metric"])[:, 1] for r in res_gsvgd]).T, 
            eff_dims=eff_dims,
            metric_maxsvgd=[s[1] for s in metric_maxsvgd],
            name=metric.split("_")[0] + "_sub",
            savedir=results_folder
        )
    else:
        metrics_df = pd.DataFrame(
           {
                **{"epochs": particles_epochs[1:]},
                **{"svgd": metric_svgd},
                **{f"gsvgd_{d}":r["metric"] for d, r in zip(eff_dims, res_gsvgd)},
                **{"maxsvgd": metric_maxsvgd},
            }
        )
        metrics_df.to_csv(
            results_folder + f'/{metric}.csv',
            index=False
        )

        plot_metrics(
            epochs=metrics_df.epochs,
            metric_svgd=metric_svgd,
            metric_gsvgd=np.array([np.array(r["metric"]) for r in res_gsvgd]).T, 
            eff_dims=eff_dims,
            metric_maxsvgd=metric_maxsvgd,
            name=metric,
            savedir=results_folder
        )

