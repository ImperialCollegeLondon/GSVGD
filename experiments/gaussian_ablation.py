import os
import numpy as np
import torch
import torch.optim as optim
import torch.distributions as D

from src.svgd import SVGD
from src.gsvgd import FullGSVGDBatch
from src.kernel import RBF, BatchRBF
from src.utils import plot_particles
from src.metrics import Metric
from src.manifold import Grassmann
from src.s_svgd import SlicedSVGD

import pickle
import argparse
import time

parser = argparse.ArgumentParser(description='Running xshaped experiment.')
parser.add_argument('--dim', type=int, help='dimension')
parser.add_argument('--effdim', type=int, default=-1, help='dimension')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--T', type=float, default=1e-4, help='noise multiplier for projections')
parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for g')
parser.add_argument('--nparticles', type=int, help='no. of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--metric', type=str, default="energy", help='distance metric')
parser.add_argument('--noise', type=str, default="True", help='whether to add noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--suffix', type=str, default="", help='suffix for res folder')
parser.add_argument('--m', type=int, help='no. of projections')
parser.add_argument('--save_every', type=int, default=200, help='step intervals to save particles')
parser.add_argument('--method', type=str, default="all", help='which method to use')

args = parser.parse_args()
dim = args.dim
lr = args.lr
lr_gsvgd = args.lr
delta = args.delta
T = args.T
nparticles = args.nparticles
epochs = args.epochs
seed = args.seed
eff_dims = [args.effdim] if args.effdim > 0 else [1, 2, 5]
save_every = args.save_every # save metric values
print(f"Running for dim: {dim}, lr: {lr}, nparticles: {nparticles}")

device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')

metric = args.metric

results_folder = f"./res/gaussian{args.suffix}/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}_dim{dim}"
results_folder = f"{results_folder}/seed{seed}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF

if __name__ == "__main__":
    print(f"Device: {device}")
    torch.manual_seed(seed)

    ## target density
    means = torch.zeros(dim, device=device)
    cov = torch.eye(dim, device=device)
    distribution = D.MultivariateNormal(means.to(device), cov)

    # sample from target (for computing metric)
    x_target = distribution.sample((nparticles, ))
    # sample from variational density
    x_init = 2 + np.sqrt(2) * torch.randn(nparticles, *distribution.event_shape, device=device)
    # initialize metric
    metric_fn = Metric(metric=metric, x_init=x_init.clone, x_target=x_target)


    ## SVGD
    if args.method == "SVGD":
        print("Running SVGD")
        # sample from variational density
        x = x_init.clone().to(device)
        kernel = Kernel(method="med_heuristic")
        svgd = SVGD(distribution, kernel, optim.Adam([x], lr=lr), device=device)
        start = time.time()
        svgd.fit(x, epochs, verbose=True, save_every=save_every)
        elapsed_time_svgd = time.time() - start

        # plot particles
        fig_svgd = plot_particles(
            x_init.detach(), 
            x.detach(), 
            distribution, 
            d=6.0, 
            step=0.1, 
            concat=means[2:],
            savedir=results_folder + f"/svgd.png"
        )

        pickle.dump({"svgd": svgd.particles}, open(results_folder + "/particles_svgd.p", "wb"))


    ## GSVGD
    elif args.method == "GSVGD":
        res_gsvgd = [0] * len(eff_dims)
        def run_gsvgd(eff_dims):
            for i, eff_dim in enumerate(eff_dims):
                print(f"Running GSVGD with eff dim = {eff_dim}")
                m = args.m

                print("number of projections:", m)

                # sample from variational density
                x_init_gsvgd = x_init.clone()
                x_gsvgd = x_init_gsvgd.clone()

                # kernel_gsvgd = RBF(method="med_heuristic")
                kernel_gsvgd = BatchKernel(method="med_heuristic")
                optimizer = optim.Adam([x_gsvgd], lr=lr_gsvgd)
                manifold = Grassmann(dim, eff_dim)
                U = torch.eye(dim).requires_grad_().to(device)
                U = U[:, :(m*eff_dim)]

                gsvgd = FullGSVGDBatch(
                    target=distribution,
                    kernel=kernel_gsvgd,
                    manifold=manifold,
                    optimizer=optimizer,
                    delta=delta,
                    T=T,
                    device=device
                )
                start = time.time()
                U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
                    verbose=True, save_every=save_every, threshold=0.0001*m)
                elapsed_time = time.time() - start

                # plot particles
                fig_gsvgd = plot_particles(
                    x_init_gsvgd.detach(), 
                    x_gsvgd.detach(), 
                    distribution, 
                    d=6.0, 
                    step=0.1, 
                    concat=means[2:],
                    savedir=results_folder + f"/fullgsvgd_effdim{eff_dim}_lr{lr_gsvgd}_delta{delta}_m{m}_T{T}.png"
                )

                # store results
                res_gsvgd[i] = {"init":x_init_gsvgd, "final":x_gsvgd, "metric":metric_gsvgd, 
                    "fig":fig_gsvgd, "particles":gsvgd.particles, "pam":gsvgd.pam, "res": gsvgd,
                    "elapsed_time": elapsed_time}
            return res_gsvgd

        res_gsvgd = run_gsvgd(eff_dims)

        pickle.dump(
            {f"gsvgd_effdim{d}": r["particles"] for d, r in zip(eff_dims, res_gsvgd)},
            open(results_folder + f"/particles_gsvgd_m{eff_dims[0]}_M{args.m}.p", "wb")
        )


    ## S-SVGD
    elif args.method == "S-SVGD":
        print("Running S-SVGD")
        # sample from variational density
        x_init_s_svgd = x_init.clone()
        x_s_svgd = x_init_s_svgd.clone().requires_grad_()
        s_svgd = SlicedSVGD(distribution, device=device)

        start = time.time()
        x_s_svgd, metric_s_svgd = s_svgd.fit(
            samples=x_s_svgd, 
            n_epoch=epochs, 
            lr=args.lr_g,
            eps=lr,
            save_every=save_every
        )
        elapsed_time_s_svgd = time.time() - start

        # plot particles
        fig_s_svgd = plot_particles(
            x_init_s_svgd.detach(), 
            x_s_svgd.detach(), 
            distribution, 
            d=6.0, 
            step=0.1, 
            concat=means[2:],
            savedir=results_folder + f"/ssvgd_lr{lr}_lrg{args.lr_g}.png"
        )

        pickle.dump({"s_svgd": s_svgd.particles}, open(results_folder + "/particles_s-svgd.p", "wb"))


    ## save results and figs
    particles_epochs = np.arange(0, epochs+save_every, save_every)
    pickle.dump(
        {
            **{"epochs": particles_epochs},
            **{"effdims": eff_dims},
            **{"target": x_target.cpu()},
        },
        open(results_folder + "/particles.p", "wb")
    )

    # target distribution
    torch.save(distribution, results_folder + '/target_dist.p')