import sys

sys.path.append(".")
import os


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.Sliced_KSD_Clean.Divergence.Network import Bayesian_NN_eff
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import json
from src.Sliced_KSD_Clean.Util import *
from src.Sliced_KSD_Clean.Divergence.Network import *
from src.Sliced_KSD_Clean.Divergence.Def_Divergence import *
from src.Sliced_KSD_Clean.Divergence.Kernel import *
from src.Sliced_KSD_Clean.Divergence.Dataloader import *
import pickle
from src.kernel import RBF, IMQ, BatchRBF
from src.manifold import Grassmann
from src.gsvgd import FullGSVGDBatchBNN, FullGSVGDBatch
from src.svgd import SVGDBNN, SVGD

parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_g", type=float, default=0.001, help="learning rate for S-SVGD")
parser.add_argument(
    "--delta", type=float, default=0.01, help="stepsize for projections"
)
parser.add_argument("--nparticles", type=int, default=50, help="no. of particles")
parser.add_argument("--noise", type=str, default="True", help="whether to add noise")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--dataset", type=str, default="boston_housing", help="dataset")
parser.add_argument("--seed", type=int, default=2, help="seed")
parser.add_argument("--method", type=str, default="SVGD", help="SVGD Method")
parser.add_argument("--gpu", type=int, default=0, help="SVGD Method")
parser.add_argument("--m", type=int, default=5, help="Number of Projections")
parser.add_argument("--M", type=int, default=15, help="Projection Dimension")
parser.add_argument("--T", type=float, default=1e-6, help="Noise Variance")
parser.add_argument("--suffix", type=str, default="", help="suffix for res folder")
parser.add_argument("--dev", type=bool, default=False, help="whether to use a valid set")

args = parser.parse_args()
lr = args.lr
lr_g = args.lr_g
nparticles = args.nparticles
add_noise = True if args.noise == "True" else False
batch_size = 100
if args.kernel == "rbf":
    Kernel = RBF
elif args.kernel == "imq":
    Kernel = IMQ
dataset = args.dataset
# Load data
# put into dataloader
seed = int(args.seed)
method = args.method
gpu = args.gpu
m = args.m
M = args.M
T = args.T

# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# epochs based on Sliced Kernelized Stein Discrepancy, Gong et al. (2021)
if dataset in ["boston_housing", "concrete", "energy", "yacht"]:
    # epochs = 4000
    # epochs = 6000
    epochs = 5000
elif dataset == "wine":
    # epochs = 1000
    epochs = 2000
elif dataset == "protein":
    # epochs = 100
    epochs = 200
else:
    epochs = 400

np.random.seed(seed)
torch.manual_seed(seed)
if dataset != "debug":
    data = np.loadtxt(f"data/{dataset}")
    X = data[:, range(data.shape[1] - 1)]
    y = data[:, data.shape[1] - 1]
else:
    d = 20
    d_eff = 10
    n = 1000
    X = np.random.normal(loc=2, scale=3, size=(n, d))
    alpha = np.random.normal(size=(d,))
    alpha[d_eff:] = 0
    y = X @ alpha + np.random.normal(size=(n,))
    print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=seed
)

if args.dev:
    # use a validation set and overload the name as X_test and y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed
    )

std_X_train = np.std(X_train, 0)
std_X_train[std_X_train == 0] = 1
mean_X_train = np.mean(X_train, 0)
mean_y_train = np.mean(y_train)
std_y_train = np.std(y_train)

X_train = (X_train - np.full(X_train.shape, mean_X_train)) / np.full(
    X_train.shape, std_X_train
)
y_train = (y_train - mean_y_train) / std_y_train

X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(
    X_test.shape, std_X_train
)
y_test_scaled = (y_test - mean_y_train) / std_y_train

# place all numpy arrays into torch.Tensor
X_train, X_test, y_train, y_test = (
    torch.from_numpy(X_train).to(device).double(),
    torch.from_numpy(X_test).to(device).double(),
    torch.from_numpy(y_train).to(device).double(),
    torch.from_numpy(y_test).to(device).double(),
)

y_test_scaled = torch.from_numpy(y_test_scaled).to(device).double()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

n_train = X_train.shape[0]
dim = X_train.shape[1]
net = Bayesian_NN_eff(dim, device=device, n_h=50)
# use same configurations as Liu and Wang (2016)
w_init = net.init_weights(nparticles, a0=1, b0=0.1)
w = w_init.clone()
dim_particles = w.shape[1]
# print(net.forward(torch.ones(50, dim, device=device).double(), w))
# print(net.log_prob(X_train, y_train, n_train, w).shape, X_train.shape)

print(
    f"Running for dataset: {dataset}, dim_particles: {dim_particles}, lr: {lr}, nparticles: {nparticles}, seed: {seed}"
)

particles = [0] * (1 + epochs // 50)
particles[0] = w.clone().detach().cpu()
# lists to store results
rmse_ls_train = [0] * epochs
rmse_ls_test = [0] * epochs
ll_ls_train = [0] * epochs
ll_ls_test = [0] * epochs

save_dir = f"res/uci{args.suffix}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if method == "SVGD":
    kernel = Kernel(method="med_heuristic")
    # svgd = SVGDBNN(net, kernel, optim.Adam([w], lr=lr), device=device)
    svgd = SVGD(net, kernel, optim.Adam([w], lr=lr), device=device)

    svgd.lr = lr
    svgd.adagrad_state_dict = {
            'M': torch.zeros(w.shape, device=device),
            'V': torch.zeros(w.shape, device=device),
            't': 1,
            'beta1': 0.9,
            'beta2': 0.99
        }

    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            svgd.optim.zero_grad()
            # w.grad = -svgd.phi(w, features=X_batch, labels=y_batch, n_train=n_train)
            w.grad = -svgd.phi(w, X_data=X_batch, Y=y_batch, n_train=n_train)[0]
            svgd.optim.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
                rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
                ll = net.lik(X_test, y_test_scaled, w).item()
                dist = torch.pdist(w).sum().item()
                print(
                    f"Epoch: {epoch}, RMSE: {rmse}, LL: {ll}, spread: {dist}"
                )
                
                rmse_ls_test[epoch // 100] = rmse
                ll_ls_test[epoch // 100] = ll

                # train LL and RMSE
                y_pred_train = net.forward(X_train, w).squeeze(-1) * std_y_train + mean_y_train
                rmse_train = (y_pred_train.mean(0) - (y_train * std_y_train + mean_y_train)).pow(2).mean().sqrt()
                # y_batch_scaled = (y_batch - mean_y_train) / std_y_train
                # ll_train = net.lik(X_batch, y_batch_scaled, w).item()

                print(
                    f"train RMSE: {rmse_train}"
                )

                rmse_ls_train[epoch // 100] = rmse_train
                # ll_ls_train[epoch // 100] = ll_train

        if (epoch + 1) % 50 == 0:
            particles[1 + epoch // 50] = w.clone().detach().cpu()

    y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    dist = torch.pdist(w).sum().item()
    print(
        f"Epoch: {epoch}, RMSE: {rmse}, LL: {net.lik(X_test, y_test_scaled, w).item()}, spread: {dist}"
    )
    results_pickle = {
        # "init": w_init,
        # "final": w,
        # **{"target_dist": net},
        # "particles": particles,
        # "X_train": X_train,
        # "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        # "y_test_scaled": y_test_scaled,
        # "mean_y_train": mean_y_train,
        # "std_y_train": std_y_train,
        "rmse_test": rmse_ls_test,
        "ll_test": ll_ls_test,
        "rmse_train": rmse_ls_train,
        "ll_train": ll_ls_train
    }
    results = {
        "LL": net.lik(X_test, y_test_scaled, w).item(),
        "RMSE": rmse.item(),
        "spread": dist,
    }

    with open(
        f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_{seed}.json", "w"
    ) as outfile:
        json.dump(results, outfile)

    pickle.dump(
        results_pickle,
        open(
            f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_{seed}.p",
            "wb",
        ),
    )
if method == "GSVGD":
    # threshold = 0.0001*m # threshold for annealing of T
    threshold = 0.1*m # threshold for annealing of T

    kernel_gsvgd = BatchRBF(method="med_heuristic")
    optimizer = optim.Adam([w], lr=lr)
    eff_dims = [1]
    particles = [0] * (1 + epochs // 50)
    particles[0] = w.clone().detach().cpu()
    # TODO: can we use fewer projections? Or would using float speed things up?
    # m = dim_particles // eff_dims[0]
    manifold = Grassmann(dim_particles, M)
    # U = torch.eye(dim_particles)[:, :M*m].requires_grad_().to(device).double()
    U = (
        torch.nn.init.orthogonal_(torch.empty(dim_particles, m * M))
        .requires_grad_(True)
        .to(device)
        .double()
    )
    
    gsvgd = FullGSVGDBatch(
        target=net,
        kernel=kernel_gsvgd,
        manifold=manifold,
        optimizer=optimizer,
        delta=args.delta,
        T=T,
        device=device
    )

    # gsvgd.lr = lr
    # gsvgd.adagrad_state_dict = {
    #         'M': torch.zeros(w.shape, device=gsvgd.device),
    #         'V': torch.zeros(w.shape, device=gsvgd.device),
    #         't': 1,
    #         'beta1': 0.9,
    #         'beta2': 0.99
    #     }
    
    pam_old = 1e5
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            # gsvgd.optim.zero_grad()
            # phi, U = gsvgd.phi(w, U, m, X_batch, y_batch, n_train)
            # w.grad = -torch.einsum("bij -> ij", phi) / m
            # gsvgd.optim.step()
            
            U, phi, _, _, _ = gsvgd.step(
                X=w, 
                A=U.detach().requires_grad_(), 
                m=m, 
                X_data=X_batch, 
                Y=y_batch, 
                n_train=n_train
            )

            ## PAM and annealling variance multiplier
            with torch.no_grad():
                perturbation = torch.sum(phi.detach().clone(), dim=0)
                pert_norm = torch.max(perturbation.abs(), dim=1)[0]
                pam = pert_norm.mean().item()

                pam_diff = np.abs(pam - pam_old)
                # print(pam)
                if pam_diff < threshold and gsvgd.T < 1e6:
                    gsvgd.T *= 10
                    # print(f"Increase T to {gsvgd.T} at iteration {epoch+1} as delta PAM {pam_diff} is less than {threshold}")
                pam_old = pam

        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
                rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
                ll = net.lik(X_test, y_test_scaled, w).item()
                dist = torch.pdist(w).sum().item()
                print(
                    f"Epoch: {epoch}, RMSE: {rmse}, LL: {ll}, spread: {dist}"
                )

                rmse_ls_test[epoch // 100] = rmse
                ll_ls_test[epoch // 100] = ll

                # train LL and RMSE
                y_pred_train = net.forward(X_train, w).squeeze(-1) * std_y_train + mean_y_train
                rmse_train = (y_pred_train.mean(0) - (y_train * std_y_train + mean_y_train)).pow(2).mean().sqrt()
                # y_batch_scaled = (y_batch - mean_y_train) / std_y_train
                # ll_train = net.lik(X_batch, y_batch_scaled, w).item()

                print(
                    f"train RMSE: {rmse_train}"
                )

                rmse_ls_train[epoch // 100] = rmse_train
                # ll_ls_train[epoch // 100] = ll_train

        # if epoch % 50 == 0:
        #     U, _ = torch.qr(U)
        if (epoch + 1) % 50 == 0:
            particles[1 + epoch // 50] = w.clone().detach().cpu()
    y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    dist = torch.pdist(w).sum().item()
    print(
        f"Epoch: {epoch}, RMSE: {rmse}, LL: {net.lik(X_test, y_test_scaled, w).item()}, spread: {dist}"
    )
    results_pickle = {
        "init": w_init,
        "final": w,
        "particles": particles,
        **{"target_dist": net},
        # "X_train": X_train,
        # "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        # "y_test_scaled": y_test_scaled,
        # "mean_y_train": mean_y_train,
        # "std_y_train": std_y_train,
        "rmse_test": rmse_ls_test,
        "ll_test": ll_ls_test,
        "rmse_train": rmse_ls_train,
        "ll_train": ll_ls_train
    }
    results = {
        "LL": net.lik(X_test, y_test_scaled, w).item(),
        "RMSE": rmse.item(),
        "spread": dist,
    }

    with open(
        f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.json", "w"
    ) as outfile:
        json.dump(results, outfile)
    pickle.dump(
        results_pickle,
        open(
            f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.p",
            "wb",
        ),
    )

# if method == "GSVGDcano":
#     from src.gsvgd_canonical import FullGSVGDBatch

#     # threshold = 0.0001*m # threshold for annealing of T
#     threshold = 0.1*m # threshold for annealing of T

#     kernel_gsvgd = BatchRBF(method="med_heuristic")
#     optimizer = optim.Adam([w], lr=lr)
#     eff_dims = [1]
#     particles = [0] * (1 + epochs // 50)
#     particles[0] = w.clone().detach().cpu()

#     m = w.shape[1] if m == -1 else m
#     manifold = Grassmann(dim_particles, M)

#     U = torch.eye(dim_particles)[:, :M*m].requires_grad_().to(device).double()
#     B = torch.eye(dim_particles, device=device).double()
#     B = B[:, :(m*M)]

#     gsvgd = FullGSVGDBatch(
#         target=net,
#         kernel=kernel_gsvgd,
#         manifold=manifold,
#         optimizer=optimizer,
#         delta=args.delta,
#         T=T,
#         device=device
#     )

#     U = U.clone().detach().requires_grad_()
#     gsvgd.optim_proj = torch.optim.Adam([U], lr=gsvgd.delta, betas=(0.5, 0.9))

#     # alpha = U.sum() * 2
#     # (-alpha).backward()
#     # gsvgd.optim_proj.step()
#     # print("A", U[:3, :5])

#     gsvgd.lr = lr
#     gsvgd.adagrad_state_dict = {
#             'M': torch.zeros(w.shape, device=gsvgd.device),
#             'V': torch.zeros(w.shape, device=gsvgd.device),
#             't': 1,
#             'beta1': 0.9,
#             'beta2': 0.99
#         }
    
#     pam_old = 1e5
#     for epoch in tqdm(range(epochs)):
#         # print(A[:3, :5])
#         for X_batch, y_batch in train_loader:
#             # gsvgd.optim.zero_grad()
#             # phi, U = gsvgd.phi(w, U, m, X_batch, y_batch, n_train)
#             # w.grad = -torch.einsum("bij -> ij", phi) / m
#             # gsvgd.optim.step()
            
#             U, phi, _, _, _, _ = gsvgd.step(
#                 X=w, 
#                 A=U.detach().requires_grad_(), 
#                 B=B,
#                 m=m, 
#                 X_data=X_batch, 
#                 Y=y_batch, 
#                 n_train=n_train
#             )

#         ## PAM and annealling variance multiplier
#         perturbation = torch.sum(phi.detach().clone(), dim=0)
#         pert_norm = torch.max(perturbation.abs(), dim=1)[0]
#         pam = pert_norm.mean().item()

#         pam_diff = np.abs(pam - pam_old)
#         if pam_diff < threshold and gsvgd.T < 1e6:
#             gsvgd.T *= 10
#             print(f"Increase T to {gsvgd.T} at iteration {epoch+1} as delta PAM {pam_diff} is less than {threshold}")
#         pam_old = pam

#         if epoch % 100 == 0:
#             with torch.no_grad():
#                 y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
#                 rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
#                 ll = net.lik(X_test, y_test_scaled, w).item()
#                 dist = torch.pdist(w).sum().item()
#                 print(
#                     f"Epoch: {epoch}, RMSE: {rmse}, LL: {ll}, spread: {dist}"
#                 )

#                 rmse_ls_test[epoch // 100] = rmse
#                 ll_ls_test[epoch // 100] = ll

#                 # train LL and RMSE
#                 y_pred_train = net.forward(X_batch, w).squeeze(-1) * std_y_train + mean_y_train
#                 rmse_train = (y_pred_train.mean(0) - y_batch).pow(2).mean().sqrt()
#                 y_batch_scaled = (y_batch - mean_y_train) / std_y_train
#                 ll_train = net.lik(X_batch, y_batch_scaled, w).item()

#                 rmse_ls_train[epoch // 100] = rmse_train
#                 ll_ls_train[epoch // 100] = ll_train

#         # if epoch % 50 == 0:
#         #     U, _ = torch.qr(U)
#         if (epoch + 1) % 50 == 0:
#             particles[1 + epoch // 50] = w.clone().detach().cpu()
#     y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
#     rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
#     dist = torch.pdist(w).sum().item()
#     print(
#         f"Epoch: {epoch}, RMSE: {rmse}, LL: {net.lik(X_test, y_test_scaled, w).item()}, spread: {dist}"
#     )
#     results_pickle = {
#         "init": w_init,
#         "final": w,
#         "particles": particles,
#         **{"target_dist": net},
#         # "X_train": X_train,
#         # "y_train": y_train,
#         # "X_test": X_test,
#         # "y_test": y_test,
#         # "y_test_scaled": y_test_scaled,
#         # "mean_y_train": mean_y_train,
#         # "std_y_train": std_y_train,
#         "rmse_test": rmse_ls_test,
#         "ll_test": ll_ls_test,
#         "rmse_train": rmse_ls_train,
#         "ll_train": ll_ls_train
#     }
#     results = {
#         "LL": net.lik(X_test, y_test_scaled, w).item(),
#         "RMSE": rmse.item(),
#         "spread": dist,
#     }

#     with open(
#         f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.json", "w"
#     ) as outfile:
#         json.dump(results, outfile)
#     pickle.dump(
#         results_pickle,
#         open(
#             f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.p",
#             "wb",
#         ),
#     )

if method == "S-SVGD":
    band_scale = 1  # control the bandwidth scale
    counter_record = 0

    g = torch.eye(w.shape[-1]).to(device).requires_grad_()
    r = torch.eye(w.shape[-1]).to(device)
    # the bandwidth is computed inside functions, so here we set it to None
    kernel_hyper_maxSVGD = {"bandwidth": None}
    # Record the previous samples. This will be used for determining whether S-SVGD converged. If the change is not huge between samples, then we stop the g optimization, and
    # continue the S-SVGD update until the accumulated changes between samples are large enough. This is to avoid the overfitting of g to small number of samples.
    samples_pre_fix = w.clone().detach()
    samples = w.clone().detach().to(device).requires_grad_()
    del w

    #? g update
    Adam_g = torch.optim.Adam([g], lr=lr_g, betas=(0.5, 0.9))
    # g update epoch
    g_update = 1
    #? grassmann
    # manifold = Grassmann(samples.shape[1], 1)

    #? adagrad
    mixSVGD_state_dict = {
        "M": torch.zeros(samples.shape).to(device),
        "V": torch.zeros(samples.shape).to(device),
        "t": 1,
        "beta1": 0.9,
        "beta2": 0.99,
    }
    #? adam
    # optimizer = torch.optim.Adam([samples], lr=lr)
    
    avg_change = 0
    counter_not_opt = 0
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            # flag_opt = True
            if (
                get_distance(samples_pre_fix, samples)
                > np.sqrt(samples.shape[1]) * 0.15
            ):
                # the change between samples are large enough, so S-SVGD not coverged, we update g direction
                flag_opt = True
                samples_pre_fix = samples.clone().detach()
                counter_not_opt = 0
            else:
                # accumulate the epochs that the changes between samples are small.
                counter_not_opt += 1
                # accumulate 40 epochs, we update g.
                if counter_not_opt % 40 == 0:
                    flag_opt = True
                else:
                    flag_opt = False

            # Update g direction
            for i_g in range(1):
                #? g update
                Adam_g.zero_grad()
                # Normalize
                g_n = g / (torch.norm(g, 2, dim=-1, keepdim=True).to(device) + 1e-10)
                #? grassmann
                # g_n = g.clone().detach().requires_grad_() #! to be deleted

                # whether update g or not
                if flag_opt:
                    samples1 = samples.clone().detach().requires_grad_()
                    # compute target score
                    log_like1 = net.log_prob(W=samples1, X_data=X_batch, Y=y_batch, n_train=n_train)
                    score1 = torch.autograd.grad(log_like1.sum(), samples1)[0]
                    diver, divergence = compute_max_DSSD_eff(
                        samples1.detach(),
                        samples1.clone().detach(),
                        None,
                        SE_kernel,
                        d_kernel=d_SE_kernel,
                        dd_kernel=dd_SE_kernel,
                        r=r,
                        g=g_n,
                        kernel_hyper=kernel_hyper_maxSVGD,
                        score_samples1=score1,
                        score_samples2=score1.clone(),
                        flag_median=True,
                        flag_U=False,
                        median_power=0.5,
                        bandwidth_scale=band_scale,
                    )

                    #? g update
                    (-diver).backward()
                    Adam_g.step()
                    g_n = g / (
                        torch.norm(g, 2, dim=-1, keepdim=True).to(device) + 1e-10
                    )
                    g_n = g_n.clone().detach()
                    #? grassmann
                    # grad_g_n = torch.autograd.grad(diver, g_n)[0] # each row is a projection
                    # with torch.no_grad():
                    #     gT_r = g.unsqueeze(-1) # batch x dim x 1
                    #     grad_g_n_r = grad_g_n.unsqueeze(-1) # batch x dim x 1
                    #     gT_r_cp = manifold.retr(
                    #         gT_r.clone(),
                    #         manifold.egrad2rgrad(gT_r.clone(), lr*grad_g_n_r),
                    #     ) # batch x dim x 1
                    #     g = gT_r_cp.squeeze(-1).clone()
                    # g_n = g.clone().detach()

                log_like1 = net.log_prob(W=samples, X_data=X_batch, Y=y_batch, n_train=n_train)

                score1 = torch.autograd.grad(log_like1.sum(), samples)[0]

                maxSVGD_force, repulsive = max_DSSVGD(
                    samples,
                    None,
                    SE_kernel,
                    repulsive_SE_kernel,
                    r=r,
                    g=g_n,
                    flag_median=True,
                    median_power=0.5,
                    kernel_hyper=kernel_hyper_maxSVGD,
                    score=score1,
                    bandwidth_scale=band_scale,
                    repulsive_coef=1,
                    flag_repulsive_output=True,
                )

                repulsive_max, _ = torch.max(repulsive, dim=1)

                # particle-averaged magnitude (batch x num_particles)
                # pam = torch.linalg.norm(maxSVGD_force.detach(), dim=1).mean().item()

                # update particles
                #? adagrad
                samples, mixSVGD_state_dict = SVGD_AdaGrad_update(
                    samples, maxSVGD_force, lr, mixSVGD_state_dict
                )
                samples = samples.clone().requires_grad_()
                #? adam
                # optimizer.zero_grad()
                # samples.grad = -maxSVGD_force
                # optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred = (
                    net.forward(X_test, samples).squeeze(-1) * std_y_train + mean_y_train
                )
                rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
                ll = net.lik(X_test, y_test_scaled, samples).item()
                dist = torch.pdist(samples).sum().item()
                print(
                    f"Epoch: {epoch}, RMSE: {rmse}, LL: {ll}, spread: {dist}"
                )

                rmse_ls_test[epoch // 100] = rmse
                ll_ls_test[epoch // 100] = ll

                # train LL and RMSE
                y_pred_train = (
                    net.forward(X_batch, samples).squeeze(-1) * std_y_train + mean_y_train
                )
                rmse_train = (y_pred_train.mean(0) - y_batch).pow(2).mean().sqrt()
                y_batch_scaled = (y_batch - mean_y_train) / std_y_train
                ll_train = net.lik(X_batch, y_batch_scaled, samples).item()

                rmse_ls_train[epoch // 100] = rmse_train
                ll_ls_train[epoch // 100] = ll_train

        if (epoch + 1) % 50 == 0:
            particles[1 + epoch // 50] = samples.clone().detach().cpu()
    y_pred = net.forward(X_test, samples).squeeze(-1) * std_y_train + mean_y_train
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    dist = torch.pdist(samples).sum().item()
    print(
        f"Epoch: {epoch}, RMSE: {rmse}, LL: {net.lik(X_test, y_test_scaled, samples).item()}, spread: {dist}"
    )
    results_pickle = {
        "init": w_init,
        "final": samples,
        "particles": particles,
        **{"target_dist": net},
        # "X_train": X_train,
        # "y_train": y_train,
        # "X_test": X_test,
        # "y_test": y_test,
        # "y_test_scaled": y_test_scaled,
        # "mean_y_train": mean_y_train,
        # "std_y_train": std_y_train,
        "rmse_test": rmse_ls_test,
        "ll_test": ll_ls_test,
        "rmse_train": rmse_ls_train,
        "ll_train": ll_ls_train
    }
    results = {
        "LL": net.lik(X_test, y_test_scaled, samples).item(),
        "RMSE": rmse.item(),
        "spread": dist,
    }

    with open(
        f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_{seed}.json", "w"
    ) as outfile:
        json.dump(results, outfile)
    pickle.dump(
        results_pickle,
        open(
            f"{save_dir}/{dataset}_{method}_nparticles{nparticles}_{seed}.p",
            "wb",
        ),
    )
