import sys

sys.path.append(".")
import os


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.Sliced_KSD_Clean.Divergence.Network import Bayesian_NN_eff
from src.svgd import SVGDBNN
from src.full_gsvgd_seq import FullGSVGDBNN
from src.kernel import RBF, IMQ, BatchRBF
from src.manifold import Grassmann
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

parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--nparticles", type=int, default=50, help="no. of particles")
parser.add_argument("--noise", type=str, default="True", help="whether to add noise")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--dataset", type=str, default="boston_housing", help="dataset")
parser.add_argument("--seed", type=int, default=2, help="seed")
parser.add_argument("--method", type=str, default="SVGD", help="SVGD Method")
parser.add_argument("--gpu", type=int, default=0, help="SVGD Method")
parser.add_argument("--m", type=int, default=5, help="Number of Projections")
parser.add_argument("--M", type=int, default=15, help="Projection Dimension")

args = parser.parse_args()
lr = args.lr
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

os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epochs based on Sliced Kernelized Stein Discrepancy, Gong et al. (2021)
if dataset in ["boston_housing", "concrete", "energy"]:
    epochs = 3000
elif dataset == "wine":
    epochs = 700
elif dataset == "protein":
    epochs = 100
else:
    epochs = 300
np.random.seed(seed)
data = np.loadtxt(f"data/{dataset}")
X = data[:, range(data.shape[1] - 1)]
y = data[:, data.shape[1] - 1]
# X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=seed
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
if method == "GSVGD":
    if dataset in ["boston_housing", "concrete", "energy"]:
        epochs = 3000
    elif dataset == "wine":
        epochs = 700
    elif dataset == "protein":
        epochs = 100
    else:
        epochs = 300
    kernel_gsvgd = RBF(method="med_heuristic")
    optimizer = optim.Adam([w], lr=lr)
    eff_dims = [1]
    # TODO: can we use fewer projections? Or would using float speed things up?
    # m = dim_particles // eff_dims[0]
    # m = 10
    # m = 5
    # M = 10
    # manifold = Grassmann(dim_particles, eff_dims[0])
    manifold = Grassmann(dim_particles, M)
    # U = torch.eye(dim_particles)[:, :M*m].requires_grad_(True).to(device).double()
    U = (
        torch.nn.init.orthogonal_(torch.empty(dim_particles, m * M))
        .requires_grad_(True)
        .to(device)
        .double()
    )
    gsvgd = FullGSVGDBNN(
        target=net,
        kernel=kernel_gsvgd,
        manifold=manifold,
        optimizer=optimizer,
        delta=0.001,
        device=device,
        noise=True,
    )
    particles = [0] * (1 + epochs // 50)
    particles[0] = w.clone().detach().cpu()
    # print(net.forward(X_test, w))
    # print(svgd.phi(w, X_test, y_test, n_train))
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:

            U_new = torch.zeros_like(U).detach()
            M = U.shape[1] // m  # proj dim
            # for i in range(U.shape[1]):
            for i in range(m):
                gsvgd.optim.zero_grad()
                phi, U_ = gsvgd.phi(
                    w, U[:, (i * M) : ((i + 1) * M)], X_batch, y_batch, n_train
                )
                U_new[:, (i * M) : ((i + 1) * M)] = U_
                w.grad = -phi
                gsvgd.optim.step()
            U = U_new.detach().clone()

        if epoch % 50 == 0:
            y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
            rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
            dist = torch.pdist(w).sum().item()
            print(
                f"Epoch: {epoch}, LL: {net.lik(X_test, y_test_scaled, w).item()}, RMSE: {rmse}, spread: {dist}"
            )
        if (epoch + 1) % 50 == 0:
            particles[1 + epoch // 50] = w.clone().detach().cpu()
        # if epoch % 50 == 0:
        #     U, _ = torch.qr(U)

    y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    dist = torch.pdist(w).sum().item()
    print(
        f"Epoch: {epoch}, LL: {net.lik(X_test, y_test_scaled, w).item()}, RMSE: {rmse}, spread: {dist}"
    )
    results_pickle = {
        "init": w_init,
        "final": w,
        "particles": particles,
        **{"target_dist": net},
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_scaled": y_test_scaled,
        "mean_y_train": mean_y_train,
        "std_y_train": std_y_train
    }
    results = {
        "LL": net.lik(X_test, y_test_scaled, w).item(),
        "RMSE": rmse.item(),
        "spread": dist,
    }

    with open(
        f"res/uci/seq_{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.json",
        "w",
    ) as outfile:
        json.dump(results, outfile)

    pickle.dump(
        results_pickle,
        open(
            f"res/uci/seq_{dataset}_{method}_nparticles{nparticles}_m{m}_M{M}_{seed}.p",
            "wb",
        ),
    )
