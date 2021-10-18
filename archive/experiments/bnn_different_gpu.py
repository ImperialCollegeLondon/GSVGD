import sys

sys.path.append(".")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.Sliced_KSD_Clean.Divergence.Network import Bayesian_NN_eff
from src.svgd import SVGDBNN
from src.full_gsvgd import FullGSVGDBNN
from src.kernel import RBF, IMQ, BatchRBF
from src.manifold import Grassmann
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--nparticles", type=int, default=50, help="no. of particles")
parser.add_argument("--noise", type=str, default="True", help="whether to add noise")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--dataset", type=str, default="boston_housing", help="dataset")
parser.add_argument("--seed", type=int, default=2, help="seed")
parser.add_argument("--method", type=str, default="SVGD", help="SVGD Method")

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

# epochs based on Sliced Kernelized Stein Discrepancy, Gong et al. (2021)
if dataset in ["boston_housing", "conrete", "energy"]:
    epochs = 2000
elif dataset == "wine":
    epochs = 500
elif dataset == "protein":
    epochs = 50
else:
    epochs = 200
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
# y_test_scaled = (y_test - mean_y_train) / std_y_train

# place all numpy arrays into torch.Tensor
X_train, X_test, y_train, y_test = (
    torch.from_numpy(X_train).to(device).double(),
    torch.from_numpy(X_test).to(device).double(),
    torch.from_numpy(y_train).to(device).double(),
    torch.from_numpy(y_test).to(device).double(),
)

# y_test_scaled = torch.from_numpy(y_test_scaled).to(device).double()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

n_train = X_train.shape[0]
dim = X_train.shape[1]
net = Bayesian_NN_eff(dim, device=device)
# use same configurations as Liu and Wang (2016)
w_init = net.init_weights(nparticles, a0=1, b0=0.1)
w = w_init.clone()
dim_particles = w.shape[1]
# print(net.forward(torch.ones(50, dim, device=device).double(), w))
# print(net.log_prob(X_train, y_train, n_train, w).shape, X_train.shape)

print(f"Running for dataset: {dataset}, dim_particles: {dim_particles}, lr: {lr}, nparticles: {nparticles}, seed: {seed}")

if method == "SVGD":
    kernel = Kernel(method="med_heuristic")
    svgd = SVGDBNN(net, kernel, optim.Adam([w], lr=lr), device=device)

    # print(net.forward(X_test, w))
    # print(svgd.phi(w, X_test, y_test, n_train))
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            svgd.optim.zero_grad()
            w.grad = -svgd.phi(w, X_batch, y_batch, n_train)
            svgd.optim.step()

        if epoch % 100 == 0:
            y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
            rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
            print(
                f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
            )

    y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    print(
        f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
    )

    results = {
        "LL": net.lik(X_test, y_test, y_pred, w).mean(0).log().mean().item(),
        "RMSE": rmse.item()
    }

    with open(f"res/uci/{dataset}_{method}_{seed}.json", "w") as outfile:
        json.dump(results, outfile)

if method == "GSVGD":
    kernel_gsvgd = BatchRBF(method="med_heuristic")
    optimizer = optim.Adam([w], lr=lr)
    eff_dims = [1]
    # TODO: can we use fewer projections? Or would using float speed things up?
    m = dim_particles // eff_dims[0]
    manifold = Grassmann(dim_particles, eff_dims[0])
    U = torch.eye(dim_particles).requires_grad_(True).to(device).double()
    gsvgd = FullGSVGDBNN(
        target=net,
        kernel=kernel_gsvgd,
        manifold=manifold,
        optimizer=optimizer,
        delta=0.1,
        device=device,
        noise=add_noise
    )

    # print(net.forward(X_test, w))
    # print(svgd.phi(w, X_test, y_test, n_train))
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            # print(w.shape, U.shape)
            phi, U = gsvgd.phi(w, U, m, X_batch, y_batch, n_train)
            gsvgd.optim.zero_grad()
            w.grad = -torch.einsum("bij -> ij", phi)
            gsvgd.optim.step()

        if epoch % 50 == 0:
            y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
            rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
            print(
                f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
            )
        if epoch % 1000 ==0:
            U, _ = torch.qr(U)

    y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
    rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
    print(
        f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
    )

    results = {
        "LL": net.lik(X_test, y_test, y_pred, w).mean(0).log().mean().item(),
        "RMSE": rmse.item()
    }

    with open(f"res/uci/{dataset}_{method}_{seed}.json", "w") as outfile:
        json.dump(results, outfile)

# if method == "MaxSVGD":

#     # print(net.forward(X_test, w))
#     # print(svgd.phi(w, X_test, y_test, n_train))
#     for epoch in tqdm(range(2)):
#         for X_batch, y_batch in train_loader:
#             # print(w.shape, U.shape)
#             phi, U = gsvgd.phi(w, U, m, X_batch, y_batch, n_train)
#             gsvgd.optim.zero_grad()
#             w.grad = -torch.einsum("bij -> ij", phi)
#             gsvgd.optim.step()

#         if epoch % 50 == 0:
#             y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
#             rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
#             print(
#                 f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
#             )
#         if epoch % 1000 ==0:
#             U, _ = torch.qr(U)

#     y_pred = (net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train)
#     rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
#     print(
#         f"Epoch: {epoch}, LL: {net.lik(X_test, y_test, y_pred, w).mean(0).log().mean()}, RMSE: {rmse}"
#     )

#     results = {
#         "LL": net.lik(X_test, y_test, y_pred, w).mean(0).log().mean().item(),
#         "RMSE": rmse.item()
#     }

#     with open(f"res/uci/{dataset}_{method}_{seed}.json", "w") as outfile:
#         json.dump(results, outfile)