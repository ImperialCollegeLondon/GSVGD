import sys

sys.path.append(".")
import os


import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data import DataLoader, TensorDataset

from src.Sliced_KSD_Clean.Divergence.Network import Bayesian_NN_eff
from src.svgd import SVGDBNN
from src.gsvgd import FullGSVGDBatchBNN
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
    epochs = 2000
elif dataset == "wine":
    epochs = 500
elif dataset == "protein":
    epochs = 50
else:
    epochs = 200

# Dilin's code
np.random.seed(seed)
data = np.loadtxt("data/boston_housing")

# Please make sure that the last column is the label and the other columns are features
X_input = data[ :, range(data.shape[ 1 ] - 1) ]
y_input = data[ :, data.shape[ 1 ] - 1 ]

''' build the training and testing data set'''
train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
permutation = np.arange(X_input.shape[0])
random.shuffle(permutation) 

size_train = int(np.round(X_input.shape[ 0 ] * train_ratio))
index_train = permutation[ 0 : size_train]
index_test = permutation[ size_train : ]

X_train, y_train = X_input[ index_train, : ], y_input[ index_train ]
X_test, y_test = X_input[ index_test, : ], y_input[ index_test ]

size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

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

X_dev = (X_dev - np.full(X_dev.shape, mean_X_train)) / np.full(
    X_dev.shape, std_X_train
)

# place all numpy arrays into torch.Tensor
X_train, X_dev, X_test, y_train, y_dev, y_test = (
    torch.from_numpy(X_train).to(device).double(),
    torch.from_numpy(X_dev).to(device).double(),
    torch.from_numpy(X_test).to(device).double(),
    torch.from_numpy(y_train).to(device).double(),
    torch.from_numpy(y_dev).to(device).double(),
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


if method == "SVGD":
    kernel = Kernel(method="med_heuristic")
    # svgd = SVGDBNN(net, kernel, optim.Adam([w], lr=lr), device=device)
    svgd = SVGDBNN(net, kernel, optim.Adam([w], lr=lr), device=device)

    # print(net.forward(X_test, w))
    # print(svgd.phi(w, X_test, y_test, n_train))
    for epoch in tqdm(range(epochs)):
        for X_batch, y_batch in train_loader:
            svgd.optim.zero_grad()
            w.grad = -svgd.phi(w, X_batch, y_batch, n_train)
            svgd.optim.step()

        if epoch % 100 == 0:
            y_pred = net.forward(X_test, w).squeeze(-1) * std_y_train + mean_y_train
            rmse = (y_pred.mean(0) - y_test).pow(2).mean().sqrt()
            dist = torch.pdist(w).sum().item()
            print(
                f"Epoch: {epoch}, LL: {net.lik(X_test, y_test_scaled, w).item()}, RMSE: {rmse}, spread: {dist}"
            )
        if (epoch + 1) % 50 == 0:
            particles[1 + epoch // 50] = w.clone().detach().cpu()

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
        f"res/uci/dilin_{dataset}_{method}_nparticles{nparticles}_{seed}.json", "w"
    ) as outfile:
        json.dump(results, outfile)

    pickle.dump(
        results_pickle,
        open(
            f"res/uci/dilin_{dataset}_{method}_nparticles{nparticles}_{seed}.p",
            "wb",
        ),
    )