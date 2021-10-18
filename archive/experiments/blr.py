import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
import sys

sys.path.append(".")

from src.svgd import SVGDLR
# from src.full_gsvgd_seq import FullGSVGDLR
from src.gsvgd import FullGSVGDBatchLR
from src.kernel import RBF, IMQ, SumKernel, BatchRBF
from src.utils import plot_particles
from src.metrics import Metric
from src.manifold import Grassmann
from src.maxsvgd import MaxSVGDLR
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse


class BayesianLR:
    def __init__(self, X, Y, a0=1, b0=0.01):
        self.X, self.Y = X, Y.T
        # Y in \in{+1, -1}
        self.a0, self.b0 = a0, b0
        self.N = X.shape[0]

    def log_prob(self, theta, X_batch, y_batch):
        """
            theta: nparticles x (d + 1)
            X_batch: n_b x d
            y_batch: n_b
        """
        theta = theta.clone().requires_grad_()
        nb, d = X_batch.shape
        w, alpha = theta[:, :-1], theta[:, -1:].exp()

        # log-likelihood term
        wx = w @ X_batch.t() # nparticles x nbatch
        loglik = - (1 + (-wx).exp()).log() 
        loglik = torch.sum(loglik, axis=1, keepdim=True)
        loglik += - wx @ (1 - y_batch) / 2 
        loglik *= 1 / nb * self.N # nparticles
        # print("loglik", loglik[:5])
        
        # log-prior of w given alpha
        logp_w = (
            0.5 * d * alpha.log() 
            - 0.5 * alpha * torch.einsum("ij, ij -> i", w, w).reshape((w.shape[0], 1))
        ) # nparticles
        # print("logp_w", logp_w[:5])

        # log-prior for alpha
        logp_alpha = (self.a0 - 1) * alpha.log() - self.b0 * alpha # nparticles
        # print("logp_alpha", logp_alpha[:5])

        logprob = loglik + logp_w + logp_alpha

        return logprob

    def evaluation(self, theta, X_test, y_test):
        theta = theta.clone().requires_grad_(False)
        w = theta[:, :-1]
        wx = w @ X_test.t()  # nparticles x ndata
        prob = 1 / (1 + (-wx).exp())  # nparticles x ndata
        prob = torch.mean(prob, axis=0, keepdim=True).t()  # ndata x 1
        # y_pred = torch.Tensor.float(prob > 0.5)
        # acc = torch.mean(torch.Tensor.float(y_pred == y_test))
        y_pred = torch.Tensor.float(prob > 0.5)
        y_pred[y_pred == 0] = -1
        acc = torch.mean(torch.Tensor.float(y_pred == y_test)).cpu().item()

        wx_ave = torch.mean(wx, axis=0, keepdim=True).t() # ndata x 1
        ll = (
            -wx_ave * (1 - y_test)/2 + (1 + (-wx_ave).exp()).log()
        ).mean().cpu().item()

        return prob, y_pred, acc, ll

    # def grad_logprob(self, theta, X_batch, y_batch, n_train):
    #     theta = theta.clone()
    #     nb, d = X_batch.shape
    #     assert d == (theta.shape[1] - 1)
    #     w, alpha = theta[:, :-1], theta[:, -1:].exp()

    #     wx = w @ X_batch.t() # nparticles x nbatch

    #     ## grad_w
    #     gradw_data = (
    #         (-wx).exp() / (1 + (-wx).exp()) 
    #         - (1 - y_batch.t().repeat(nparticles, 1)) / 2
    #     ) @ X_batch # nparticles x d
    #     gradw_prior = - alpha * w # nparticles x d
    #     gradw = gradw_data / nb * n_train + gradw_prior
        
    #     ## grad_gamma (gamma := log alpha)
    #     ww = torch.einsum("ij, ij -> i", w, w).reshape((theta.shape[0], 1))
    #     gradgamma = d * 0.5 - 0.5 * alpha * ww + (self.a0 - 1) - self.b0 * alpha
        
    #     grad = torch.hstack([gradw, gradgamma])

    #     return grad


def load_data_from_torch():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(
        #     (0.1307,), (0.3081,))
    ])
    mnist_train_raw = torchvision.datasets.MNIST("data/", train=True, download=True,
        transform=transform)
    mnist_test_raw = torchvision.datasets.MNIST("data/", train=False, download=True,
        transform=transform)
    return mnist_train_raw, mnist_test_raw

def get_digits(dataset, d1, d2):
    """Take out images for digits d1 and d2 only, and set label
    of d1 as -1, of d2 as 1
    """
    idx0 = dataset.targets == d1
    idx1 = dataset.targets == d2
    idx = idx0 | idx1

    dataset.targets[idx0] = -1
    dataset.targets[idx1] = 1
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    return dataset

def dataset_to_numpy(dataset, ndata):
    X = dataset.data
    y = dataset.targets
    ndata = min(ndata, X.shape[0])
    # idx = np.random.choice(X.shape[0], ndata)
    idx = np.array(range(X.shape[0]))[:ndata]
    X = np.float32(X[idx, :])
    X = X.reshape((X.shape[0], 28*28)) / 255.
    y = np.float32(y[idx])
    y = y.reshape((-1, 1))
    return X, y

def load_and_process_from_torch(d1, d2, ndata):
    mnist_train_raw, mnist_test_raw = load_data_from_torch()

    mnist_train_raw = get_digits(mnist_train_raw, d1, d2)
    mnist_test_raw = get_digits(mnist_test_raw, d1, d2)

    X_train, y_train = dataset_to_numpy(mnist_train_raw, ndata)
    X_test, y_test = dataset_to_numpy(mnist_test_raw, ndata)

    return X_train, y_train, X_test, y_test


def load_mnist(ndata, d1, d2, test_size, seed, device="cpu"):
    """Load and process MNIST"""

    X_train, y_train, X_test, y_test = load_and_process_from_torch(d1, d2, ndata)

    # split into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=test_size, random_state=seed)

    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    X_valid = torch.Tensor(X_valid).to(device)
    y_train = torch.Tensor(y_train).to(device)
    y_test = torch.Tensor(y_test).to(device)
    y_valid = torch.Tensor(y_valid).to(device)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--effdim", type=int, default=-1, help="dimension")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_g", type=float, default=0.001, help="learning rate for S-SVGD")
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
parser.add_argument("--data", type=str, default="covertype", help="which dataset to use")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--method", type=str, default="svgd", help="svgd, gsvgd or s-svgd")
parser.add_argument("--save_every", type=int, default=10, help="batch size")
parser.add_argument("--d1", type=int, default=3, help="first digit")
parser.add_argument("--d2", type=int, default=5, help="second digit")

args = parser.parse_args()
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
batch_size = args.batch
print(f"Running for lr: {lr}, nparticles: {nparticles}")

device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")
# device = "cuda"


results_folder = f"./res/blr{args.suffix}/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}"
results_folder = f"{results_folder}/seed{seed}"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if args.kernel == "rbf":
    Kernel = RBF
    BatchKernel = BatchRBF
elif args.kernel == "imq":
    Kernel = IMQ


if __name__ == "__main__":
    print(f"Device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_mnist(30000, args.d1, args.d2, 0.2, seed, device)
    d = X_train.shape[-1]
    D = d + 1
    # #! hard-coded not to use test
    # X_test, y_test = X_valid, y_valid

    # labels must be (-1, 1)
    assert set(y_train.unique().tolist()) == set([-1, 1])

    print("[Train, valid, test] sizes:", 
        list(X_train.shape), list(X_valid.shape), list(X_test.shape))
    print("class 1 in train set: ", ((y_train==1).sum()/y_train.shape[0]).item())

    n_train = X_train.shape[0]
    train_dataset = TensorDataset(X_train, y_train)
    batch_size = batch_size if batch_size > 0 else X_train.shape[0]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    print("num batches:", len(train_loader))


    ## target density
    a0, b0 = 1.0, 0.01  # hyper-parameters
    distribution = BayesianLR(X_train, y_train, a0, b0)

    # initialization
    theta0 = torch.zeros([nparticles, D], device=device)
    gamma_d = torch.distributions.gamma.Gamma(torch.Tensor([a0]), torch.Tensor([b0]))
    alpha0 = gamma_d.sample((nparticles,)).to(device)
    for i in range(nparticles):
        w0 = alpha0[i].pow(-1).sqrt() * torch.randn(d, device=device)
        theta0[i, :] = torch.hstack([w0, alpha0[i].log()])
    theta0 = theta0.to(device)


    # ## logistic regression
    # from sklearn.linear_model import LogisticRegression
    # logisticRegr = LogisticRegression()
    # logisticRegr.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
    # score = logisticRegr.score(X_test.cpu().numpy(), y_test.cpu().numpy())
    # print("score", score)

    
    if args.method == "svgd":
        ## SVGD
        print("Running SVGD")
        x = theta0.clone().requires_grad_()
        # sample from variational density
        kernel = Kernel(method="med_heuristic")
        svgd = SVGDLR(distribution, kernel, optim.Adam([x], lr=lr), device=device)

        _ = svgd.fit(x0=x, epochs=epochs, save_every=save_every, train_loader=train_loader,
            test_data=(X_test, y_test), valid_data=(X_valid, y_valid))

        fitted_method = svgd
        method_name = "svgd"

    elif args.method == "gsvgd":
        eff_dim = args.effdim

        print(f"Running GSVGD with eff dim = {eff_dim}")
        # m = min(D, 20) // eff_dim
        m = min(20, D // eff_dim)
        # m = min(100, D // eff_dim)
        print("number of projections:", m)

        # sample from variational density
        x_gsvgd = theta0.clone().requires_grad_()

        # kernel_gsvgd = Kernel(method="med_heuristic")
        kernel_gsvgd = BatchKernel(method="med_heuristic")
        optimizer = optim.Adam([x_gsvgd], lr=lr)
        manifold = Grassmann(D, eff_dim)
        # U = torch.eye(D).requires_grad_(True).to(device)
        # U = U[:, :(m*eff_dim)]
        U = torch.nn.init.orthogonal_(
            torch.empty(D, m*eff_dim)
        ).requires_grad_(True).to(device)

        gsvgd = FullGSVGDBatchLR(
            target=distribution,
            kernel=kernel_gsvgd,
            manifold=manifold,
            optimizer=optimizer,
            delta=delta,
            T=T,
            device=device,
            noise=add_noise
        )
        U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
            verbose=True, save_every=save_every, threshold=0.0001*m,
            train_loader=train_loader, test_data=(X_test, y_test), 
            valid_data=(X_valid, y_valid))

        fitted_method = gsvgd
        method_name = f"gsvgd_effdim{eff_dim}"

    elif args.method == "s-svgd":
        ## S-SVGD
        # sample from variational density
        print("Running maxSVGD")
        x_maxsvgd = theta0.clone().requires_grad_()
        maxsvgd = MaxSVGDLR(distribution, device=device)

        x_maxsvgd, metric_maxsvgd = maxsvgd.fit(
            samples=x_maxsvgd, 
            n_epoch=epochs, 
            lr=lr_g,
            eps=lr,
            save_every=save_every,
            train_loader=train_loader,
            test_data=(X_test, y_test),
            valid_data=(X_valid, y_valid)
        )

        fitted_method = maxsvgd
        method_name = f"s-svgd_lrg{lr_g}"

    ## save results
    # particles
    particles_epochs = [tup[0] for tup in fitted_method.test_accuracy]
    test_acc = [tup[1] for tup in fitted_method.test_accuracy]
    valid_acc = [tup[1] for tup in fitted_method.valid_accuracy]
    test_ll = [tup[2] for tup in fitted_method.test_accuracy]
    valid_ll = [tup[2] for tup in fitted_method.valid_accuracy]
    pickle.dump(
        {
            **{"epochs": particles_epochs},
            **{"test_accuracy": test_acc},
            **{"valid_accuracy": valid_acc},
            **{"test_ll": test_ll},
            **{"valid_ll": valid_ll},
            **{"particles": fitted_method.particles},
            **{"nbatches": len(train_loader)}
        },
        open(results_folder + f"/particles_{method_name}.p", "wb")
    )

    # target distribution
    torch.save(distribution, results_folder + '/target_dist.p')
    
    