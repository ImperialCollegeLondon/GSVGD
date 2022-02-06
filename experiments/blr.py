import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.io
from sklearn.model_selection import train_test_split

from src.svgd import SVGDLR
from src.gsvgd import FullGSVGDBatchLR
from src.kernel import RBF, BatchRBF
from src.manifold import Grassmann
from src.s_svgd import SlicedSVGDLR
from src.blr import BayesianLR
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
import time


parser = argparse.ArgumentParser(description="Running xshaped experiment.")
parser.add_argument("--effdim", type=int, default=-1, help="dimension")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_g", type=float, default=0.001, help="learning rate for S-SVGD")
parser.add_argument(
    "--delta", type=float, default=0.01, help="stepsize for projections"
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
parser.add_argument("--save_every", type=int, default=10, help="save results per xxx epoch")
parser.add_argument("--subsample_size", type=int, default=0, help="HMCECS subsample size")

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

device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cuda")

results_folder = f"./res/blr{args.suffix}/{args.kernel}_epoch{epochs}_lr{lr}_delta{delta}_n{nparticles}"
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

    ## load data
    if args.data == "covertype":
        data = scipy.io.loadmat("./data/covertype.mat")

        X_input = data["covtype"][:, 1:]
        y_input = data["covtype"][:, :1]
        y_input[y_input == 2] = -1

    if args.data == "covertype_sub":
        data = scipy.io.loadmat("./data/covertype.mat")

        X_input = data["covtype"][:, 1:]
        y_input = data["covtype"][:, :1]
        nsubsample = 1000
        ind = np.random.choice(X_input.shape[0], nsubsample, replace=False)
        X_input, y_input = X_input[ind, :], y_input[ind, :]
        y_input[y_input == 2] = -1

    elif args.data == "debug":
        X_input = torch.randn(100, 2)
        y_input = ((1.5 * X_input[:, 0] + 0.5 * X_input[:, 1] - 0) > 0)
        y_input = y_input.reshape((X_input.shape[0], 1)).type(torch.float32)
        y_input[y_input == 0] = -1.

    elif args.data == "arcene":
        data = scipy.io.loadmat("./data/arcene.mat")
        X_input = data["X"] / 1000.
        y_input = data["labels"] * 1.

    elif args.data == "benchmarks":
        # high-dim datasets:
        # german, image, ringnorm, splice, twonorm, waveform, breast_cancer, flare_solar
        data = scipy.io.loadmat("./data/benchmarks.mat")

        data = data[args.data][0][0]
        X_input = data[0]
        y_input = data[1]

        # check format of labels
        assert (y_input.min(), y_input.max()) == (-1, 1), "range of y is not (-1, 1)"
    
    elif args.data == "sonar":
        data = pd.read_csv("data/sonar.csv")
        y_input = data.Class
        y_input.replace({"Rock": -1, "Mine": 1}, inplace=True)
        y_input = np.array(y_input, dtype=np.float32).reshape((-1, 1))
        data.drop("Class", axis=1, inplace=True)
        X_input = np.array(data)

    elif args.data == "higgs":
        from numpyro.examples.datasets import HIGGS, load_dataset
        _, fetch = load_dataset(
            HIGGS, shuffle=False, num_datapoints=1000
        )
        data, obs = fetch()
        X_input = np.array(data, dtype=np.float32)
        y_input = np.array(obs, dtype=np.float32)
        y_input[y_input == 0] = -1.


    # load data into datasets
    N = X_input.shape[0]
    d = X_input.shape[1]
    D = d + 1

    # split the dataset into training and testing
    if args.data in ["arcene", "debug"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_input, y_input, test_size=0.2, random_state=seed
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )
        print("train prevalence:", (y_train == 1).sum() / y_train.shape[0])
        print("valid prevalence:", (y_valid == 1).sum() / y_valid.shape[0])

    
    else:
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X_input, y_input, test_size=0.2, random_state=seed
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid, y_train_valid, test_size=0.2, random_state=seed
        )

    X_train = torch.Tensor(X_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    X_valid = torch.Tensor(X_valid).to(device)
    y_train = torch.Tensor(y_train).to(device)
    y_test = torch.Tensor(y_test).to(device)
    y_valid = torch.Tensor(y_valid).to(device)
    print("[Train, test] sizes:", X_train.shape, X_test.shape)
    print("class 1 in train set: ", ((y_train==1).sum()/y_train.shape[0]).item())
    print("class 1 in test set: ", ((y_test==1).sum()/y_test.shape[0]).item())

    n_train = X_train.shape[0]
    train_dataset = TensorDataset(X_train, y_train)
    batch_size = batch_size if batch_size > 0 else X_train.shape[0]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    print("num batches:", len(train_loader))


    ## target density
    a0, b0 = 1.0, 0.01 # hyper-parameters
    distribution = BayesianLR(X_train, y_train, a0, b0)

    # initialization
    theta0 = torch.zeros([nparticles, D], device=device)
    gamma_d = torch.distributions.gamma.Gamma(torch.Tensor([a0]), torch.Tensor([b0]))
    alpha0 = gamma_d.sample((nparticles,)).to(device)
    for i in range(nparticles):
        w0 = alpha0[i].pow(-1).sqrt() * torch.randn(d, device=device)
        theta0[i, :] = torch.hstack([w0, alpha0[i].log()])
    theta0 = theta0.to(device)


    if args.method == "svgd":
        ## SVGD
        print("Running SVGD")
        x = theta0.clone().requires_grad_()
        # sample from variational density
        kernel = Kernel(method="med_heuristic")
        svgd = SVGDLR(distribution, kernel, optim.Adam([x], lr=lr), device=device)

        start = time.time()
        svgd.fit(x0=x, epochs=epochs, save_every=save_every, train_loader=train_loader,
            test_data=(X_test, y_test), valid_data=(X_valid, y_valid))
        elapsed_time = time.time() - start

        fitted_method = svgd
        method_name = "svgd"

    elif args.method == "gsvgd":
        eff_dim = args.effdim

        print(f"Running GSVGD with eff dim = {eff_dim}")
        m = min(20, D // eff_dim)
        print("number of projections:", m)

        # sample from variational density
        x_gsvgd = theta0.clone().requires_grad_()

        kernel_gsvgd = BatchKernel(method="med_heuristic")
        optimizer = optim.Adam([x_gsvgd], lr=lr)
        manifold = Grassmann(D, eff_dim)
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
        start = time.time()
        U, metric_gsvgd = gsvgd.fit(x_gsvgd, U, m, epochs, 
            verbose=True, save_every=save_every, threshold=0.0001*m,
            train_loader=train_loader, test_data=(X_test, y_test), 
            valid_data=(X_valid, y_valid))
        elapsed_time = time.time() - start

        fitted_method = gsvgd
        method_name = f"gsvgd_effdim{eff_dim}"

    elif args.method == "s-svgd":
        ## S-SVGD
        # sample from variational density
        print("Running S-SVGD")
        x_s_svgd = theta0.clone().requires_grad_()
        s_svgd = SlicedSVGDLR(distribution, device=device)

        start = time.time()
        x_s_svgd, metric_s_svgd = s_svgd.fit(
            samples=x_s_svgd, 
            n_epoch=epochs, 
            lr=lr_g,
            eps=lr,
            save_every=save_every,
            train_loader=train_loader,
            test_data=(X_test, y_test),
            valid_data=(X_valid, y_valid)
        )
        elapsed_time = time.time() - start

        fitted_method = s_svgd
        method_name = f"s-svgd_lrg{lr_g}"

    elif args.method == "hmc":
        import numpyro
        from numpyro.infer import MCMC, NUTS, HMCECS
        import numpyro.distributions as npr_dist
        import jax.random as random
        import jax.numpy as jnp

        def model(Y, X, subsample_size=None):
            _, dim = X.shape
            alpha = numpyro.sample("alpha", npr_dist.Gamma(a0, b0))
            mean = jnp.zeros(dim)
            var = 1 / alpha * jnp.ones(dim)
            w = numpyro.sample("w", npr_dist.Normal(mean, var))

            with numpyro.plate("N", X.shape[0], subsample_size=subsample_size) as idx:
                logits = X[idx] @ w
                numpyro.sample(
                    "Y", npr_dist.Bernoulli(logits=logits), obs=Y[idx]
                )

        def run_inference(model, rng_key, Y, X, a0, b0, dim, subsample_size):
            if subsample_size == None:
                print("Sampler: HMC")
                kernel = NUTS(model)
            else:
                print(f"Sampler: HMCECS with subsample size = {subsample_size}")
                inner_kernel = NUTS(model)
                svi_key, mcmc_key = random.split(rng_key)

                # find reference parameters for second order taylor expansion to estimate likelihood (taylor_proxy)
                optimizer = numpyro.optim.Adam(step_size=1e-3)
                guide = numpyro.infer.autoguide.AutoDelta(model)
                svi = numpyro.infer.SVI(model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())
                svi_result = svi.run(svi_key, 20000, Y, X, subsample_size)
                params, losses = svi_result.params, svi_result.losses
                ref_params = {"alpha": params["alpha_auto_loc"], "w": params["w_auto_loc"]}

                # taylor proxy estimates log likelihood (ll) by
                # taylor_expansion(ll, theta_curr) +
                #     sum_{i in subsample} ll_i(theta_curr) - taylor_expansion(ll_i, theta_curr) around ref_params
                proxy = HMCECS.taylor_proxy(ref_params)
                kernel = HMCECS(inner_kernel, num_blocks=100, proxy=proxy)

            thinning = 20
            mcmc = MCMC(
                kernel,
                num_warmup=20000,
                num_samples=10000 * thinning,
                num_chains=1,
                thinning=thinning,
                progress_bar=True,
            )
            start = time.time()
            mcmc.run(rng_key, Y, X, subsample_size)
            elapsed_time = time.time() - start
            mcmc.print_summary()
            print("\nMCMC elapsed time:", elapsed_time)
            return mcmc.get_samples(), elapsed_time, mcmc

        
        # set random seeds for HMC
        rng_key, rng_key_predict = random.split(random.PRNGKey(0))
        Y = jnp.array(y_train.cpu()).reshape((-1,))
        Y = Y.at[Y == -1].set(0.) # change labels from {-1, 1} to {0, 1}
        X = jnp.array(X_train.cpu())
        print("shape of input data to NUTS:", X.shape, "\nD:", D)
        print("To be saved to:", results_folder + f"/particles_hmc.p")
        
        subsample_size = None if args.subsample_size == 0 else args.subsample_size
        particles_dict, elapsed_time, hmc_res = run_inference(
            model, rng_key, Y, X, a0, b0, D-1, subsample_size=subsample_size)
        theta = np.hstack([particles_dict["w"], particles_dict["alpha"].reshape((-1, 1))])
        theta = torch.Tensor(theta).to(device)
        print("shape of results:", theta.shape)
        method_name = "hmc"

        test_prob, test_y_pred, test_acc, test_ll = [], [], [], []

    ## save results
    if args.method != "hmc":
        # particles
        particles_epochs = [tup[0] for tup in fitted_method.test_accuracy]
        test_acc = [tup[1] for tup in fitted_method.test_accuracy]
        valid_acc = [tup[1] for tup in fitted_method.valid_accuracy]
        particles = fitted_method.particles
    else:
        particles_epochs = []
        test_acc = [test_acc]
        valid_acc = []
        particles = theta

    print("saved to ", results_folder + f"/particles_{method_name}.p")
    pickle.dump(
        {
            **{"epochs": particles_epochs},
            **{"test_accuracy": test_acc},
            **{"valid_accuracy": valid_acc},
            **{"particles": particles},
            **{"nbatches": len(train_loader)},
            **{"elapsed_time": elapsed_time}
        },
        open(results_folder + f"/particles_{method_name}.p", "wb")
    )

    # target distributions
    torch.save(distribution, results_folder + '/target_dist.p')

    # data
    data = {"X_train": X_train, "y_train": y_train, "X_valid": X_valid, "y_valid": y_valid,
        "X_test": X_test, "y_test": y_test}
    torch.save(data, results_folder + '/data.p')
    