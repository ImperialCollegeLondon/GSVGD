import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch import einsum
from tqdm import tqdm
import autograd.numpy as np
from pymanopt.manifolds import Grassmann


class FullGSVGD:
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, T=1, device="cpu", noise=True):
        self.target = target
        self.k = kernel
        self.optim = optimizer
        self.manifold = manifold
        self.delta = delta
        self.sigma_list = []
        self.device = device
        self.noise = noise
        self.T = T
        self.langevin = torch.distributions.Normal(
            torch.zeros(1).to(device), torch.tensor(1).to(device)
        )

    def alpha(self, X, A):
        '''A here has orthonormal cols, so is actually P^T in the notes
        '''
        # PX
        # X = X.detach().requires_grad_()
        # A = A.detach().requires_grad_()

        y0 = einsum("ij, jk -> ik", X.detach(), A)


        # score
        # num_particles
        log_prob = self.target.log_prob(X)
        # num_particles x dim
        score = autograd.grad(log_prob.sum(), X)[0]
        #! changed
        # num_particles x num_particles
        score2 = (score @ score.T).unsqueeze(-1)
        # num_particles x proj_dim
        A_score = einsum("ij, jk -> ik", score, A)

        # Gram matrix
        # num_particles x num_particles
        K_AxAx = self.k(y0, y0.detach())
        # proj_dim x num_particles x num_particles
        grad_first_K_Ax = self.k.grad_first(y0, y0)
        # proj_dim x proj_dim x num_particles x num_particles
        gradgrad_K_AxAx = self.k.gradgrad(y0, y0)

        # term 1
        #! changed
        # num_particles x num_particles
        # prod = einsum("ij, ij -> ij", A_score @ A_score.T, K_AxAx)
        prod = score2 * K_AxAx
        term1 = prod.sum()

        # term 2
        term2 = einsum("ij, jki -> ki", A_score, grad_first_K_Ax).sum()

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = term2

        # term 4
        ## compute grad_grad_K directly in matrix form
        term4 = einsum("iijk -> jk", gradgrad_K_AxAx).sum()

        return (term1 + term2 + term3 + term4) / X.shape[0]**2

    def phi(self, X, A):
        # detach from current graph and create a new graph
        X = X.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)
        # TODO check later whether need detach
        AX = (A.t() @ X.t()).t()

        log_prob = self.target.log_prob(X)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X)[0]

        # num_particles x num_particles
        K_XX = self.k(AX, AX.detach())
        # num_particles x proj_dim
        grad_K = self.k.grad_first(AX, AX)
        grad_K = einsum("ijk->ki", grad_K)

        # compute phi
        #! changed
        phi = (
            K_XX.detach().matmul(score_func)
            + torch.einsum("ij, jk -> ik", grad_K, A.t())
        ) / X.size(0)
        # phi = (
        #     K_XX.detach().matmul(torch.einsum("ij, jk -> ik", score_func, A @ A.t()))
        #     + torch.einsum("ij, jk -> ik", grad_K, A.t())
        # ) / X.size(0)

        # compute alpha(P)
        # dim x proj_dim
        grad_A = autograd.grad(self.alpha(X, A), A)[0]
        # Add noise
        if self.noise:
            A_noise = self.langevin.sample((self.manifold._n, self.manifold._p))
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                    + np.sqrt(2.0 * self.T * self.delta)
                    * A_noise.squeeze(-1),
                ),
            )
        else:
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                ),
            )
        return phi, A

    def step(self, X, U, m):

        U_new = torch.zeros_like(U).detach()
        M = U.shape[1] // m # proj dim

        sum_phi = 0
        for i in range(m):
            self.optim.zero_grad()
            phi, U_ = self.phi(X, U[:, (i*M):((i+1)*M)])
            sum_phi += phi.clone().detach()
            U_new[:, (i*M):((i+1)*M)] = U_
            X.grad = -phi
            self.optim.step()

        # sum_phi = 0
        # for i in range(U.shape[1]):
        #     phi, U_ = self.phi(X, U[:,i, None])
        #     U_new[:,i] = U_.reshape(-1)
        #     sum_phi += phi
        # self.optim.zero_grad()
        # X.grad = -sum_phi
        # self.optim.step()

        return U_new, sum_phi 

    def fit(self, X, U, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0):
        self.sigma_list = [0] * epochs
        self.U_list = [0] * epochs
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = X.clone().detach().cpu()
        self.pam = [0] * (epochs//save_every)
        pam_old = 1e5

        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        for i in tqdm(range(epochs)):
            U, sum_phi = self.step(X, U, m)

            # # This is slow, so we only do it once every XXX timesteps (or maybe not at all?)
            # if ((i + 1) % 1000) == 0:
            #     U, _ = torch.qr(U)

            # # PAM and annealling variance multiplier
            # if (i+1) % 100 == 0:
            #     pam = torch.linalg.norm(sum_phi, dim=1)[0].mean() / X.shape[1]
            #     if pam < threshold:
            #         # print(f"Increase T at epoch = {i+1} as PAM {pam} is less than {threshold}")
            #         self.T = np.min([10 * self.T, 1e6])

            # # PAM and annealling variance multiplier
            if (i+1) % 100 == 0:
                pam = torch.linalg.norm(sum_phi, dim=1)[0].mean() / X.shape[1]
                pam_diff = torch.abs(pam - pam_old)
                if pam_diff < threshold:
                    # print(f"Increase T at iteration {i+1} as delta PAM {pam_diff} is less than {threshold}")
                    self.T = np.min([10 * self.T, 1e6])
                pam_old = pam

            self.sigma_list[i] = self.k.sigma
            self.U_list[i] = U.cpu().detach().numpy()
            if metric and (i+1) % save_every==0:
                self.metrics[i//save_every] = metric(X.detach())
                self.particles[1 + i//save_every] = X.clone().detach().cpu()
                pam = torch.max(sum_phi, dim=1)[0].mean() / X.shape[1]
                self.pam[i//save_every] = pam

        return U, self.metrics


class FullGSVGDLR:
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, T=1, device="cpu", noise=True):
        self.target = target
        self.k = kernel
        self.optim = optimizer
        self.manifold = manifold
        self.delta = delta
        self.sigma_list = []
        self.device = device
        self.noise = noise
        self.T = T
        self.langevin = torch.distributions.Normal(
            torch.zeros(1).to(device), torch.tensor(1).to(device)
        )

    def alpha(self, X, A, X_batch, y_batch):
        '''A here has orthonormal cols, so is actually P^T in the notes
        '''
        # PX
        y0 = einsum("ij, jk -> ik", X.detach(), A)

        # score
        # num_particles
        log_prob = self.target.log_prob(X, X_batch, y_batch)
        # num_particles x dim
        score = autograd.grad(log_prob.sum(), X)[0]
        #! changed
        # # num_particles x num_particles
        # score2 = (score @ score.T).unsqueeze(-1)
        # num_particles x proj_dim
        A_score = einsum("ij, jk -> ik", score, A)

        # Gram matrix
        # K_xx = self.k(X, X.detach())
        # num_particles x num_particles
        K_AxAx = self.k(y0, y0.detach())
        # proj_dim x num_particles x num_particles
        grad_first_K_Ax = self.k.grad_first(y0, y0)
        # proj_dim x proj_dim x num_particles x num_particles
        gradgrad_K_AxAx = self.k.gradgrad(y0, y0)

        # term 1
        #! changed
        # num_particles x num_particles
        prod = einsum("ij, ij -> ij", A_score @ A_score.T, K_AxAx)
        # prod = score2 * K_AxAx
        term1 = prod.sum()

        # term 2
        term2 = einsum("ij, jki -> ki", A_score, grad_first_K_Ax).sum()

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = term2

        # term 4
        ## compute grad_grad_K directly in matrix form
        term4 = einsum("iijk -> jk", gradgrad_K_AxAx).sum()

        return (term1 + term2 + term3 + term4) / X.shape[0]**2

    def phi(self, X, A, X_batch, y_batch):
        # detach from current graph and create a new graph
        X = X.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)
        # TODO check later whether need detach
        AX = (A.t() @ X.t()).t()

        log_prob = self.target.log_prob(X, X_batch, y_batch)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X)[0]

        # num_particles x num_particles
        K_XX = self.k(AX, AX.detach())
        # num_particles x proj_dim
        # grad_K = -autograd.grad(K_XX.sum(), AX)[0]
        grad_K = self.k.grad_first(AX, AX)
        grad_K = einsum("ijk->ki", grad_K)

        # compute phi
        #! changed
        # phi = (
        #     K_XX.detach().matmul(score_func)
        #     + torch.einsum("ij, jk -> ik", grad_K, A.t())
        # ) / X.size(0)
        phi = (
            K_XX.detach().matmul(torch.einsum("ij, jk -> ik", score_func, A @ A.t()))
            + torch.einsum("ij, jk -> ik", grad_K, A.t())
        ) / X.size(0)

        # compute alpha(P)
        # dim x proj_dim
        #? correct
        grad_A = autograd.grad(self.alpha(X, A, X_batch, y_batch), A)[0]
        # Add noise
        if self.noise:
            A_noise = self.langevin.sample((self.manifold._n, self.manifold._p))
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                    + np.sqrt(2.0 * self.T * self.delta)
                    * A_noise.squeeze(-1),
                ),
            )
        else:
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                ),
            )
        return phi, A

    def step(self, X, U, m, X_batch, y_batch):

        U_new = torch.zeros_like(U).detach()
        M = U.shape[1] // m # proj dim

        sum_phi = 0
        for i in range(m):
            self.optim.zero_grad()
            phi, U_ = self.phi(X, U[:, (i*M):((i+1)*M)], X_batch, y_batch)
            sum_phi += phi.clone().detach()
            U_new[:, (i*M):((i+1)*M)] = U_
            X.grad = -phi
            self.optim.step()

        return U_new, sum_phi 

    def fit(self, X, U, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0,
            train_loader=None, test_data=None, valid_data=None):

        self.sigma_list = [0] * epochs
        self.U_list = [0] * epochs
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = X.clone().detach().cpu()
        self.pam = [0] * (epochs//save_every)
        self.test_accuracy = []
        self.valid_accuracy = []
        pam_old = 1e5

        X_valid, y_valid = valid_data
        X_test, y_test = test_data

        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for i in iterator:
            for j, (X_batch, y_batch) in enumerate(train_loader):
                U, sum_phi = self.step(X, U, m, X_batch, y_batch)

                # orthogonalize projections
                if (j+1) % 1000 == 0:
                    U, _ = torch.qr(U)
                    U = U.detach().requires_grad_()
                    
                
                # # PAM and annealling variance multiplier
                # pam = torch.linalg.norm(sum_phi, dim=1)[0].mean() / X.shape[1]
                # pam_diff = torch.abs(pam - pam_old)
                # if pam_diff < threshold:
                #     print(f"Increase T at iteration {j+1} as delta PAM {pam_diff} is less than {threshold}")
                #     self.T = np.min([10 * self.T, 1e6])
                # pam_old = pam
                
                # PAM and annealling variance multiplier
                if (j+1) % 100 == 0:
                    pam = torch.linalg.norm(sum_phi, dim=1)[0].mean() / X.shape[1]
                    pam_diff = torch.abs(pam - pam_old)
                    if pam_diff < threshold:
                        # print(f"Increase T at iteration {i+1} as delta PAM {pam_diff} is less than {threshold}")
                        self.T = np.min([10 * self.T, 1e6])
                    pam_old = pam

                if j % 100 == 0:
                    train_step = i * len(train_loader) + j
                    _, _, test_acc = self.target.evaluation(X.clone().detach(), X_test, y_test)
                    _, _, valid_acc = self.target.evaluation(X.clone().detach(), X_valid, y_valid)
                    self.test_accuracy.append((train_step, test_acc))
                    self.valid_accuracy.append((train_step, valid_acc))

                    if j % 1000 == 0:
                        print(f"Epoch {i} batch {j} accuracy:", valid_acc)

            self.sigma_list[i] = self.k.sigma
            self.U_list[i] = U.cpu().detach().numpy()
            if metric and (i+1) % save_every==0:
                self.metrics[i//save_every] = metric(X.detach())
                self.particles[1 + i//save_every] = X.clone().detach().cpu()
                pam = torch.max(sum_phi, dim=1)[0].mean() / X.shape[1]
                self.pam[i//save_every] = pam

        return U, self.metrics


class FullGSVGDBNN:
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, device="cpu", noise=True):
        self.target = target
        self.k = kernel
        self.optim = optimizer
        self.manifold = manifold
        self.delta = delta
        self.sigma_list = []
        self.device = device
        self.noise = noise
        self.langevin = torch.distributions.Normal(
            torch.zeros(1).to(device), torch.tensor(1).to(device)
        )

    def alpha(self, X, A, features, labels, n_train):
        '''A here has orthonormal cols, so is actually P^T in the notes
        '''
        # PX
        y0 = einsum("ij, jk -> ik", X.detach(), A)
        #? correct

        # score
        # num_particles
        log_prob = self.target.log_prob(features, labels, n_train, X)
        # num_particles x dim
        score = autograd.grad(log_prob.sum(), X)[0]
        # num_particles x proj_dim
        A_score = einsum("ij, jk -> ik", score, A)
        #? correct

        # Gram matrix
        # K_xx = self.k(X, X.detach())
        # num_particles x num_particles
        K_AxAx = self.k(y0, y0.detach())
        # print(y0)
        # print("k", K_AxAx)
        #? correct
        # proj_dim x num_particles x num_particles
        grad_first_K_Ax = self.k.grad_first(y0, y0)
        #? correct
        # proj_dim x proj_dim x num_particles x num_particles
        gradgrad_K_AxAx = self.k.gradgrad(y0, y0)
        #? correct

        # term 1
        # num_particles x num_particles
        prod = einsum("ij, ij -> ij", A_score @ A_score.T, K_AxAx)
        term1 = prod.sum()
        #? correct

        # term 2
        term2 = einsum("ij, jki -> ki", A_score, grad_first_K_Ax).sum()
        #? correct

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = term2

        # term 4
        ## compute grad_grad_K directly in matrix form
        term4 = einsum("iijk -> jk", gradgrad_K_AxAx).sum()
        #? correct

        return (term1 + term2 + term3 + term4) / X.shape[0]**2

    def phi(self, X, A, features, labels, n_train):
        # detach from current graph and create a new graph
        X = X.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)
        # TODO check later whether need detach
        AX = (A.t() @ X.t()).t()

        log_prob = self.target.log_prob(features, labels, n_train, X)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X)[0]

        # num_particles x num_particles
        K_XX = self.k(AX, AX.detach())
        # print("K_XX", K_XX)
        #? correct
        # num_particles x proj_dim
        # grad_K = -autograd.grad(K_XX.sum(), AX)[0]
        grad_K = self.k.grad_first(AX, AX)
        grad_K = einsum("ijk->ki", grad_K)
        #? correct

        # compute phi
        phi = (
            K_XX.detach().matmul(torch.einsum("ij, jk -> ik", score_func, A @ A.t()))
            + torch.einsum("ij, jk -> ik", grad_K, A.t())
        ) / X.size(0)
        #? correct

        # compute alpha(P)
        # dim x proj_dim
        #? correct
        grad_A = autograd.grad(self.alpha(X, A, features, labels, n_train), A)[0]
        # Add noise
        if self.noise:
            A_noise = self.langevin.sample((self.manifold._n, self.manifold._p))
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                    + np.sqrt(2.0 * self.delta)
                    * A_noise.squeeze(
                        -1
                    ),
                ),
            )
        else:
            A = self.manifold.retr(
                A.detach(),
                self.manifold.egrad2rgrad(
                    A.detach(),
                    self.delta * grad_A.detach()
                ),
            )
        return phi, A




# if __name__ == "__main__":
#     import sys
#     sys.path.append("./")
#     from experiments.synthetic import GaussianMixture
#     from src.kernel import RBF
#     import torch.optim as optim
#     from src.manifold import Grassmann

#     torch.manual_seed(0)
   
#     nparticles = 5
#     d=6
#     eff_dim=1
#     batch=int(d/eff_dim)
#     X = torch.randn(nparticles, d)
#     Y = torch.randn(nparticles, d)

#     dist = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
#     kernel = RBF()
#     manifold = Grassmann(d, eff_dim)

#     nparticles = 5
#     X = torch.randn(nparticles, d).detach().requires_grad_(True)

#     full_gsvgd = FullGSVGD(target=dist, kernel=kernel, optimizer=optim.Adam([X], lr=1e-1), manifold=manifold)

#     # U = np.eye(d)
#     # full_gsvgd.alpha(X, U, batch)
#     # %timeit alpha(X, np.eye(d), 100.0, d, 2, nparticles)


#     U = torch.eye(d).requires_grad_(True)
#     # U = np.eye(d)
#     full_gsvgd.fit(X, U, epochs=20)

#     print("done")
