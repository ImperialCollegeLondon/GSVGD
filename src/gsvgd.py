import pandas as pd
import torch
import torch.autograd as autograd
from torch import einsum
from tqdm import tqdm, trange
import autograd.numpy as np

import sys
sys.path.append(".")
from src.utils import AdaGrad_update


class FullGSVGDBatch:
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, T=1, device="cpu", noise=True):
        self.target = target
        self.k = kernel
        self.optim = optimizer
        self.manifold = manifold
        self.delta = delta
        self.sigma_list = []
        self.T = T
        self.device = device
        self.noise = noise
        self.langevin = torch.distributions.Normal(
            torch.zeros(1).to(device), torch.tensor(1).to(device)
        )
        self.alpha_tup = []

    def alpha(self, X, A, m, **kwargs):
        '''A here has orthonormal cols, so is actually P^T in the notes
        '''
        M = int(A.shape[1]/m) # proj_dim
        assert M == A.shape[1]/m, "calculation of projected dim is wrong!"
        num_particles = X.shape[0]

        X_cp = X.clone().detach().requires_grad_()
        Y_cp = X.clone().detach().requires_grad_()
        assert X.requires_grad is False, "X needs to be detached"

        # PX
        XP = X.detach() @ A
        Xr = XP.reshape((num_particles, m, M))
        YP = X.clone().detach() @ A
        Yr = YP.reshape((num_particles, m, M))

        # score
        # num_particles
        lp_X = self.target.log_prob(X_cp, **kwargs)
        lp_Y = self.target.log_prob(Y_cp, **kwargs)
        # num_particles x dim
        score_X = autograd.grad(lp_X.sum(), X_cp)[0]
        score_Y = autograd.grad(lp_Y.sum(), Y_cp)[0]

        # num_particles x d 
        A_score_X = score_X @ A
        A_score_Y = score_Y @ A
        # num_particles x m x proj_dim
        A_score_r_X = A_score_X.reshape((num_particles, m, M))
        A_score_r_Y = A_score_Y.reshape((num_particles, m, M))

        # Gram matrix (num_particles x num_particles x m)
        with torch.no_grad():
            self.k.bandwidth(Xr, Yr)
        K_AXAY = self.k(Xr, Yr)
        # proj_dim x num_particles x num_particles x m
        grad_second_K_Px = self.k.grad_second(Xr, Yr)
        grad_first_K_Px = self.k.grad_first(Xr, Yr)
        # dim x dim x num_particles x num_particles x m
        gradgrad_K_Px = self.k.gradgrad(Xr, Yr)
        
        # num_particles x num_particles x m
        div_K_Px = einsum('iiklm->klm', gradgrad_K_Px)

        # term 1
        prod = einsum("imk, jmk, ijm -> ijm", A_score_r_X, A_score_r_Y, K_AXAY)
        term1 = prod.sum()

        # term 2
        term2 = einsum("imj, jikm -> ikm", A_score_r_X, grad_second_K_Px).sum()

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = einsum("kmj, jikm -> ikm", A_score_r_Y, grad_first_K_Px).sum()

        # term 4
        term4 = div_K_Px.sum()

        return (term1 + term2 + term3 + term4) / num_particles**2

    def phi(self, X, A, m, **kwargs):
        '''Optimal test function
        Args:
            m: num of batches
        Output: 
            torch.Tensor of shape (m, n, dim).
        '''
        assert A.requires_grad is False, "A needs to be detached"
        dim = X.shape[1]
        M = int(A.shape[1]/m)
        num_particles = X.shape[0]
        
        # detach from current graph and create a new graph
        X_cp = X.clone().detach().requires_grad_()
        X = X.detach()
        assert X.requires_grad is False, "X needs to be detached"
        
        AX = X @ A
        AXr = AX.reshape((num_particles, m, M)).detach()
        assert AXr.requires_grad is False, "AXr needs to be detached"
        AY = X.clone() @ A
        AYr = AY.reshape((num_particles, m, M)).detach()

        # score
        log_prob = self.target.log_prob(X_cp, **kwargs)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X_cp)[0]
        assert score_func.requires_grad is False, "score needs to be detached"

        # num_particles x num_particles x batch 
        # -> batch x num_particles x num_particles (assuming symmetirc kernel)
        with torch.no_grad():
            self.k.bandwidth(AXr, AYr)
        K_AxAx = self.k(AXr, AYr)
        K_AxAx_trasp = K_AxAx.permute(2, 0, 1)
        # proj_dim x num_particles x num_particles x batch
        grad_first_K_Ax = self.k.grad_first(AXr, AYr)
        # batch x num_particles x num_particles x proj_dim
        grad_first_K_Ax = torch.transpose(grad_first_K_Ax, dim0=0, dim1=3)
        # batch x num_particles x proj_dim
        sum_grad_first_K_Ax = torch.sum(grad_first_K_Ax, dim=1)

        # compute phi
        A_r = A.reshape((dim, m, M)) # dim x batch x proj_dim
        A_r = torch.transpose(A_r, dim0=0, dim1=1) # batch x dim x proj_dim

        A_score = score_func @ A # num_particles x (proj_dim x m)
        # num_particles x batch x proj_dim
        A_score_r = A_score.reshape((num_particles, m, M))
        # batch x num_particles x dim
        AT_A_score_r = einsum("bij, kbj -> bki", A_r, A_score_r)

        # compute GSVGD update rule
        attraction = torch.einsum(
            "bjk, bji -> bik",
            AT_A_score_r,
            K_AxAx_trasp.detach()
        ) / num_particles

        repulsion = torch.einsum(
            "bdj, bij -> bid", A_r, sum_grad_first_K_Ax
        ) / num_particles

        phi = attraction + repulsion

        return phi, repulsion, AT_A_score_r, K_AxAx


    def update_projection(self, X, A, dim, m, M, **kwargs):
        """
        Inputs:
            A: dim x dim
        Return:
            A: dim x dim
        """
        # update A
        alpha = self.alpha(X, A, m, **kwargs)
        grad_A = autograd.grad(alpha, A)[0]

        with torch.no_grad():
            A_r_noise = self.langevin.sample((m, self.manifold._n, self.manifold._p))
            grad_A_r = grad_A.reshape((dim, m, M)).permute(1, 0, 2) # batch x dim x proj_dim
            A_r = A.reshape((dim, m, M)).permute(1, 0, 2) # batch x dim x proj_dim

            # Riemannian update
            A_r = self.manifold.retr(
                A_r.clone(),
                self.manifold.egrad2rgrad(
                    A_r.clone(),
                    self.delta * grad_A_r.clone() \
                    + np.sqrt(2.0 * self.T * self.delta)
                    * A_r_noise.squeeze(-1),
                ),
            ) # batch x dim x proj_dim

            A = torch.cat([A_r[b, :, :] for b in range(m)], dim=1).to(X.device) # dim x (dim * proj_dim)

        return A, alpha


    def step(self, X, A, m, **kwargs):

        # update A
        dim = X.shape[1]
        M = int(A.shape[1]/m)

        # dim x dim
        X_alpha_cp = X.clone().detach()
        A = A.detach().requires_grad_()
        A, alpha = self.update_projection(X_alpha_cp, A, dim, m, M, **kwargs)

        A = A.detach().requires_grad_()
        A_n = A.clone().detach()

        ## compute phi
        phi, repulsion, score, K_AxAx = self.phi(X=X, A=A_n, m=m, **kwargs)
        
        ## update x
        self.optim.zero_grad()
        X.grad = -einsum("bij -> ij", phi)
        self.optim.step()

        return A, phi, repulsion, score, K_AxAx

    def fit(self, X, A, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0, **kwargs):
        '''Run GSVGD
        Args:
            m: number of projection matrices.
        '''
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = X.clone().detach()
        self.U_list = [0] * (1 + epochs//save_every)
        self.U_list[0] = A.clone().detach()
        self.pam = [0] * (epochs//save_every)
        self.pamrf = [0] * (epochs//save_every)
        self.phi_list = [0] * (epochs//save_every)
        self.repulsion_list = [0] * (epochs//save_every)
        self.score_list = [0] * (epochs//save_every)
        self.k_list = [0] * (epochs//save_every)
        pam_old = 1e5

        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        self.lr = self.optim.state_dict()["param_groups"][0]["lr"]
        self.adagrad_state_dict = {
            'M': torch.zeros(X.shape, device=self.device),
            'V': torch.zeros(X.shape, device=self.device),
            't': 1,
            'beta1': 0.9,
            'beta2': 0.99
        }

        for i in iterator:
            A, phi, repulsion, score, K_AxAx = self.step(X=X, A=A.detach().requires_grad_(), m=m, **kwargs)

            ## This is slow, so we only do it once every 1000 timesteps
            if ((i+1) % 1000) == 0:
                A, _ = torch.qr(A)

            ## PAM and annealling variance multiplier
            perturbation = torch.sum(phi.detach().clone(), dim=0)
            pert_norm = torch.max(perturbation.abs(), dim=1)[0]
            pam = pert_norm.mean().item()

            perturbation_rf = torch.sum(repulsion.detach().clone(), dim=0)
            pert_rf_norm = torch.max(perturbation_rf.abs(), dim=1)[0]
            pamrf = pert_rf_norm.mean().item()

            pam_diff = np.abs(pam - pam_old)
            ## annealing for temperature param T
            if pam_diff < threshold and self.T < 1e6:
                self.T *= 10
            pam_old = pam

            ## save results
            if (i+1) % save_every==0:
                self.U_list[1 + i//save_every] = A.clone().detach()
                self.particles[1 + i//save_every] = X.clone().detach().cpu()
                self.pam[i//save_every] = pam
                self.pamrf[i//save_every] = pamrf
                self.phi_list[i//save_every] = phi
                self.repulsion_list[i//save_every] = repulsion
                self.score_list[i//save_every] = score
                self.k_list[i//save_every] = K_AxAx

        return A, self.metrics


class FullGSVGDBatchLR(FullGSVGDBatch):
    def fit(self, X, A, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0,
            train_loader=None, test_data=None, valid_data=None):
        '''Run GSVGD
        Args:
            m: number of projection matrices.
        '''
        self.metrics = [0] * (epochs//save_every)
        self.particles = [X.clone().detach()]
        self.U_list = [0] * (1 + epochs//save_every)
        self.U_list[0] = A.clone().detach()
        self.pam = [0] * (epochs//save_every)
        self.test_accuracy = []
        self.valid_accuracy = []
        pam_old = 1e5

        X_valid, y_valid = valid_data
        X_test, y_test = test_data

        iterator = trange(epochs) if verbose else range(epochs)

        for i in iterator:
            for j, (X_batch, y_batch) in enumerate(train_loader):
                A, phi, repulsion, score, K_AxAx = self.step(X=X, A=A.detach().requires_grad_(), m=m, 
                    X_batch=X_batch, y_batch=y_batch)

                ## This is slow, so we only do it once every 1000 timesteps
                if (j+1) % 1000 == 0:
                    A, _ = torch.qr(A)

                ## PAM and annealling variance multiplier
                perturbation = torch.sum(phi.detach().clone(), dim=0)
                pert_norm = torch.max(perturbation.abs(), dim=1)[0]
                pam = pert_norm.mean().item()

                pam_diff = np.abs(pam - pam_old)
                ## annealing for temperature param T
                if pam_diff < threshold and self.T < 1e6:
                    self.T *= 10
                pam_old = pam

                ## save results
                train_step = i * len(train_loader) + j
                if train_step % save_every == 0:
                    self.particles.append((i, X.clone().detach()))
                    _, _, test_acc, test_ll = self.target.evaluation(X.clone().detach(), X_test, y_test)
                    valid_prob, _, valid_acc, valid_ll = self.target.evaluation(X.clone().detach(), X_valid, y_valid)
                    self.test_accuracy.append((train_step, test_acc, test_ll))
                    self.valid_accuracy.append((train_step, valid_acc, valid_ll))

                    if train_step % 100 == 0:
                        iterator.set_description(f"Epoch {i} batch {j} accuracy: {valid_acc} ll: {valid_ll}")

        return A, self.metrics
