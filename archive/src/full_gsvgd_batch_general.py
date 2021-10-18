import pandas as pd
import torch
import torch.autograd as autograd
from torch import einsum
from tqdm import tqdm
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

    def alpha(self, X, P, m, **kwargs):
        '''A here has orthonormal cols, so is actually P^T in the notes
            P = [A, B]: dim x (2*dim)
        '''
        A = P[:, :, 0]
        B = P[:, :, 1]

        dim = X.shape[1]
        M = int(A.shape[1]/m)
        num_particles = X.shape[0]

        X_cp = X.clone().detach().requires_grad_()
        Y_cp = X.clone().detach().requires_grad_()
        
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
        B_score_X = score_X @ B
        B_score_Y = score_Y @ B
        # num_particles x m x proj_dim
        B_score_r_X = B_score_X.reshape((num_particles, m, M))
        B_score_r_Y = B_score_Y.reshape((num_particles, m, M))

        # Gram matrix (num_particles x num_particles x m)
        with torch.no_grad():
            self.k.bandwidth(Xr, Yr)
        K_AxAy = self.k(Xr, Yr)

        # proj_dim x num_particles x num_particles x m
        grad_first_K_Px = self.k.grad_first(Xr, Yr)
        grad_second_K_Px = self.k.grad_second(Xr, Yr)
        # proj_dim x proj_dim x num_particles x num_particles x m
        gradgrad_K_Px = self.k.gradgrad(Xr, Yr)

        # num_particles x num_particles x m
        A_r = A.reshape((dim, m, M)) # dim x m x proj_dim
        B_r = B.reshape((dim, m, M)) # dim x m x proj_dim
        BAt = einsum("imj, imk -> jmk", B_r, A_r) # proj_dim x m x proj_dim
        ABt_BAt = einsum("jmi, jmk -> imk", BAt, BAt) # proj_dim x m x proj_dim
        # proj_dim x proj_dim x num_particles x num_particles x m
        term4_prod = einsum("imj, jklrm -> iklrm", ABt_BAt, gradgrad_K_Px) 
        div_K_Px = einsum('iiklm->klm', term4_prod) # num_particles x num_particles x m

        # term 1
        prod = einsum("imk, jmk, ijm -> ijm", B_score_r_X, B_score_r_Y, K_AxAy)
        term1 = prod.sum()

        # term 2
        B_score_r_X_BAt = einsum("imj, jmk -> imk", B_score_r_X, BAt) # num_particles x m x proj_dim
        term2 = einsum("imj, jikm -> ikm", B_score_r_X_BAt, grad_second_K_Px).sum()

        # term 3 (NOT equal to term 2 for a general kernel!)
        B_score_r_Y_BAt = einsum("imj, jmk -> imk", B_score_r_Y, BAt) # num_particles x m x proj_dim
        term3 = einsum("kmj, jikm -> ikm", B_score_r_Y_BAt, grad_first_K_Px).sum()
        # assert term2 == term3, f"term 2 and 3 should be the same, {term2}, {term3}"

        # term 4
        ## compute grad_grad_K directly in matrix form
        term4 = div_K_Px.sum()

        return (term1 + term2 + term3 + term4) / num_particles**2


    def phi(self, X, A, B, m, **kwargs):
        '''Optimal test function
        Args:
            m: num of batches
        Output: 
            torch.Tensor of shape (m, n, dim).
        '''
        assert A.requires_grad is False, "A needs to be detached"
        assert B.requires_grad is False, "B needs to be detached"
    
        # detach from current graph and create a new graph
        X_cp = X.clone().detach().requires_grad_()
        X = X.detach()
    
        dim = X.shape[1]
        M = int(A.shape[1]/m)
        num_particles = X.shape[0]

        AX = X.detach() @ A
        AXr = AX.reshape((num_particles, m, M)).detach()
        #! try this!
        # AY = X.clone().detach() @ A
        # AYr = AY.reshape((num_particles, m, M)).detach()
        assert AXr.requires_grad is False, "AXr needs to be detached"

        log_prob = self.target.log_prob(X_cp, **kwargs)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X_cp)[0].detach()
        assert score_func.requires_grad is False, "score needs to be detached"

        # num_particles x num_particles x batch 
        # -> batch x num_particles x num_particles (#! assuming symmetirc kernel)
        with torch.no_grad():
            self.k.bandwidth(AXr, AXr)
        K_AxAx = self.k(AXr, AXr)
        K_AxAx_trasp = K_AxAx.permute(2, 0, 1)
        # proj_dim x num_particles x num_particles x batch
        # -> batch x num_particles x num_particles x proj_dim
        # -> batch x num_particles x proj_dim
        grad_first_K_Ax = self.k.grad_first(AXr, AXr)
        grad_first_K_Ax = torch.transpose(grad_first_K_Ax, dim0=0, dim1=3)
        sum_grad_first_K_Ax = torch.sum(grad_first_K_Ax, dim=1)

        # compute phi
        # batch x dim x proj_dim
        A_r = A.reshape((dim, m, M))
        A_r = torch.transpose(A_r, dim0=0, dim1=1)
        B_r = B.reshape((dim, m, M))
        B_r = torch.transpose(B_r, dim0=0, dim1=1)

        # num_particles x dim
        B_score = score_func @ B
        # num_particles x batch x proj_dim
        B_score_r = B_score.reshape((num_particles, m, M))
        # batch x num_particles x dim
        BT_B_score_r = einsum("bij, kbj -> bki", B_r, B_score_r)

        BAt =  einsum("bij, bik -> jbk", B_r, A_r) # proj_dim x batch x proj_dim
        BtBAt = einsum("bij, jbk -> bik", B_r, BAt) # batch x dim x proj_dim

        # batch x num_particles x dim
        attraction = torch.einsum(
            "bjk, bji -> bik",
            BT_B_score_r,
            K_AxAx_trasp.detach()
        ) / num_particles

        repulsion = torch.einsum(
            "bdj, bij -> bid", BtBAt, sum_grad_first_K_Ax
        ) / num_particles
        phi = attraction + repulsion

        return phi, repulsion, BT_B_score_r, K_AxAx


    def update_projection(self, X, A, B, dim, m, M, **kwargs):
        """
            A_r: batch x dim x proj_dim
            grad_A: dim x dim

        Return:
            A: dim x dim
        """
        ## update A using RGD
        P = torch.stack([A, B], dim=2).to(X.device).detach()
        P_n = P.clone().requires_grad_()
        alpha = self.alpha(X=X, P=P_n, m=m, **kwargs)
        # print("alpha", alpha.item())
        grad_P = autograd.grad(alpha, P_n)[0]
        grad_A = grad_P[:, :, 0]
        grad_B = grad_P[:, :, 1]
        # print("grad_A", grad_A.cpu().numpy()[:3, :5])

        mat_new = [0, 0]
        with torch.no_grad():
            for i in [0, 1]:
                grad_U = grad_P[:, :, i]
                grad_U_r = grad_U.reshape((dim, m, M)).permute(1, 0, 2) # batch x dim x proj_dim
                
                U = P[:, :, i]
                U_r = U.clone().reshape((dim, m, M)).permute(1, 0, 2) # batch x dim x proj_dim
                
                U_r_noise = self.langevin.sample((m, self.manifold._n, self.manifold._p))

                U_r = self.manifold.retr(
                    U_r.clone(),
                    self.manifold.egrad2rgrad(
                        U_r.clone(),
                        self.delta * grad_U_r.clone()
                    ),
                )

                # U_new = U_r.permute(1, 0, 2).reshape(A.shape) # dim x (proj_dim x m)
                U_new = torch.cat([U_r[b, :, :] for b in range(m)], dim=1).to(X.device) 
                mat_new[i] = U_new.clone()

        return mat_new[0], mat_new[1], alpha


    def step(self, X, A, B, m, **kwargs):
        # update A
        dim = A.shape[0]
        M = int(A.shape[1]/m)

        # dim x dim
        X_alpha_cp = X.clone().detach()
        A, B, alpha = self.update_projection(X_alpha_cp, A, B, dim, m, M, **kwargs)

        A = A.detach().requires_grad_()
        B = B.detach().requires_grad_()
        A_n = A.clone().detach()
        B_n = B.clone().detach()

        ## compute phi
        # print("A after", A_n[:3, :5])
        # print("before", X[:3, :5])
        phi, repulsion, score, K_AxAx = self.phi(X=X, A=A_n, B=B_n, m=m, **kwargs)
        
        ## update x
        # self.optim.zero_grad()
        # X.grad = -einsum("bij -> ij", phi)
        # self.optim.step()
        #? Adagrad update
        self.optim.zero_grad()
        X, self.adagrad_state_dict = AdaGrad_update(X, einsum("bij -> ij", phi), self.lr, self.adagrad_state_dict)

        return A, phi, repulsion, score, K_AxAx

    def fit(self, X, A, B, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0, **kwargs):
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
            A, phi, repulsion, score, K_AxAx = self.step(X, A, B, m, **kwargs)

            # #This is slow, so we only do it once every XXX timesteps (or maybe not at all?)
            # if ((i+1) % 1000) == 0:
            #     A, _ = torch.qr(A)

            # # # PAM and annealling variance multiplier
            perturbation = torch.sum(phi.detach().clone(), dim=0)
            pert_norm = torch.max(perturbation.abs(), dim=1)[0]
            pam = pert_norm.mean().item()

            perturbation_rf = torch.sum(repulsion.detach().clone(), dim=0)
            pert_rf_norm = torch.max(perturbation_rf.abs(), dim=1)[0]
            pamrf = pert_rf_norm.mean().item()

            pam_diff = np.abs(pam - pam_old)
            if pam_diff < threshold:
                # print(f"Increase T at iteration {i+1} as delta PAM {pam_diff} is less than {threshold}")
                self.T = np.min([10 * self.T, 1e6])
            pam_old = pam

            if metric and (i+1) % save_every==0:
                self.U_list[1 + i//save_every] = A.detach().clone()
                self.particles[1 + i//save_every] = X.clone().detach().cpu()
                self.pam[i//save_every] = pam
                self.pamrf[i//save_every] = pamrf
                self.phi_list[i//save_every] = phi
                self.repulsion_list[i//save_every] = repulsion
                self.score_list[i//save_every] = score
                self.k_list[i//save_every] = K_AxAx

        return A, self.metrics

class FullGSVGDBatchLR(FullGSVGDBatch):
    def fit(self, X, U, m, epochs=100, verbose=True, metric=None, save_every=100, threshold=0,
            train_loader=None, test_data=None, valid_data=None):
        '''Run GSVGD
        Args:
            m: number of projection matrices.
        '''
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = X.clone().detach()
        self.U_list = [0] * (1 + epochs//save_every)
        self.U_list[0] = U.clone().detach()
        self.pam = [0] * (epochs//save_every)
        self.test_accuracy = []
        self.valid_accuracy = []
        pam_old = 1e5

        X_valid, y_valid = valid_data
        X_test, y_test = test_data

        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        for i in iterator:
            for j, (X_batch, y_batch) in enumerate(train_loader):
                U, phi = self.step(X, U, m, X_batch, y_batch)

                # # orthogonalize projections
                # if (j+1) % 1000 == 0:
                #     U, _ = torch.qr(U)
                #     U = U.detach().requires_grad_()
                    
                
                # PAM and annealling variance multiplier
                # if (j+1) % 100 == 0:
                perturbation = torch.sum(phi.detach().clone(), dim=0)
                pam = torch.linalg.norm(perturbation, dim=1)[0].mean() / X.shape[1]
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

        return U, self.metrics

class FullGSVGDBatchBNN(FullGSVGDBatch):
    def __init__(self, target, kernel, optimizer, manifold, delta=0.1, device="cpu", noise=True, T=1e-4):
        super().__init__(target, kernel, optimizer, manifold, delta, T, device, noise)
    def alpha(self, X, P, m, features, labels, n_train):
        '''A here has orthonormal cols, so is actually P^T in the notes
        '''
        # M = int(X.shape[1]/m)
        M = int(P.shape[1]/m)
        num_particles = X.shape[0]
        # PX
        XP = X.detach() @ P
        Xr = XP.reshape((num_particles, m, M))
        #? correct

        # score
        # num_particles
        lp = self.target.log_prob(features, labels, n_train, X)
        # num_particles x dim
        score = autograd.grad(lp.sum(), X)[0]
        # num_particles x d 
        P_score = score @ P
        # num_particles x m x proj_dim
        P_score_r = P_score.reshape((num_particles, m, M))
        #? correct

        # Gram matrix (num_particles x num_particles x m)
        #! should we detach here (old code is detaching)?
        K_PxPx = self.k(Xr, Xr.detach())
        # print(Xr[:, 0, :])
        # print("k", K_PxPx[:, :, 0])
        #? correct
        # proj_dim x num_particles x num_particles x m
        grad_first_K_Px = self.k.grad_first(Xr, Xr)
        # #? correct
        # dim x dim x num_particles x num_particles x m
        gradgrad_K_Px = self.k.gradgrad(Xr, Xr)
        #? correct
        # num_particles x num_particles x m
        div_K_Px = einsum('iiklm->klm', gradgrad_K_Px)

        # term 1
        prod = einsum("imk, lmk, ilm -> ilm", P_score_r, P_score_r, K_PxPx)
        term1 = prod.sum()
        #? correct

        # term 2
        term2 = einsum("imj, jkim -> kim", P_score_r, grad_first_K_Px).sum()
        #? correct

        # term 3 (NOT equal to term 2 for a general kernel!)
        term3 = term2

        # term 4
        ## compute grad_grad_K directly in matrix form
        term4 = div_K_Px.sum()
        #? correct

        return (term1 + term2 + term3 + term4) / num_particles**2

    def phi(self, X: torch.Tensor, A, m, features, labels, n_train):
        dim = X.shape[1]
        # M = int(dim/m)
        M = int(A.shape[1]/m)
        num_particles = X.shape[0]
        # detach from current graph and create a new graph
        X = X.detach().requires_grad_(True)
        A = A.detach().requires_grad_(True)
        # TODO check later whether need detach
        # nparticles x dim -> nparticles x dim
        # Will it work with nparticles x dim -> nparticles x m*dim?
        AX = X.detach() @ A
        AXr = AX.reshape((num_particles, m, M))

        log_prob = self.target.log_prob(features, labels, n_train, X)
        # num_particles x dim
        score_func = autograd.grad(log_prob.sum(), X)[0]

        # num_particles x num_particles x batch 
        # -> batch x num_particles x num_particles (assuming symmetirc kernel)
        K_AxAx = self.k(AXr, AXr.detach())
        # print("K_Ax", K_AxAx[0, :, :])
        #! need to delete this line somehow
        K_AxAx_trasp = torch.transpose(K_AxAx, dim0=0, dim1=2)
        #? correct
        # proj_dim x num_particles x num_particles x batch
        # -> batch x num_particles x num_particles x proj_dim
        # -> batch x num_particles x proj_dim
        #! the old code has -1, but I think it should not be there
        grad_first_K_Ax = self.k.grad_first(AXr, AXr)
        grad_first_K_Ax = torch.transpose(grad_first_K_Ax, dim0=0, dim1=3)
        sum_grad_first_K_Ax = torch.sum(grad_first_K_Ax, dim=1)
        # print("gradK", sum_grad_first_K_Ax[0, :, :])
        #? correct

        # compute phi
        # num_particles x d 
        A_score = score_func @ A
        # num_particles x batch x proj_dim
        A_score_r = A_score.reshape((num_particles, m, M))
        # batch x dim x proj_dim
        A_r = A.reshape((dim, m, M))
        A_r = torch.transpose(A_r, dim0=0, dim1=1)
        # batch x num_particles x dim
        AT_A_score_r = einsum("bij, kbj -> bki", A_r, A_score_r)
        # batch x num_particles x dim
        phi = (
            torch.einsum(
                "bij, bjk -> bik",
                K_AxAx_trasp.detach(),
                AT_A_score_r
            )
            + torch.einsum(
                "bij, bkj -> bik", sum_grad_first_K_Ax, A_r
            )
        ) / num_particles
        #? correct

        # update A
        # dim x dim
        #? correct
        # A = A.detach().requires_grad_()
        #! grad_A is wrong
        # print("A", A)
        grad_A = autograd.grad(self.alpha(X, A, m, features, labels, n_train), A)[0]
        # print("sigma", self.k.sigma)
        # print("grad_A")
        # print(grad_A)
        # dim x batch x proj_dim
        # -> batch x dim x proj_dim
        #! grad_A_r is wrong, as grad_A is wrong
        grad_A_r = grad_A.reshape((dim, m, M))
        grad_A_r = torch.transpose(grad_A_r, dim0=0, dim1=1)
        # Add noise
        if self.noise:
            A_r = self.manifold.retr(
                A_r.detach(),
                self.manifold.egrad2rgrad(
                    A_r.detach(),
                    self.delta * grad_A_r.detach()
                    + np.sqrt(self.T * 2.0 * self.delta)
                    * self.langevin.sample((m, self.manifold._n, self.manifold._p)).squeeze(
                        -1
                    ),
                ),
            )
        else:
            A_r = self.manifold.retr(
                A_r.detach(),
                self.manifold.egrad2rgrad(
                    A_r.detach(),
                    self.delta * grad_A_r.detach()
                ),
            )
        # reconstruct full projection: dim x dim
        A = torch.cat([A_r[b, :, :] for b in range(m)], dim=1)
        A = A.detach().requires_grad_().to(X.device)
        return phi, A
# if __name__ == "__main__":
#     import sys
#     sys.path.append("./")
#     from experiments.synthetic import GaussianMixture
#     from src.kernel import BatchRBF
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
#     kernel = BatchRBF()
#     manifold = Grassmann(d, eff_dim)

#     nparticles = 5
#     X = torch.randn(nparticles, d).detach().requires_grad_(True)

#     full_gsvgd = FullGSVGD(target=dist, kernel=kernel, optimizer=optim.Adam([X], lr=1e-1), manifold=manifold)

#     U = np.eye(d)
#     full_gsvgd.alpha(X, U, batch)
#     # %timeit alpha(X, np.eye(d), 100.0, d, 2, nparticles)


#     U = torch.eye(d).requires_grad_(True)
#     # U = np.eye(d)
#     full_gsvgd.fit(X, U, batch, epochs=20)

#     print("done")

