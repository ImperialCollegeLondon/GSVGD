import torch
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm, trange

import sys
sys.path.append(".")
from src.utils import AdaGrad_update

class SVGD:
    def __init__(
        self,
        target: torch.distributions.Distribution,
        kernel: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device="cpu",
    ):
        """
        Args:
            target (torch.distributions.Distribution): Only require the log_probability of the target distribution e.g. unnormalised posterior distribution
            kernel (torch.nn.Module): [description]
            optimizer (torch.optim.Optimizer): [description]
        """
        self.p = target
        self.k = kernel
        self.optim = optimizer
        self.device = device

    def phi(self, X: torch.Tensor, **kwargs):
        """

        Args:
            X (torch.Tensor): Particles being transported to the target distribution

        Returns:
            phi (torch.Tensor): Functional gradient
        """
        # copy the data for X into X
        X_cp = X.clone().detach().requires_grad_()
        Y = X.clone().detach()

        log_prob = self.p.log_prob(X_cp, **kwargs)
        score_func = autograd.grad(log_prob.sum(), X_cp)[0]

        X_cp = X.clone().detach().requires_grad_()
        with torch.no_grad():
            self.k.bandwidth(X, X)
        K_XX = self.k(X_cp, Y)
        grad_K = -autograd.grad(K_XX.sum(), X_cp)[0]

        # phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)
        attraction = K_XX.detach().matmul(score_func) / X.size(0)
        repulsion = grad_K / X.size(0)
        phi = attraction + repulsion
        # print("debug attrac", K_XX.detach()[0, :] @ score_func[:, 0], torch.isnan(K_XX).sum(), torch.isnan(score_func).sum())
        # print("k_xx", K_XX)
        # ind = torch.Tensor(np.array([[float(i)]*score_func.shape[1] for i in range(score_func.shape[0])])) #, device=score_func.device)
        # print("score", ind[torch.isnan(score_func)])
        # print("K-xx", K_XX[:3, :5])
        # print("score", score_func[:3, :5])
        # print("attraction", attraction[:3, :5])
        # print("repulsion", torch.isnan(repulsion).sum(), repulsion.shape)

        return phi, repulsion

    def step(self, X: torch.Tensor, **kwargs):
        """Gradient descent step

        Args:
            X (torch.Tensor): Particles to transport to the target distribution
        """
        self.optim.zero_grad()
        phi, repulsion = self.phi(X, **kwargs)
        X.grad = -phi
        self.optim.step()
        # #? Adagrad update
        # phi, repulsion = self.phi(X, **kwargs)
        # self.optim.zero_grad()
        # X, self.adagrad_state_dict = AdaGrad_update(X, phi, self.lr, self.adagrad_state_dict)

        # particle-averaged magnitude
        pam = torch.max(phi.detach().abs(), dim=1)[0].mean()
        pamrf = torch.max(repulsion.detach().abs(), dim=1)[0].mean()

        return pam.item(), pamrf.item()

    def fit(self, x0: torch.Tensor, epochs: torch.int64, verbose: bool = True,
        metric: callable = None,
        save_every: int = 100,
        threshold: float = 0
    ):
        """
        Args:
            x0 (torch.Tensor): Initial set of particles to be updated
            epochs (torch.int64): Number of gradient descent iterations
        """
        self.metrics = [0] * (epochs//save_every)
        self.particles = [0] * (1 + epochs//save_every)
        self.particles[0] = x0.clone().detach().cpu()
        self.pam = [0] * (epochs//save_every)
        self.pamrf = [0] * (epochs//save_every)

        iterator = tqdm(range(epochs)) if verbose else range(epochs)

        #? Adagrad update
        self.adagrad_state_dict = {
            'M': torch.zeros(x0.shape, device=self.device),
            'V': torch.zeros(x0.shape, device=self.device),
            't': 1,
            'beta1': 0.9,
            'beta2': 0.99
        }
        
        for i in iterator:
            pam, pamrf = self.step(x0)
            if (i+1) % save_every == 0:
                # self.metrics[i//save_every] = metric(x0.detach())
                self.particles[1 + i//save_every] = x0.clone().detach().cpu()
                # self.particles[1 + i//save_every] = x0.clone().detach()
                self.pam[i//save_every] = pam
                self.pamrf[i//save_every] = pamrf

            # # early stop
            # if pam < threshold:
            #     print(f"GSVGD converged in {i+1} epochs as PAM {pam} is less than {threshold}")
            #     break
        
        return self.metrics

class SVGDLR(SVGD):
    def fit(self, x0: torch.Tensor, epochs: torch.int64, verbose: bool = True,
        metric: callable = None,
        save_every: int = 100,
        threshold: float = 0,
        train_loader = None,
        valid_data = None,
        test_data = None
    ):
        """
        Args:
            x0 (torch.Tensor): Initial set of particles to be updated
            epochs (torch.int64): Number of gradient descent iterations
        """
        self.metrics = [0] * (epochs//save_every)
        self.particles = [x0.clone().detach().cpu()]
        self.pam = [0] * (epochs//save_every)
        self.test_accuracy = []
        self.valid_accuracy = []

        X_valid, y_valid = valid_data
        X_test, y_test = test_data

        # iterator = tqdm(range(epochs)) if verbose else range(epochs)
        iterator = trange(epochs) if verbose else range(epochs)

        for ep in iterator:
            for j, (X_batch, y_batch) in enumerate(train_loader):
                pam, pamrf = self.step(x0, X_batch=X_batch, y_batch=y_batch)
                train_steps = ep * len(train_loader) + j
        
                if train_steps % save_every == 0:
                    self.particles.append((ep, x0.clone().detach()))
                    _, _, test_acc, test_ll = self.p.evaluation(x0.clone().detach(), X_test, y_test)
                    valid_prob, _, valid_acc, valid_ll = self.p.evaluation(x0.clone().detach(), X_valid, y_valid)
                    # print(valid_prob.min(), valid_prob.max())
                    self.test_accuracy.append((train_steps, test_acc, test_ll))
                    self.valid_accuracy.append((train_steps, valid_acc, valid_ll))

                    if train_steps % 100 == 0:
                        iterator.set_description(f"Epoch {ep} batch {j} accuracy: {valid_acc}, ll: {valid_ll}, class 1 prop:, {((y_valid==1).sum()/y_valid.shape[0]).item()}")
                        # print(f"Epoch {ep} batch {j} accuracy:", valid_acc, 
                        #     "ll:", valid_ll,
                        #     "class 1 prop:", ((y_valid==1).sum()/y_valid.shape[0]).item())

            if metric and (ep + 1) % save_every==0:
                self.particles[1 + ep//save_every] = x0.clone().detach().cpu()
        
        return self.metrics
