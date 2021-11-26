import torch
import numpy as np

class BayesianLR:
    """Bayesian Logistic Regression class
    """
    def __init__(self, X, Y, a0=1, b0=0.01):
        """ Y in \in{+1, -1}
        """
        self.X, self.Y = X, Y.T
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
        
        # log-prior of w given alpha
        logp_w = (
            0.5 * d * alpha.log() 
            - 0.5 * alpha * torch.einsum("ij, ij -> i", w, w).reshape((w.shape[0], 1))
        ) # nparticles

        # log-prior for alpha
        logp_alpha = (self.a0 - 1) * alpha.log() - self.b0 * alpha # nparticles

        logprob = loglik + logp_w + logp_alpha

        return logprob

    def evaluation(self, theta, X_test, y_test):
        theta = theta.clone().requires_grad_(False)
        w = theta[:, :-1]
        wx = w @ X_test.t()  # nparticles x ndata
        prob = 1 / (1 + (-wx).exp())  # nparticles x ndata
        prob = torch.mean(prob, axis=0, keepdim=True).t()  # ndata x 1
        y_pred = torch.Tensor.float(prob > 0.5)
        y_pred[y_pred == 0] = -1
        acc = torch.mean(torch.Tensor.float(y_pred == y_test)).cpu().item()

        wx_ave = torch.mean(wx, axis=0, keepdim=True).t() # ndata x 1
        ll = (
            -wx_ave * (1 - y_test)/2 + (1 + (-wx_ave).exp()).log()
        ).mean().cpu().item()

        return prob, y_pred, acc, ll
