import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
from src.Sliced_KSD_Clean.Util import *

class Bayesian_NN_eff(object):
    """Modified from code provided by Wenbo Gong.

    """
    def __init__(self, input_dim, n_h=50, device="cpu"):
        self.input_dim = input_dim
        self.n_h = n_h
        self.device = device

    def init_weights(self, n_particles, a0, b0, scale=1.0, flag_same=False):
        # Bayesian NN weight initialization
        # raise NotImplementedError
        if not flag_same:
            w1 = (
                scale
                / (np.sqrt(float(self.input_dim)) + 1)
                * torch.randn(n_particles, self.input_dim * self.n_h)
            )
            b1 = torch.zeros(n_particles, self.n_h)
            w2 = scale / (np.sqrt(float(self.n_h)) + 1) * torch.randn(n_particles, self.n_h)
            b2 = torch.zeros(n_particles, 1)
        else:
            #SVGD
            # w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
            # b1 = np.zeros((self.n_hidden,))
            # w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden)
            # b2 = 0.
            # loggamma = np.log(np.random.gamma(a0, b0))
            # loglambda = np.log(np.random.gamma(a0, b0))

            w1 = (
                scale
                / (np.sqrt(float(self.input_dim)) + 1)
                * torch.randn(1, self.input_dim * self.n_h).repeat(n_particles, 1)
            )
            b1 = torch.zeros(1, self.n_h).repeat(n_particles, 1)
            w2 = (
                scale
                / (np.sqrt(float(self.n_h)) + 1)
                * torch.randn(1, self.n_h).repeat(n_particles, 1)
            )
            b2 = torch.zeros(1, 1).repeat(n_particles, 1)

        loggamma = torch.from_numpy(np.log(np.random.gamma(a0, b0, n_particles))).float()
        loggamma = torch.unsqueeze(loggamma, dim=-1)  # n_p x 1
        loglambda = torch.from_numpy(np.log(np.random.gamma(a0, b0, n_particles))).float()
        loglambda = torch.unsqueeze(loglambda, dim=-1)
        W = torch.cat((w1, b1, w2, b2, loggamma, loglambda), dim=-1)  # n_p x ....
        
        self.a0 = a0
        self.b0 = b0
        self.scale = scale

        return W.to(self.device).double()

    def forward(self, X, W):
        # X is N x dim
        # W is n_particle x dim
        w1_idx = self.n_h * self.input_dim
        b1_idx = w1_idx + self.n_h
        n_p = W.shape[0]
        w1 = W[:, 0:w1_idx].reshape((n_p, self.input_dim, self.n_h))  # np x dim x out
        b1 = W[:, w1_idx:b1_idx].reshape((n_p, 1, self.n_h))  # np x 1 x out
        X = torch.unsqueeze(X, dim=0).matmul(w1) + b1  # np x N x n_h
        X = F.relu(X)

        w2_idx = b1_idx + self.n_h
        b2_idx = w2_idx + 1
        w2 = W[:, b1_idx:w2_idx].reshape((n_p, self.n_h, 1))  # np x nh x 1
        b2 = W[:, w2_idx:b2_idx].reshape(n_p, 1, 1)  # np x 1 x  1
        out = X.matmul(w2) + b2  # np x N x 1

        # prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2
        return out

    def log_prob(self, W, X_data, Y, n_train):
        """

        X: batch covariate vector
        Y: batch label vector
        n_train: total number of datapoints in training set
        """
        X = X_data
        loggamma = W[:, -2].unsqueeze(-1)  # np x 1
        loglambda = W[:, -1].unsqueeze(-1)  # np x 1

        Y_pred = self.forward(X, W).squeeze(-1)  # np x N
        #! this broadcasting might be wrong!
        Y = torch.unsqueeze(Y, dim=0)  # 1 x N
        log_lik_data = -0.5 * X.shape[0] * (np.log(2 * np.pi) - loggamma) - (
            torch.exp(loggamma) / 2
        ) * torch.sum(
            (Y_pred - Y) ** 2, dim=-1, keepdim=True
        )  # np x 1

        #! what is the extra loggamma at the end?
        log_prior_data = 1 * (
            (self.a0 - 1) * loggamma - self.b0 * torch.exp(loggamma) + loggamma
        )  # np x 1

        #! what is the extra loglambda at the end?
        log_prior_w = (
            -0.5 * (W.shape[-1] - 2) * (np.log(2 * np.pi) - loglambda)
            - (torch.exp(loglambda) / 2)
            * torch.sum(W[:, 0:-2] ** 2, dim=-1, keepdim=True)
            + 1 * ((self.a0 - 1) * loglambda - self.b0 * torch.exp(loglambda) + loglambda)
        )  # np x 1
        log_posterior = torch.squeeze(
            log_lik_data / X.shape[0] * n_train + log_prior_data + log_prior_w
        )  # np

        return log_posterior


    def lik(self, X, Y, W):
        """

        X: batch covariate vector
        Y: batch label vector
        n_train: total number of datapoints in training set
        """
        Y_pred = self.forward(X, W).squeeze(-1)

        loggamma = W[:, -2].unsqueeze(1)  # np x 1
        gamma = loggamma.exp()
        # gamma_posterior = gamma.mean()
        Y = torch.unsqueeze(Y, dim=0)  # 1 x N
        # print((Y_pred - Y).shape)
        # np x N
        #! should we take the mean before or after evaluation?
        #! Dilin's code is after
        log_lik_data = -0.5 * (np.log(2 * np.pi) - gamma.sqrt().mean().log()) + (
            -gamma / 2 * (Y_pred - Y) ** 2).mean(0).exp().log().mean()  # np x 1  
        return log_lik_data

def BNN_compute_score_eff(
    Net, W, X, Y, n_train, a0, b0, flag_gamma=True, flag_create=False
):
    loggamma = W[:, -2].unsqueeze(-1)  # np x 1
    loglambda = W[:, -1].unsqueeze(-1)  # np x 1

    Y_pred = torch.squeeze(Net.forward(X, W))  # np x N
    Y = torch.unsqueeze(Y, dim=0)  # 1 x N
    log_lik_data = -0.5 * X.shape[0] * (np.log(2 * np.pi) - loggamma) - (
        torch.exp(loggamma) / 2
    ) * torch.sum(
        (Y_pred - Y) ** 2, dim=-1, keepdim=True
    )  # np x 1

    log_prior_data = 1 * (
        (a0 - 1) * loggamma - b0 * torch.exp(loggamma) + loggamma
    )  # np x 1

    log_prior_w = (
        -0.5 * (W.shape[-1] - 2) * (np.log(2 * np.pi) - loglambda)
        - (torch.exp(loglambda) / 2) * torch.sum(W[:, 0:-2] ** 2, dim=-1, keepdim=True)
        + 1 * ((a0 - 1) * loglambda - b0 * torch.exp(loglambda) + loglambda)
    )  # np x 1
    if flag_gamma:
        log_posterior = torch.squeeze(
            log_lik_data / X.shape[0] * n_train + log_prior_data + log_prior_w
        )  # np
    else:
        log_posterior = torch.squeeze(
            log_lik_data / X.shape[0] * n_train + 0 * log_prior_w
        )  # np

    score = torch.autograd.grad(
        torch.sum(log_posterior), W, create_graph=flag_create, retain_graph=True
    )[
        0
    ]  # Score of the joint likelihood (including the prior)
    score_g = torch.autograd.grad(log_lik_data.mean(), W, create_graph=flag_create)[
        0
    ]  # Score of the likelihood, no prior involved (for debug purpose)

    if flag_create:
        return score
    else:

        return score.clone().detach(), score_g.clone().detach()
