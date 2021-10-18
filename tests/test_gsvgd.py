## TODO: add them in!

import pytest
import torch
from src.kernel import BatchRBF, RBF

def test_bacthkernel():

    nparticles = 5
    d = 6
    m = d
    M = d//m
    torch.manual_seed(0)

    X = torch.randn(nparticles, d).detach().requires_grad_(True)
    Y = X.clone().requires_grad_()

    Xr = X.reshape((nparticles, m, M))
    Yr = Xr.clone().requires_grad_()

    rbf = RBF()
    batchrbf = BatchRBF()

    for b in range(m):
        XX = Xr[:, b, :]
        YY = Yr[:, b, :]
        rbf_k = rbf(XX, YY)
        batchrbf_k = batchrbf(Xr, Yr)
        assert rbf.sigma == batchrbf.sigma[b], f"Sigma is wrong for batch {b}"
        assert torch.isclose(rbf_k, batchrbf_k[:, :, b]).all() 

        grad_first_rbf = rbf.grad_first(XX, YY)
        grad_first_batchrbf = batchrbf.grad_first(Xr, Yr)
        assert torch.isclose(grad_first_rbf, grad_first_batchrbf[:, :, :, b]).all()

        gradgrad_rbf = rbf.gradgrad(XX, YY)
        gradgrad_batchrbf = batchrbf.gradgrad(Xr, Yr)
        assert torch.isclose(gradgrad_rbf, gradgrad_batchrbf[:, :, :, :, b]).all() 
