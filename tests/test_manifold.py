import pytest
import torch
from src.manifold import Grassmann as Grassman_torch
from pymanopt.manifolds import Grassmann


def test_egrad2rgrad():

    d = 100
    m = 5

    manifold_torch = Grassman_torch(d, m)
    manifold = Grassmann(d, m)
    A = manifold.rand()
    ## keep the precision the same
    A_torch = torch.Tensor(A).double()

    res_torch = manifold_torch.egrad2rgrad(A_torch, A_torch)
    res = manifold.egrad2rgrad(A, A)

    assert res_torch.shape == res.shape
    assert torch.allclose(res_torch, torch.Tensor(res).double())


def test_retr():

    d = 100
    m = 5

    manifold_torch = Grassman_torch(d, m)
    manifold = Grassmann(d, m)
    A = manifold.rand()
    ## keep the precision the same
    A_torch = torch.Tensor(A).double()

    res_torch = manifold_torch.retr(A_torch, A_torch)
    res = manifold.retr(A, A)

    assert res_torch.shape == res.shape
    assert torch.allclose(res_torch, torch.Tensor(res).double())
