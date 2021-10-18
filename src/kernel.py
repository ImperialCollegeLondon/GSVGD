import torch
import autograd.numpy as np


def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    """
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2


def l2norm_batch(Xr, Yr):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    Return:
        dnorm2: n x n x b
    """
    # XrXr = torch.einsum("ibk,jbk->ijb", Xr, Xr)
    # XrYr = torch.einsum("ibk,jbk->ijb", Xr, Yr)
    # YrYr = torch.einsum("ibk,jbk->ijb", Yr, Yr)

    # dnorm2 = (
    #     -2 * XrYr
    #     + torch.einsum("iib->ib", XrXr).unsqueeze(1)
    #     + torch.einsum("iib->ib", YrYr).unsqueeze(0)
    # ) # n x m x b

    Xr = Xr.permute(1, 0, 2) # b x n x dim
    Yr = Yr.permute(1, 0, 2) # b x n x dim
    
    XrXr = torch.sum(Xr * Xr, dim=-1).unsqueeze(-1) # b x n x 1
    YrYr = torch.sum(Yr * Yr, dim=-1).unsqueeze(-2) # b x 1 x m
    XrYr = torch.matmul(Yr, torch.transpose(Xr, -1, -2))
    
    dnorm2 = XrXr + YrYr - 2 * XrYr # b x n x m
    dnorm2 = dnorm2.permute(1, 2, 0) # n x m x b

    return dnorm2


def median_heuristic(dnorm2, device):
    """Compute median heuristic.
    Inputs:
        dnorm2: (n x n) tensor of \|X - Y\|_2^2
    Return:
        med(\|X_i - Y_j\|_2^2, 1 \leq i < j \leq n)
    """
    ind_array = torch.triu(torch.ones_like(dnorm2, device=device), diagonal=1) == 1
    # ind_array = torch.triu(torch.ones_like(dnorm2, device=device)) == 1
    med_heuristic = torch.median(dnorm2[ind_array])
    return med_heuristic


def median_heuristic_batch(dnorm2, device):
    """Compute median heuristic.
    Inputs:
        dnorm2: (n x n x b) tensor of \|X - Y\|_2^2
    Return:
        (b, ) tensor of med( \|X_i - Y_j\|_2, 1 \leq i < j \leq n )
    """
    dnorm2 = dnorm2.permute(2, 0, 1) # batch x n x n
    ind_array = torch.triu(torch.ones_like(dnorm2, device=device), diagonal=1) == 1
    selected_dnorm2 = dnorm2[ind_array].view((dnorm2.shape[0], -1)) # batch x (n * (n - 1) / 2)

    # ind_array = torch.triu(torch.ones_like(dnorm2, device=device)) == 1
    # selected_dnorm2 = dnorm2[ind_array].view((dnorm2.shape[0], -1)) # batch x (n * (n - 1) / 2)
    # selected_dnorm2 = (selected_dnorm2 - torch.tril(selected_dnorm2)).view((dnorm2.shape[0], -1))

    med_heuristic = torch.median(selected_dnorm2, dim=1)[0]
    return med_heuristic


class RBF(torch.nn.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma=None, method="med_heuristic"):
        super().__init__()
        self.sigma = sigma
        self.method = method

    def bandwidth(self, X, Y):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2, device=X.device)
        sigma2 = med_heuristic_sq / np.log(X.shape[0])
        self.sigma = sigma2.detach().sqrt()

    def forward(self, X, Y):
        
        # dnorm2 = l2norm(X, Y)

        # # median heuristic
        # with torch.no_grad():
        #     if self.method == "med_heuristic" or self.sigma is None:
        #             # sigma2 = torch.median(dnorm2) / (2 * np.log(X.size(0)))
        #             # sigma2 = torch.median(dnorm2)
        #             med_heuristic_sq = median_heuristic(dnorm2, device=X.device)
        #             sigma2 = med_heuristic_sq / np.log(X.size(0))
        #             self.sigma = sigma2.sqrt().item()

        #     # 1/sigma^2
        #     sigma2_inv = 1.0 / (1e-9 + self.sigma ** 2)

        # K_XY = (-sigma2_inv * dnorm2).exp()

        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma**2 + 1e-9)
        sigma2_inv = sigma2_inv.unsqueeze(0).unsqueeze(0)
        K_XY = (- sigma2_inv * dnorm2).exp()

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form 
        """
        # (symmetric kernel)
        # Gram matrix
        sigma2_inv = 1 / (1e-9 + self.sigma ** 2)
        # TODO: Is this correct?
        a = (-l2norm(X, Y) * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Y.unsqueeze(1) - X).transpose(0, -1)
        # compute grad_K
        grad_K_XY = - 2 * sigma2_inv * diff * a

        return grad_K_XY

    def gradgrad(self, X, Y):
        """
        Args:
            X: torch.Tensor of shape (n, dim)
            Y: torch.Tensor of shape (m, dim)
        Output:
            torch.Tensor of shape (dim, dim, n, m)
        """
        # Gram matrix
        sigma2_inv = 1 / (1e-9 + self.sigma ** 2)
        a = (-l2norm(X, Y) * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Y.unsqueeze(1) - X).transpose(0, -1)
        # product of differences
        diff_outerprod = -diff.unsqueeze(0) * diff.unsqueeze(1)
        # compute gradgrad_K
        diag = 2 * sigma2_inv * torch.eye(X.shape[1], device=X.device).unsqueeze(
            -1
        ).unsqueeze(-1)
        gradgrad_K_all = (diag + 4 * sigma2_inv ** 2 * diff_outerprod) * a
        return gradgrad_K_all


class IMQ(torch.nn.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma=None, method="med_heuristic"):
        super().__init__()
        self.sigma = sigma
        self.method = method

    def forward(self, X, Y):
        dnorm2 = l2norm(X, Y)

        # median heuristic
        if self.method == "med_heuristic" or self.sigma is None:
            sigma2 = torch.median(dnorm2) / (X.size(0) ** 2 - 1)
            # sigma2 = torch.median(dnorm2)
            self.sigma = sigma2.sqrt().item()

        sigma2_inv = 1.0 / (1e-9 + self.sigma ** 2)
        K_XY = 1 / (sigma2_inv * dnorm2 + 1).sqrt()

        return K_XY

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form 
        """
        # (symmetric kernel)
        # Gram matrix
        sigma2_inv = 1 / (1e-9 + self.sigma ** 2)
        # TODO: Is this correct?
        a = 1 / (l2norm(X, Y) * sigma2_inv + 1).sqrt().pow(3).unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Y.unsqueeze(1) - X).transpose(0, -1)
        # compute grad_K
        grad_K_XY = -sigma2_inv * diff * a
        return grad_K_XY

    def gradgrad(self, X, Y):
        """
        Args:
            X: torch.Tensor of shape (n, dim)
            Y: torch.Tensor of shape (m, dim)
        Output:
            torch.Tensor of shape (dim, dim, n, m)
        """
        # Gram matrix
        sigma2_inv = 1 / (1e-9 + self.sigma ** 2)
        a = (l2norm(X, Y) * sigma2_inv + 1).sqrt().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (X - Y.unsqueeze(1)).transpose(0, -1)
        # product of differences
        diff_outerprod = -diff.unsqueeze(0) * diff.unsqueeze(1)
        # compute gradgrad_K
        diag = sigma2_inv * torch.eye(X.shape[1], device=X.device).unsqueeze(
            -1
        ).unsqueeze(-1)
        a_inv = 1 / a
        gradgrad_K_all = (
            diag + 3 * sigma2_inv ** 2 * diff_outerprod * a_inv.pow(2)
        ) * a_inv.pow(3)
        return gradgrad_K_all


class SumKernel(torch.nn.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(
        self, sigma_rbf=None, sigma_imq=None, method="med_heuristic", weight=[0.5, 0.5]
    ):
        super().__init__()
        self.rbf = RBF(sigma_rbf, method=method)
        self.imq = IMQ(sigma_imq, method=method)

    def forward(self, X, Y):
        return 0.5 * self.rbf(X, Y) + 0.5 * self.imq(X, Y)

    def grad_first(self, X, Y):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return 0.5 * self.rbf.grad_first(X, Y) + 0.5 * self.imq.grad_first(X, Y)

    def grad_second(self, X, Y):
        """Compute grad_K in wrt second argument in matrix form 
        """
        return 0.5 * self.rbf.grad_second(X, Y) + 0.5 * self.imq.grad_second(X, Y)

    def gradgrad(self, X, Y):
        """
        Args:
            X: torch.Tensor of shape (n, dim)
            Y: torch.Tensor of shape (m, dim)
        Output:
            torch.Tensor of shape (dim, dim, n, m)
        """
        return 0.5 * self.rbf.gradgrad(X, Y) + 0.5 * self.imq.gradgrad(X, Y)


class BatchRBF(torch.nn.Module):
    """For GSVGD to work, a kernel class need to have the following methods:
        forward: kernel evaluation k(x, y)
        grad_first: grad_x k(x, y)
        grad_second: grad_y k(x, y)
        gradgrad: grad_x grad_y k(x, y)
    """

    def __init__(self, sigma=None, method="med_heuristic", device="cpu"):
        super().__init__()
        self.sigma = sigma
        self.method = method
        self.device = device

    def bandwidth(self, Xr, Yr):
        """Compute magic bandwidth
        """
        dnorm2 = l2norm_batch(Xr, Yr)
        med_heuristic_sq = median_heuristic_batch(dnorm2, device=Xr.device)
        sigma2 = med_heuristic_sq / np.log(Xr.shape[0])
        self.sigma = sigma2.detach().sqrt()

    def forward(self, Xr, Yr):
        """
        Args:
            Xr: torch.Tensor of shape (n, batch, dim)
            Yr: torch.Tensor of shape (m, batch, dim)
        Output:
            torch.Tensor of shape (n, m, batch)
        """

        # # median heuristic
        # dnorm2 = l2norm_batch(Xr, Yr)
        # with torch.no_grad():

        #     # if self.method == "med_heuristic" or self.sigma is None:
        #     #     # med_heuristic_sq = median_heuristic_batch(dnorm2, device=Xr.device)
        #     #     # sigma2 = med_heuristic_sq / np.log(Xr.shape[0])
        #     #     # self.sigma = sigma2.detach().requires_grad_(False).sqrt()

        #     #     self.bandwidth(Xr, Yr)

        #     # 1/sigma^2
        #     sigma2_inv = 1.0 / (1e-9 + self.sigma**2)
        #     sigma2_inv = sigma2_inv.unsqueeze(0).unsqueeze(0)
        
        # K_XrYr = (-sigma2_inv * dnorm2).exp()


        #?
        # Xr_ = Xr.permute(1, 0, 2).unsqueeze(-2) # b x n x 1 x dim
        # Yr_ = Yr.permute(1, 0, 2).unsqueeze(-3) # b x 1 x m x dim
        # norm2 = torch.sum((Xr_ - Yr_)**2, dim=-1) # b x n x m
        # print(self.sigma.shape, norm2.shape)
        # K_XrYr = (-norm2 / (self.sigma.unsqueeze(-1).unsqueeze(-1)**2 + 1e-9)).exp() # b x n x m
        # K_XrYr = K_XrYr.permute(1, 2, 0)

        dnorm2 = l2norm_batch(Xr, Yr)
        sigma2_inv = 1.0 / (self.sigma**2 + 1e-9)
        sigma2_inv = sigma2_inv.unsqueeze(0).unsqueeze(0)
        K_XrYr = (- sigma2_inv * dnorm2).exp()

        #? 
        # K_XrYr = (- dnorm2)
        # print("dist", dnorm2.detach().cpu().numpy()[:5, :5, 6])
        # print("med", med_heuristic_sq.cpu().numpy())
        # print("sigma", self.sigma.cpu().numpy())

        return K_XrYr

    def grad_first(self, Xr, Yr):
        """Compute grad_K in wrt first argument in matrix form 
        """
        return -self.grad_second(Xr, Yr)

    def grad_second(self, Xr, Yr):
        """Compute grad_K in wrt second argument in matrix form.
        Args:
            Xr: torch.Tensor of shape (n, batch, dim)
            Yr: torch.Tensor of shape (m, batch, dim)
        Output:
            torch.Tensor of shape (dim, n, m, batch)
        """
        sigma2_inv = 1.0 / (1e-9 + self.sigma ** 2)
        sigma2_inv = sigma2_inv.unsqueeze(0).unsqueeze(0)
        K = (-l2norm_batch(Xr, Yr) * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Yr.unsqueeze(1) - Xr).permute((3, 1, 0, 2))
        # compute grad_K
        grad_K_XY = - 2 * sigma2_inv.unsqueeze(0) * diff * K

        return grad_K_XY

    def gradgrad(self, Xr, Yr):
        """
        Args:
            Xr: torch.Tensor of shape (n, batch, dim)
            Yr: torch.Tensor of shape (m, batch, dim)
        Output:
            torch.Tensor of shape (dim, dim, n, m, batch)
        """
        sigma2_inv = 1.0 / (1e-9 + self.sigma ** 2)
        sigma2_inv = sigma2_inv.unsqueeze(0).unsqueeze(0)

        K = (-l2norm_batch(Xr, Yr) * sigma2_inv).exp().unsqueeze(0)
        # [diff]_{ijk} = y^i_j - x^i_k
        diff = (Yr.unsqueeze(1) - Xr).permute((3, 1, 0, 2))
        # product of differences
        diff_outerprod = -diff.unsqueeze(0) * diff.unsqueeze(1)
        # compute gradgrad_K
        #! missing factor of 2!
        M = Xr.shape[2]
        sigma2_inv = sigma2_inv.unsqueeze(0)
        diag = 2 * sigma2_inv * torch.eye(M, device=Xr.device).unsqueeze(-1).unsqueeze(
            -1
        ).unsqueeze(-1)
        gradgrad_K_all = (diag + 4 * sigma2_inv ** 2 * diff_outerprod) * K.unsqueeze(0)
        return gradgrad_K_all
