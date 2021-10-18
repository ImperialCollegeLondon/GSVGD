import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class GaussianMixture(torch.distributions.Distribution):
    def __init__(
        self, mean: np.array, covariance_matrix: np.array, prob: np.array
    ) -> None:
        self.num_components = mean.size(0)
        self.mu = mean
        self.covariance_matrix = covariance_matrix
        self.prob = prob

        self.dists = [
            torch.distributions.MultivariateNormal(mu, covariance_matrix=sigma)
            for mu, sigma in zip(mean, covariance_matrix)
        ]

        super().__init__(
            torch.Size([]), torch.Size([mean.size(-1)])
        )

    def log_prob(self, value: np.ndarray):
        return torch.cat(
            [
                p * d.log_prob(value).unsqueeze(-1)
                for p, d in zip(self.prob, self.dists)
            ],
            dim=-1,
        ).logsumexp(dim=-1)

    def enumerate_support(self):
        return self.dists[0].enumerate_support()


def plot_particles(x_init, x_final, P, d=7.0, step=0.1, concat=None, savedir=None, figsize=(12, 6)):
    '''
    Args:
        pad: If in dim > 2, then concat the 2d grid with specified tensor (of 
            shape (1, dim - 2)) when calculating the log likelihood.
    '''
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    xv, yv = xv.to(x_init.device), yv.to(x_init.device)
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1).to(x_init.device)
    if P.event_shape[0] > 2:
        # assert concat is not None, "Need to specify concat in dim larger than 2."
        if concat is None:
            print(("Concatenating higher dims with vector of zeros. Customize by "
                "using the concat argument."))
            concat = torch.zeros((1, P.event_shape[0] - 2)).to(x_init.device)

        pos_xy = torch.cat(
            (pos_xy, concat.repeat(pos_xy.shape[0], pos_xy.shape[1], 1)), 
            dim=2).to(x_init.device)
    p_xy = P.log_prob(pos_xy).exp().unsqueeze(-1)
    xv, yv, x_init, x_final, p_xy = xv.to("cpu"), yv.to("cpu"), x_init.to("cpu"), x_final.to("cpu"), p_xy.to("cpu")
    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.contourf(xv, yv, p_xy.squeeze(-1), levels=20)
    plt.scatter(x_init[:, 0], x_init[:, 1], color="r", alpha=0.5, s=6)
    plt.title("Initial")
    plt.subplot(1, 2, 2)
    plt.contourf(xv, yv, p_xy.squeeze(-1), levels=20)
    plt.scatter(x_final[:, 0], x_final[:, 1], color="r", alpha=0.5, s=6)
    plt.title("Final")
    if savedir:
        fig.savefig(savedir)
    return fig


def plot_particles_all(x_init, x_final_dic, P, d=7.0, step=0.1, concat=None, savedir=None, figsize=(12, 6)):
    '''
    Args:
        pad: If in dim > 2, then concat the 2d grid with specified tensor (of 
            shape (1, dim - 2)) when calculating the log likelihood.
    '''
    device = x_init.device
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    xv, yv = xv.to(device), yv.to(device)
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1).to(device)
    if P.event_shape[0] > 2:
        # assert concat is not None, "Need to specify concat in dim larger than 2."
        if concat is None:
            print(("Concatenating higher dims with vector of zeros. Customize by "
                "using the concat argument."))
            concat = torch.zeros((1, P.event_shape[0] - 2), device=device)

        pos_xy = torch.cat(
            (pos_xy, concat.repeat(pos_xy.shape[0], pos_xy.shape[1], 1)), 
            dim=2
        ).to(device)
    
    p_xy = P.log_prob(pos_xy).exp().unsqueeze(-1)
    fig = plt.figure(figsize=figsize)
    x_final_dic = [("Initial", x_init)] + x_final_dic
    subplot_c = int(np.ceil(len(x_final_dic) / 2))
    for i, tup in enumerate(x_final_dic):
        method, x_final = tup
        xv, yv, x_final, p_xy = xv.cpu(), yv.cpu(), x_final.cpu(), p_xy.cpu()
        plt.subplot(2, subplot_c, i+1)
        plt.contourf(xv.cpu(), yv.cpu(), p_xy.squeeze(-1).cpu(), levels=20)
        plt.scatter(x_final[:, 0], x_final[:, 1], color="r", alpha=0.35, s=6)
        plt.title(method, fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir)
    return fig


def plot_particles_hist(x_final_ls, epoch_ls, method, P, d=7.0, step=0.1, concat=None, savedir=None, figsize=None):
    '''
    Args:
        pad: If in dim > 2, then concat the 2d grid with specified tensor (of 
            shape (1, dim - 2)) when calculating the log likelihood.
    '''
    device = x_final_ls[0].device
    xv, yv = torch.meshgrid([torch.arange(-d, d, step), torch.arange(-d, d, step)])
    xv, yv = xv.to(device), yv.to(device)
    pos_xy = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), dim=-1).to(device)
    if P.event_shape[0] > 2:
        # assert concat is not None, "Need to specify concat in dim larger than 2."
        if concat is None:
            print(("Concatenating higher dims with vector of zeros. Customize by "
                "using the concat argument."))
            concat = torch.zeros((1, P.event_shape[0] - 2), device=device)

        pos_xy = torch.cat(
            (pos_xy, concat.repeat(pos_xy.shape[0], pos_xy.shape[1], 1)), 
            dim=2
        ).to(device)
    
    p_xy = P.log_prob(pos_xy).exp().unsqueeze(-1)
    subplot_c = int(np.ceil(np.sqrt(len(x_final_ls))))
    subplot_r = int(np.ceil(len(x_final_ls) / subplot_c))

    dim1, dim2 = 0, 1
    fig = plt.figure(figsize=(subplot_c*3, subplot_r*3))
    for i, x_final in enumerate(x_final_ls):
        epoch = epoch_ls[i]
        xv, yv, x_final, p_xy = xv.cpu(), yv.cpu(), x_final.cpu(), p_xy.cpu()
        plt.subplot(subplot_r, subplot_c, i+1)
        plt.contourf(xv.cpu(), yv.cpu(), p_xy.squeeze(-1).cpu(), levels=20)
        plt.scatter(x_final[:, dim1], x_final[:, dim2], color="r", alpha=0.35, s=2)
        plt.title(f"Epoch {epoch}", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    plt.suptitle(method, fontsize=15)
    fig.tight_layout()

    if savedir:
        fig.savefig(savedir)
    return fig


def colorFader(c1="blue", c2="orange", mix=0):
    '''fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    '''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def plot_metrics(epochs, metric_svgd, metric_gsvgd, eff_dims, metric_maxsvgd, name, savefile, 
    figsize=(8, 6), alpha=1, xlab="iterations", ylog=False, markersize=6):
    fig = plt.figure(figsize=figsize)
    plt.plot(epochs, metric_svgd, color="r", label="SVGD", alpha=alpha, marker="o", markersize=markersize)
    plt.plot(epochs, metric_maxsvgd, color="g", label="S-SVGD", linestyle="dashed", alpha=alpha, marker="s", markersize=markersize)
    gsvgd_markers = [">", "^", "*", "D"]
    for i in range(metric_gsvgd.shape[1]):
        plt.plot(
            epochs, 
            metric_gsvgd[:, i], 
            color=colorFader(mix=(i+1)/metric_gsvgd.shape[1]),
            label=f"GSVGD{eff_dims[i]}",
            linestyle="dashdot",
            marker=gsvgd_markers[i],
            alpha=alpha,
            markersize=markersize
        )
    plt.xlabel(xlab, fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel(name, fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16)
    if ylog:
        plt.yscale('log')
    fig.tight_layout()
    fig.savefig(f"{savefile}.png")

def plot_metrics_individual(epochs, metrics_ls, name, savefile, figsize=(8, 6), alpha=1, xlab="iterations", ylog=False):
    '''Plot metrics given a set of tuples of the form (method_name, metrics_vals)
    '''
    fig = plt.figure(figsize=figsize)
    plt_markers = ["o", "s", ">", "^", "*", "D"]
    plt_colours = ["r", "g"] + [colorFader(mix=(i+1)/3) for i in range(3)]
    plt_ltype = ["-", "dashed"] + ["dashdot"] * 3

    for i, tup in enumerate(metrics_ls):
        method, metrics = tup
        plt.plot(
            epochs, 
            metrics, 
            color=plt_colours[i],
            label=method,
            linestyle=plt_ltype[i],
            marker=plt_markers[i],
            alpha=alpha,
            markersize=6
        )
    plt.xlabel(xlab, fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel(name, fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16)
    if ylog:
        plt.yscale('log')
    fig.tight_layout()
    fig.savefig(f"{savefile}")


def AdaGrad_update(samples, force, lr, state_dict):
    M = state_dict['M']
    t = state_dict['t']
    beta1 = state_dict['beta1']
    if t == 1:
        M = M + force * force
    else:
        M = beta1 * M + (1 - beta1) * (force * force)
    adj_grad = force / (torch.sqrt(M) + 1e-10)
    samples.data = samples.data + lr * adj_grad
    t += 1
    state_dict['M'] = M
    state_dict['t'] = t
    return samples, state_dict
