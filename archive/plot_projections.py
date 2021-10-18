import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from src.utils import plot_particles_all
from src.metrics import Metric
import argparse
import os
import torch.autograd as autograd

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--dim', type=int, default=50, help='Dimension')
parser.add_argument('--epochs', type=int, default=10000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--xlim', type=float, default=8, help='learning rate')
parser.add_argument('--format', type=str, default="png", help='format of figs')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--method', type=str, default="GSVGD1", help='which method')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if bool(args.noise) else ""
method = args.method

basedir = f"{args.root}/{args.exp}"
resdirs = [
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim{args.dim}",
]
resdirs = [f"{s}/seed0" for s in resdirs]


if __name__ == "__main__":


  path = resdirs[0]
  path = f"{basedir}/{path}"
  print(f"Plotting for {path}")
  
  savedir = f"{path}/projections_{method}" 
  if not os.path.exists(savedir):
      os.makedirs(savedir)
  
  # load results
  res = pickle.load(open(f"{path}/particles.p", "rb"))
  #target_dist = res["target_dist"]
  eff_dims = res["effdims"]
  target, svgd, maxsvgd = res["target"], res["svgd"], res["maxsvgd"]
  gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
  # save last particles in list
  dim = svgd[-1].shape[1]

  target_dist = torch.load(f'{path}/target_dist.p', map_location=device)

  gsvgd_titles = ["GSVGD" + str(i) for i in eff_dims]
  gsvgd_keys = [f"gsvgd_effdim{i}" for i in eff_dims]
  # final_particles_dic = [
  #   ("SVGD", svgd[-1].to(device)),
  #   ("S-SVGD", maxsvgd[-1].to(device))
  # ] + [(n, gsvgd[s][-1].to(device)) for n, s in zip(gsvgd_titles, gsvgd_keys)]

  # mean vector for padding
  mix_means = target_dist.sample((10000,)).mean(axis=0)


  method_names = [f"GSVGD{d}" for d in eff_dims]

  target_samples = target_dist.sample((10000,)).cpu().numpy()
  
  final_particles_dict = {
    "Target": target_samples,
    "SVGD": svgd[-1].cpu().numpy(),
    "S-SVGD": [x.cpu().numpy() for x in maxsvgd],
    **{n: [x.cpu().numpy() for x in gsvgd[s]] for n, s in zip(gsvgd_titles, gsvgd_keys)}
  }

  # get projections
  projections = {
    **{n: [u.cpu().numpy() for u in res[f"proj_gsvgd_effdim{i}"]] for n, i in zip(gsvgd_titles, eff_dims)}
  }
  phi_list = {
    **{n: [u.cpu().numpy() for u in res[f"phi_gsvgd_effdim{i}"]] for n, i in zip(gsvgd_titles, eff_dims)}
  }
  repulsion_list = {
    **{n: [u.cpu().numpy() for u in res[f"repulsion_gsvgd_effdim{i}"]] for n, i in zip(gsvgd_titles, eff_dims)}
  }
  score_list = {
    **{n: [u.cpu().numpy() for u in res[f"score_gsvgd_effdim{i}"]] for n, i in zip(gsvgd_titles, eff_dims)}
  }
  k_list = {
    **{n: [u.cpu().numpy() for u in res[f"k_gsvgd_effdim{i}"]] for n, i in zip(gsvgd_titles, eff_dims)}
  }

  # plot density
  subplot_c = int(np.ceil(np.sqrt(len(final_particles_dict))))
  subplot_r = int(np.ceil(len(final_particles_dict) / subplot_c))

  dim1, dim2 = 0, 1

  df_true = pd.DataFrame(final_particles_dict["Target"][:, :2], columns=["dim1", "dim2"])


  ## GSVGD
  # fig = plt.figure(figsize=(subplot_c*3, subplot_r*3))
  if method != "S-SVGD":
    cut = 22 if method == "SVGD" else 22
    fig = plt.figure(figsize=(8, 8))
    g = sns.kdeplot(
      data=df_true, x="dim1", y="dim2", fill=True,
      thresh=0, levels=60, cmap="viridis", cut=cut
    )
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(ylim=(-args.xlim, args.xlim), xlim=(-args.xlim, args.xlim))
    g.set(xticks=[])
    g.set(yticks=[])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(method, fontsize=22)
  
    fig.tight_layout()
    fig.savefig(savedir + f"/0_target_density.{args.format}", dpi=500)
    

    for j, proj in enumerate(projections[method]):
      if proj.ndim == 3:
        # take only matrix A for the general kernel
        proj = proj[:, :, 0]

      if (j+1) % 10 == 0:
        x = final_particles_dict[method][j]
        df = pd.DataFrame(x[:, :2], columns=["dim1", "dim2"])

        fig = plt.figure(figsize=(8, 8))
        # g = sns.kdeplot(
        #   data=df_true, x="dim1", y="dim2", fill=True,
        #   thresh=0, levels=60, cmap="viridis", cut=cut
        # )
        
        #? project 
        for i in range(proj.shape[1]):
          #! only works with effdim=1
          # proj_vec = proj[:, i].reshape((-1, 1))
          eff_dim = int(method[-1])
          proj_vec = proj[:, (i*eff_dim) : ((i+1)*eff_dim)]
          proj_mat = proj_vec @ proj_vec.T
          df_proj = pd.DataFrame((x @ proj_mat)[:, :2], columns=["dim1", "dim2"])
          
          plt.scatter(df.dim1, df.dim2, color="red")
          plt.scatter(df_proj.dim1, df_proj.dim2, color="blue", alpha=0.3)
          plt.quiver(0, 0, proj_vec[0], proj_vec[1], scale=5, width=0.002)
  
        ## phi
        phi = phi_list[method][j-1]
        # phi = phi_list[method][j-1][i, :, :]
        phi = np.einsum("bij -> ij", phi)
        pos = df.to_numpy()
        plt.quiver(pos[:, :1], pos[:, 1:2], phi[:, :1], phi[:, 1:2], scale=3, width=0.002, alpha=0.5)

        ## attraction and repulsion
        repulsion = repulsion_list[method][j-1]
        repulsion = np.einsum("bij -> ij", repulsion)
        attraction = phi - repulsion
        plt.quiver(pos[:, :1], pos[:, 1:2], attraction[:, :1], attraction[:, 1:2], scale=3, width=0.002, alpha=0.5, color="green")
        plt.quiver(pos[:, :1], pos[:, 1:2], repulsion[:, :1], repulsion[:, 1:2], scale=3, width=0.002, alpha=0.5, color="red")

        ## score
        # B = np.eye(dim)
        # x_cp = torch.Tensor(x).to("cuda:1").requires_grad_()
        # lp = target_dist.log_prob(x_cp).sum()
        # score = autograd.grad(lp, x_cp)[0].cpu().numpy()
        # B_score = score @ B @ B.T
        B_score_r = score_list[method][j-1] # batch x num_particles x dim
        B_score = np.einsum("bij -> ij", B_score_r) / x.shape[0]
        plt.quiver(pos[:, :1], pos[:, 1:2], B_score[:, :1], B_score[:, 1:2], scale=1, width=0.002, alpha=0.5, color="yellow")

        K_AxAx = k_list[method][j-1] # num_particles x num_particles x batch
        B_score_k_r = np.einsum("bij, ikb -> bkj", B_score_r, K_AxAx)
        B_score_k = np.einsum("bij -> ij", B_score_k_r) / x.shape[0]
        # print(B_score_k_r[0, :3, :3])
        # print(B_score_k_r[1, :3, :3])
        # plt.quiver(pos[:, :1], pos[:, 1:2], B_score_k[:, :1], B_score_k[:, 1:2], scale=1, width=0.002, alpha=0.5, color="yellow")


        
        g.set(xlabel=None)
        g.set(ylabel=None)
        plt.ylim(-args.xlim, args.xlim)
        plt.xlim(-args.xlim, args.xlim)
        g.set(xticks=[])
        g.set(yticks=[])
        
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(method, fontsize=22)
      
        fig.tight_layout()
        fig.savefig(savedir + f"/projections_{method}_{j}.{args.format}", dpi=500)
        print(f"Saved to ", savedir + f"/projections/{j}.png")


  ## S-SVGD
  elif method == "S-SVGD":
    projections = [x.cpu().numpy() for x in res["proj_maxsvgd"]]
    phi_list = [x.cpu().numpy() for x in res["phi_maxsvgd"]]
    repulsion_list = [x.cpu().numpy() for x in res["repulsion_maxsvgd"]]

    cut = 22 if method == "SVGD" else 22
    fig = plt.figure(figsize=(8, 8))
    g = sns.kdeplot(
      data=df_true, x="dim1", y="dim2", fill=True,
      thresh=0, levels=60, cmap="viridis", cut=cut
    )
    g.set(xlabel=None)
    g.set(ylabel=None)
    g.set(ylim=(-args.xlim, args.xlim), xlim=(-args.xlim, args.xlim))
    g.set(xticks=[])
    g.set(yticks=[])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(method, fontsize=22)
  
    fig.tight_layout()
    fig.savefig(savedir + f"/0_target_density.{args.format}", dpi=500)
    
    for j, proj in enumerate(projections):
      # each row of proj is a slice
      if (j+1) % 10 == 0:
        x = final_particles_dict[method][j]
        df = pd.DataFrame(x[:, :2], columns=["dim1", "dim2"])
        
        fig = plt.figure(figsize=(8, 8))

        plt.scatter(df.dim1, df.dim2, color="red")
      
        #? project 
        for i in range(proj.shape[0]):
          slice_vec = proj[i, :].reshape((1, -1))
          proj_mat = slice_vec.T @ slice_vec
          df_proj = pd.DataFrame((x @ proj_mat)[:, :2], columns=["dim1", "dim2"])
          
          plt.scatter(df_proj.dim1, df_proj.dim2, color="blue", alpha=0.3)
          plt.quiver(0, 0, slice_vec[:, 0], slice_vec[:, 1], scale=5, width=0.002)

        ## phi
        phi = phi_list[j-1]
        pos = df.to_numpy()
        plt.quiver(pos[:, :1], pos[:, 1:2], phi[:, :1], phi[:, 1:2], scale=3, width=0.002, alpha=0.5)

        ## attraction and repulsion
        repulsion = repulsion_list[j-1]
        attraction = phi - repulsion
        plt.quiver(pos[:, :1], pos[:, 1:2], attraction[:, :1], attraction[:, 1:2], scale=3, width=0.002, alpha=0.5, color="green")
        plt.quiver(pos[:, :1], pos[:, 1:2], repulsion[:, :1], repulsion[:, 1:2], scale=3, width=0.002, alpha=0.5, color="red")
      
        g.set(xlabel=None)
        g.set(ylabel=None)
        plt.ylim(-args.xlim, args.xlim)
        plt.xlim(-args.xlim, args.xlim)
        g.set(xticks=[])
        g.set(yticks=[])
      
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(method, fontsize=22)
      
        fig.tight_layout()
        fig.savefig(savedir + f"/projections_{method}_{j}.{args.format}", dpi=500)
        print(f"Saved to ", savedir + f"/projections/{j}.png")
  
  

