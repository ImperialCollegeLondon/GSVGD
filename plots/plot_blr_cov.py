import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,1,4,3,7"
import sys
sys.path.append(".")
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from src.diffusion import Diffusion
import argparse


device = torch.device("cuda")
# device = "cuda:5"

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=100, help='Num of particles')
parser.add_argument('--dim', type=int, default=100, help='Num of particles')
parser.add_argument('--epochs', type=int, default=1000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.01, help='learning rate for g')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--format', type=str, default="png", help='format of figs')
args = parser.parse_args()
dim = args.dim
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if args.noise=="True" else ""

basedir = f"{args.root}/{args.exp}"
resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}"
resdir_svgd = f"rbf_epoch{args.epochs}_lr0.1_delta0.01_n{nparticles}"
resdir_ssvgd = f"rbf_epoch{args.epochs}_lr0.1_delta0.01_n{nparticles}"
resdir_hmc = resdir

seeds = range(1)

if __name__ == "__main__":

  df_list = []
  for seed in seeds:
    path = f"{basedir}/{resdir}/seed{seed}"
    path_svgd = f"{basedir}/{resdir_svgd}/seed{seed}"
    path_ssvgd = f"{basedir}/{resdir_svgd}/seed{seed}"
    path_hmc = f"{basedir}/{resdir_hmc}/seed{seed}"

    # load results
    svgd_res = pickle.load(open(f"{path_svgd}/particles_svgd.p", "rb"))
    ssvgd_res = pickle.load(open(f"{path_ssvgd}/particles_s-svgd_lrg{args.lr_g}.p", "rb"))
    hmc_res = pickle.load(open(f"{path_hmc}/particles_hmc.p", "rb"))
    particles_hmc = hmc_res["particles"].cpu()
    
    method_ls = [hmc_res, svgd_res, ssvgd_res]
    method_names = ["HMC", "SVGD", "S-SVGD"]

    eff_dims = [1, 10, 50] # [1, 2, 5, 10, 20, 30, 40, 50]
    for eff_dim in eff_dims:
      gsvgd_res = pickle.load(open(f"{path}/particles_gsvgd_effdim{eff_dim}.p", "rb"))
      method_ls.append(gsvgd_res)
      method_names.append(f"GSVGD{eff_dim}")
        
    # load target distribution
    target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
    data = torch.load(f'{path}/data.p', map_location=device)

    subplot_c = 3 # int(np.ceil(np.sqrt(len(method_ls))))
    subplot_r = int(np.ceil(len(method_ls) / subplot_c))


    fig = plt.figure(figsize=(subplot_c*6, subplot_r*5))
    ## plot solutions
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):
      print("Loading", method_name)

      if method_name == "HMC":
        particles = res["particles"].cpu()
      else:
        particles = res["particles"][-1][1].cpu()

      # cov matrix
      cov_matrix = np.cov(particles.T)

      print(i, method_name)
      plt.subplot(subplot_r, subplot_c, i+1)
      plt.imshow(cov_matrix, vmin=-5, vmax=8)
      plt.xticks(fontsize=35)
      plt.yticks(fontsize=35)
      plt.title(method_name, fontsize=35)
      if i == len(method_names) - 1:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=25)

    fig.tight_layout()
    fig_name = f"{basedir}/{resdir}/seed{seed}/cov"
    fig.savefig(fig_name + ".png")
    fig.savefig(fig_name + ".pdf")
    print("saved to", fig_name)

