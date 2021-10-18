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
from geomloss import SamplesLoss


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

seeds = range(10)

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
    cov_hmc = np.cov(particles_hmc.T) # cov matrix
    
    method_ls = [svgd_res, ssvgd_res]
    method_names = ["SVGD", "S-SVGD"]

    eff_dims = [1, 2, 5, 10, 20, 30, 40, 50]
    for eff_dim in eff_dims:
      gsvgd_res = pickle.load(open(f"{path}/particles_gsvgd_effdim{eff_dim}.p", "rb"))
      method_ls.append(gsvgd_res)
      method_names.append(f"GSVGD{eff_dim}")
        
    # load target distribution
    target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
    data = torch.load(f'{path}/data.p', map_location=device)
    _, _, acc_hmc, _ = target_dist.evaluation(particles_hmc, data["X_test"].cpu(), data["y_test"].cpu())
    print("HMC test accuracy:", acc_hmc)

    subplot_c = 3 # int(np.ceil(np.sqrt(len(method_ls))))
    subplot_r = int(np.ceil(len(method_ls) / subplot_c))


    ## plot solutions
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):
      print("Loading", method_name)

      particles = res["particles"][-1][1].cpu()
      _, _, acc, _ = target_dist.evaluation(particles, data["X_test"].cpu(), data["y_test"].cpu())
      print(method_name, "test accuracy:", acc)

      # get distance for var
      # energy
      energy = SamplesLoss("energy")
      energy_dist = energy(particles_hmc, particles).item()

      # cov matrix
      cov_matrix = np.cov(particles.T)
      l2_dist = np.sqrt(np.sum((cov_matrix - cov_hmc)**2))
      l2_diag_dist = np.sqrt(np.sum(np.diag(cov_matrix - cov_hmc)**2))
      # if method_name in ["SVGD", "S-SVGD"]:
      #   print(method_name)
      #   print(np.diag(cov_matrix))
      #   print(np.diag(cov_hmc))

      if not "GSVGD" in method_name:
        rep = len(eff_dims)
        eff_dim_plot = eff_dims
      else:
        rep = 1
        eff_dim_plot = [int(method_name.split("GSVGD")[-1])]
        method_name = "GSVGD"

      df_new = pd.DataFrame({
        "Energy Distance": [energy_dist] * rep,
        "Covariance Error": [l2_dist] * rep,
        "l2_diag_dist": [l2_diag_dist] * rep,
        "method": [method_name] * rep,
        "eff_dim": eff_dim_plot,
        "seed": [seed] * rep,
      })

      df_list.append(df_new)

  # gather data into dataframe
  df = pd.concat(df_list)
  df.sort_values(["method", "eff_dim"], inplace=True)
  df.reset_index(drop=True, inplace=True)

  for metric in ["Energy Distance", "Covariance Error", "l2_diag_dist"]:
    # 95% CI
    for method_name in ["SVGD", "S-SVGD", "GSVGD"]:
      for eff_dim in eff_dims:
        cond = (df.method == method_name) & (df.eff_dim == eff_dim)
        mean = np.mean(df.loc[cond, metric])
        std = np.std(df.loc[cond, metric])
        df.loc[cond, "lower"] = mean - 1.96*std/np.sqrt(len(seeds))
        df.loc[cond, "upper"] = mean + 1.96*std/np.sqrt(len(seeds))
        
    # plot observations
    fig = plt.figure(figsize=(12, 8))
    g = sns.lineplot(
      data=df,
      x="eff_dim",
      y=metric,
      hue="method",
      hue_order=["SVGD", "S-SVGD", "GSVGD"],
      ci=None
    )

    for method_name in ["SVGD", "S-SVGD", "GSVGD"]:
      plt.fill_between(data=df.loc[df.method==method_name], 
        x="eff_dim", y1="lower", y2="upper", alpha=0.2)

    plt.xlabel("Projection Dimension", fontsize=40)
    plt.xticks(fontsize=35)
    ylab_fontsize = 40 # if metric == "Covariance Error" else 40
    plt.ylabel(metric, fontsize=ylab_fontsize)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=30)
    fig.tight_layout()

    fig_name = f"{basedir}/{resdir}/{metric}"
    fig.savefig(fig_name + ".png")
    fig.savefig(fig_name + ".pdf")
    print("saved to", fig_name)

