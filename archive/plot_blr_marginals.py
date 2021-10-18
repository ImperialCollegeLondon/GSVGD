import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,1,2,3,7"
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
# var_ind_ls = range(10)
np.random.seed(1)
var_ind_ls = np.random.choice(range(55), size=5, replace=False) # 55 features for covertype

if __name__ == "__main__":

  for seed in seeds:
    for var_ind in var_ind_ls:
      path = f"{basedir}/{resdir}/seed{seed}"
      path_svgd = f"{basedir}/{resdir_svgd}/seed{seed}"
      path_ssvgd = f"{basedir}/{resdir_svgd}/seed{seed}"
      path_hmc = f"{basedir}/{resdir_hmc}/seed{seed}"

      # load results
      svgd_res = pickle.load(open(f"{path_svgd}/particles_svgd.p", "rb"))
      ssvgd_res = pickle.load(open(f"{path_ssvgd}/particles_s-svgd_lrg{args.lr_g}.p", "rb"))
      hmc_res = pickle.load(open(f"{path_hmc}/particles_hmc.p", "rb"))
      particles_hmc = hmc_res["particles"].cpu()
      
      method_ls = [svgd_res, ssvgd_res]
      method_names = ["SVGD", "S-SVGD"]

      eff_dims = [1, 2, 5, 10, 20,] # 30, 40, 50]
      for eff_dim in eff_dims:
        gsvgd_res = pickle.load(open(f"{path}/particles_gsvgd_effdim{eff_dim}.p", "rb"))
        method_ls.append(gsvgd_res)
        method_names.append(f"GSVGD{eff_dim}")
          
      # load target distribution
      target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
      data = torch.load(f'{path}/data.p', map_location=device)
      _, _, acc_hmc, _ = target_dist.evaluation(particles_hmc, data["X_test"].cpu(), data["y_test"].cpu())
      print("HMC test accuracy:", acc_hmc)

      plot_c = 3 # int(np.ceil(np.sqrt(len(method_ls))))
      plot_r = int(np.ceil(len(method_ls) / plot_c))

      fig = plt.figure(figsize=(plot_c*6, plot_r*5))

      ## plot solutions
      for i, (res, method_name) in enumerate(zip(method_ls, method_names)):
        print("Loading", method_name)
        df_list = []

        particles = res["particles"][-1][1].cpu()
        _, _, acc, _ = target_dist.evaluation(particles, data["X_test"].cpu(), data["y_test"].cpu())
        print(method_name, "test accuracy:", acc)

        # get marginal for var
        particles = particles[:, var_ind]

        nparticles = particles.shape[0]
        df_new = pd.DataFrame({
          "particles": particles,
          "method": method_name,
          "seed": seed,
        })

        # append true samples
        df_hmc = pd.DataFrame({
          "particles": particles_hmc[:, var_ind],
          "method": "HMC",
          "seed": seed
        })
        df_new = df_new.append(df_hmc)

        df_list.append(df_new)

        # gather data into dataframe
        df = pd.concat(df_list)

        # plot observations
        plt.subplot(plot_r, plot_c, i+1)
        sns.kdeplot(
          data=df,
          x="particles",
          hue="method"
        )
        # sns.histplot(
        #   data=df,
        #   x="particles",
        #   hue="method",
        #   kde=True,
        #   stat="density",
        #   common_norm=False
        # )

        plt.title(method_name, fontsize=20)
        plt.xlabel("Time Steps", fontsize=25)
        plt.xticks(fontsize=20)
        plt.ylabel("Solution", fontsize=25)
        plt.yticks(fontsize=20)
        # plt.xlim(0, 1)
        if i == 1:
          plt.legend(bbox_to_anchor=(0.5, 1.3), loc="upper center", borderaxespad=0., 
            fontsize=25, labels=["HMC", "Method"], ncol=2)
        else:
          plt.legend([],[], frameon=False)
        fig.tight_layout()

      fig_name = f"{basedir}/{resdir}/seed{seed}/var_{var_ind}.png"
      fig.savefig(fig_name)
      # fig.savefig(f"{basedir}/{resdir}/seed{seed}/var_{var_ind}.pdf")
      print("saved to", fig_name)

