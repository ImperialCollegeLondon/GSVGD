import sys
sys.path.append(".")
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from src.metrics import Metric
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:7"

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.001, help='learning rate for g in S-SVGD')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
args = parser.parse_args()

lr = args.lr
noise = "_noise" if args.noise else ""

basedir = f"{args.root}/{args.exp}"

save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# eff_dims = [1, 2, 5]
nparticles_list = [5, 10, 20, 50, 100, 150, 200]

if __name__ == "__main__":

  df_list = []
  for nparticles in nparticles_list:
    # svgd
    svgd_resdir = f"rbf_epoch{args.epochs}_lr0.01_delta0.01_n{nparticles}/seed0"
    svgd_res = pickle.load(open(f"{basedir}/{svgd_resdir}/particles_svgd.p", "rb"))

    # s-svgd
    ssvgd_resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta0.01_n{nparticles}/seed0"
    ssvgd_res = pickle.load(open(f"{basedir}/{ssvgd_resdir}/particles_s-svgd_lrg{args.lr_g}.p", "rb"))

    # gsvgd
    # gsvgd_resdir = f"{basedir}/rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}/seed0"
    # gsvgd_effdim1_res = pickle.load(open(f"{gsvgd_resdir}/particles_gsvgd_effdim1.p", "rb"))

    gsvgd_resdir2 = f"{basedir}/rbf_epoch{args.epochs}_lr0.001_delta0.01_n{nparticles}/seed0"
    gsvgd_effdim2_res = pickle.load(open(f"{gsvgd_resdir2}/particles_gsvgd_effdim2.p", "rb"))

    gsvgd_resdir5 = f"{basedir}/rbf_epoch{args.epochs}_lr0.001_delta0.01_n{nparticles}/seed0"
    gsvgd_effdim5_res = pickle.load(open(f"{gsvgd_resdir5}/particles_gsvgd_effdim5.p", "rb"))

    gsvgd_resdir10 = f"{basedir}/rbf_epoch{args.epochs}_lr0.01_delta0.01_n{nparticles}/seed0"
    gsvgd_effdim10_res = pickle.load(open(f"{gsvgd_resdir10}/particles_gsvgd_effdim10.p", "rb"))
    gsvgd_effdim15_res = pickle.load(open(f"{gsvgd_resdir10}/particles_gsvgd_effdim15.p", "rb"))

    # store results
    res_list = [svgd_res, ssvgd_res, gsvgd_effdim2_res, gsvgd_effdim5_res, gsvgd_effdim10_res, gsvgd_effdim15_res]
    method_names = ["SVGD", "S-SVGD", "GSVGD2", "GSVGD5", "GSVGD10", "GSVGD15"]

    df_new = pd.DataFrame({
      "test_accuracy": [r["test_accuracy"][-1] for r in res_list],
      "Method": method_names,
      "nparticles": [nparticles] * len(method_names)
    })
    df_list.append(df_new)

  df = pd.concat(df_list)

  # svgd_colors = sns.color_palette("crest")[:4]
  # gsvgd_colors = sns.color_palette("flare")[:4]

  fig = plt.figure(figsize=(12, 6))
  g = sns.lineplot(
    data=df, 
    x="nparticles", 
    y="test_accuracy", 
    hue="Method", 
    style="Method", 
    markers=True,
    markersize=12,
    # palette=svgd_colors + gsvgd_colors,
    # hue_order=plot_methods
  )
  # g.set_yscale("log")
  plt.xlabel("Particle Sizes", fontsize=25)
  plt.xticks(fontsize=20)
  plt.ylabel("Test Accuracy", fontsize=25)
  plt.yticks(fontsize=20)
  plt.legend(fontsize=20, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
  fig.tight_layout()
  fig.savefig(f"{save_dir}/test_accuracy.png")
  fig.savefig(f"{save_dir}/test_accuracy.pdf")

  print(f"Saved to ", save_dir + f"/test_accuracy.png")
  
  
  
  

