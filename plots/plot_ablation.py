import os
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
from src.metrics import Metric
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
args = parser.parse_args()

lr = args.lr
noise = "_noise" if args.noise else ""

basedir = f"{args.root}/{args.exp}"

nparticles = 500
dim = 50
M_dict = [
  (1, [1, 2, 5, 10, 20, 30, 40, 50]),
  (2, [1, 2, 5, 10, 15, 20, 25]),
  (5, [1, 2, 4, 6, 8, 10])
]

save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == "__main__":


  print(f"Plotting ablation study on M")
  df_list = []

  ## initialize dirs
  resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim{dim}"
  path = f"{resdir}/seed0"
  path = f"{basedir}/{path}"

  ## load results
  res_svgd = pickle.load(open(f"{path}/particles_svgd.p", "rb"))
  svgd = res_svgd["svgd"][0].to(device)

  res_s_svgd = pickle.load(open(f"{path}/particles_s-svgd.p", "rb"))
  s_svgd = res_s_svgd["s_svgd"][0].to(device)
  
  ## initialize evaluation metric
  target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
  target = target_dist.sample((20000,))
  
  metric_fn = Metric(metric="energy", x_init=svgd[0].clone(), x_target=target.clone(), 
    target_dist=target_dist, device=device)

  ## compute metric
  svgd_metric = metric_fn(svgd)
  s_svgd_metric = metric_fn(s_svgd)

  for effdim, M_list in M_dict:
    print("loading m =", effdim, "and M =", M_list)

    ## load gsvgd results
    gsvgd = {}
    for M in M_list:
      res_gsvgd = pickle.load(open(f"{path}/particles_gsvgd_m{effdim}_M{M}.p", "rb"))
      # gsvgd = {**gsvgd, f"GSVGD{effdim}": res_gsvgd[f"gsvgd_effdim{effdim}"]}
      gsvgd = res_gsvgd[f"gsvgd_effdim{effdim}"][0].to(device)

      gsvgd_metric = metric_fn(gsvgd)
    
      df_new = pd.DataFrame(
        {
          "metric": gsvgd_metric,
          "Method": f"GSVGD{effdim}",
          "M": M,
          "m": effdim
        }
      )
      df_list.append(df_new)


  ## aggregate results
  metrics_df = pd.concat(df_list, ignore_index=True)
  metrics_df["Coverage"] = metrics_df.M * metrics_df.m / dim
  
  ## set legend colours
  svgd_colors = sns.color_palette("Greens")[2:3]
  ssvgd_colors = sns.color_palette("light:b")[2:3]
  gsvgd_colors = sns.color_palette("dark:salmon_r")[:len(M_list)]
  palatte = svgd_colors + ssvgd_colors + gsvgd_colors

  ## plot
  fig = plt.figure(figsize=(12, 6))
  g = sns.lineplot(
    data=metrics_df, 
    x="Coverage", 
    y="metric",
    hue="Method", 
    markers=True,
    markersize=14,
    palette=palatte,
    # legend=False
  )
  # g.set_yscale("log")
  plt.xlabel("Dimension Coverage", fontsize=30)
  plt.xticks(fontsize=25)
  plt.ylabel("Energy Distance", fontsize=30)
  plt.yticks(fontsize=25)
  # plt.legend(fontsize=18, markerscale=1, bbox_to_anchor=(1, 0.5), loc='center left', labels=plot_methods)
  plt.legend(fontsize=18, markerscale=1)
  fig.tight_layout()
  fig.savefig(f"{save_dir}/ablation.png")
  fig.savefig(f"{save_dir}/ablation.pdf")
  print("saved to", f"{save_dir}/ablation.png")
    
  

