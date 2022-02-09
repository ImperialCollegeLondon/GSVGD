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
] # (m, M) pairs
seeds = range(20)

save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == "__main__":

  print(f"Plotting ablation study on M")
  df_list = []

  svgd_metric_sum, s_svgd_metric_sum = 0, 0

  for seed in seeds:
    ## initialize dirs
    resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim{dim}"
    path = f"{resdir}/seed{seed}"
    path = f"{basedir}/{path}"

    ## load results
    res_svgd = pickle.load(open(f"{path}/particles_svgd.p", "rb"))
    svgd = res_svgd["svgd"][-1].to(device)

    res_s_svgd = pickle.load(open(f"{path}/particles_s-svgd.p", "rb"))
    s_svgd = res_s_svgd["s_svgd"][-1].to(device)
    
    ## initialize evaluation metric
    target_dist = torch.load(f"{path}/target_dist.p", map_location=device)
    target = target_dist.sample((20000,))
    
    metric_fn = Metric(metric="var", x_init=svgd[0].clone(), x_target=target.clone(), 
      target_dist=target_dist, device=device)

    ## compute metric
    svgd_metric_sum += metric_fn(svgd)
    s_svgd_metric_sum += metric_fn(s_svgd)

    for effdim, M_list in M_dict:
      ## load gsvgd results
      gsvgd = {}
      for M in M_list:
        res_gsvgd = pickle.load(open(f"{path}/particles_gsvgd_m{effdim}_M{M}.p", "rb"))
        gsvgd = res_gsvgd[f"gsvgd_effdim{effdim}"][-1].to(device)

        gsvgd_metric = metric_fn(gsvgd)
      
        df_new = pd.DataFrame(
          {
            "metric": [gsvgd_metric],
            "Method": [f"GSVGD{effdim}"],
            "seed": seed,
            "M": [M],
            "m": [effdim]
          }
        )
        df_list.append(df_new)


  ## aggregate results
  metrics_df = pd.concat(df_list, ignore_index=True)
  metrics_df["Coverage"] = metrics_df.M * metrics_df.m / dim

  ## compute mean result for SVGD and S-SVGD
  svgd_metric_mean = svgd_metric_sum / len(seeds)
  s_svgd_metric_mean = s_svgd_metric_sum / len(seeds)
  
  ## set legend colours
  svgd_color = sns.color_palette("Greens")[2]
  ssvgd_color = sns.color_palette("light:b")[2]
  gsvgd_colors = [sns.color_palette("dark:salmon_r")[i] for i in [0, 2, 4]]
  palatte = gsvgd_colors

  ## plot
  fig = plt.figure(figsize=(12, 6))
  plt.axhline(y=1, linewidth=3, color="k")
  plt.axhline(y=svgd_metric_mean, linewidth=2, color=svgd_color, label="SVGD", linestyle="dashed")
  plt.axhline(y=s_svgd_metric_mean, linewidth=2, color=ssvgd_color, label="S-SVGD", linestyle="dashdot")
  g = sns.lineplot(
    data=metrics_df, 
    x="Coverage", 
    y="metric",
    hue="Method", 
    style="Method", 
    markers=True,
    markersize=14,
    palette=palatte,
  )
  plt.xlabel("Dimension Coverage", fontsize=30)
  plt.xticks(fontsize=25)
  plt.ylabel("Variance", fontsize=30)
  plt.yticks(fontsize=25)
  plt.legend(fontsize=25, markerscale=1, bbox_to_anchor=(1, 0.5), loc='center left')
  fig.tight_layout()
  fig.savefig(f"{save_dir}/ablation.png")
  fig.savefig(f"{save_dir}/ablation.pdf")
  print("saved to", f"{save_dir}/ablation.png")
    
