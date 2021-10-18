import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,4,5,6,7"
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cuda:7"

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

nparticles_ls = [50, 100, 500, 800]
dims = range(10, 110, 10)

save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

eff_dims = [1, 2, 5]
# gsvgd_methods = [f"gsvgd_effdim{d}" for d in eff_dims]

if __name__ == "__main__":


  print(f"Plotting time")

  df_list = []
  for dim in dims:
    print("loading dim =", dim)
    for nparticles in nparticles_ls:
      resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim{dim}"
      path = f"{resdir}/seed0"

      path = f"{basedir}/{path}"

      # load results
      res = pickle.load(open(f"{path}/particles.p", "rb"))
      eff_dims = res["effdims"]
      epochs = res["epochs"]
      time = res["elapsed_time"]
      svgd, maxsvgd = time["svgd"], time["maxsvgd"]
      gsvgd = {f"GSVGD{d}": time[f"gsvgd_effdim{d}"] for d in eff_dims}
    
      time_dict = {
        "SVGD": svgd,
        "S-SVGD": maxsvgd,
        **gsvgd
      }
      
      method_names = ["SVGD"] + [f"GSVGD{d}" for d in eff_dims] + ["S-SVGD"]
      for method in method_names:
        df_new = pd.DataFrame(
          {
            "Time": time_dict[method],
            "Method": [f"{method}-{nparticles}"],
            "N": [nparticles],
            "dim": [dim]
          }
        )
        df_list.append(df_new)
    
  metrics_df_orig = pd.concat(df_list)
  plot_methods = [f"SVGD-{n}" for n in nparticles_ls] + \
      [f"S-SVGD-{n}" for n in nparticles_ls] + \
      [f"GSVGD1-{n}" for n in nparticles_ls]
  metrics_df = metrics_df_orig.loc[metrics_df_orig.Method.isin(plot_methods), :]

  svgd_colors = sns.color_palette("Greens")[2:len(nparticles_ls)+2]
  ssvgd_colors = sns.color_palette("light:b")[2:len(nparticles_ls)+2]
  gsvgd_colors = sns.color_palette("dark:salmon_r")[:len(nparticles_ls)]
  palatte = svgd_colors + ssvgd_colors + gsvgd_colors
  # palatte = ssvgd_colors + gsvgd_colors

  fig = plt.figure(figsize=(12, 6))
  g = sns.lineplot(
    data=metrics_df, 
    x="dim", 
    y="Time",
    hue="Method", 
    style="Method",
    markers=True,
    markersize=10,
    palette=palatte,
    hue_order=plot_methods
  )
  # g.set_yscale("log")
  plt.xlabel("Dimension", fontsize=25)
  plt.xticks(fontsize=20)
  plt.ylabel("Time", fontsize=25)
  plt.yticks(fontsize=20)
  plt.legend(fontsize=18, markerscale=2, bbox_to_anchor=(1, 0.5), loc='center left')
  fig.tight_layout()
  fig.savefig(f"{save_dir}/time.png")
  fig.savefig(f"{save_dir}/time.pdf")
  
  

