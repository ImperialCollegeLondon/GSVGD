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
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--tune_ssvgd', type=str, default="False", help='metric')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--method', type=str, default="svgd")
args = parser.parse_args()

nparticles = args.nparticles
noise = "_noise" if args.noise == "True" else ""

basedir = f"{args.root}/{args.exp}"
basedir_ssvgd = f"{args.root}/{args.exp}_ssvgd"

lr_list = [0.1, 0.01, 0.001]
delta_list = [0.1, 0.01, 0.001]
lr_g_list = [0.1, 0.01, 0.001]

eff_dims = [1, 2, 5, 10, 15]
# eff_dims = [1, 2, 5, 10]
# eff_dims = [5, 10, 15]
# eff_dims = [10, 15]
method_names = ["svgd", "s-svgd", "gsvgd1", "gsvgd2", "gsvgd5", "gsvgd10", "gsvgd15"]
# method_names = ["svgd", "s-svgd", "gsvgd1", "gsvgd2", "gsvgd5", "gsvgd10"]
# method_names = ["svgd", "s-svgd", "gsvgd10", "gsvgd15"]
hyperparams = {"svgd": "delta", "s-svgd": "lr_g", "gsvgd5": "delta",
  "gsvgd2": "delta", "gsvgd1": "delta", "gsvgd10": "delta", "gsvgd15": "delta"}

save_dir = f"{basedir}/tune_{args.kernel}_epoch{args.epochs}_n{nparticles}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == "__main__":
  
  df_ls = []

  if args.method == "svgd" or args.method == "all":
    for lr in lr_list:
      for delta in delta_list:
        path = f"{basedir}/{args.kernel}_epoch{args.epochs}_lr{lr}_delta0.01_n{nparticles}/seed0"

        # load results
        res = pickle.load(open(f"{path}/particles_svgd.p", "rb"))

        final_test_acc = res["test_accuracy"][-1]
        final_valid_acc = res["valid_accuracy"][-1]

        method_name = "svgd"
        new_df = pd.DataFrame(
          {
            "method": method_name,
            "lr": [lr],
            "delta": [delta],
            "test_accuracy": [final_test_acc],
            "valid_accuracy": [final_valid_acc]
          }
        )
        df_ls.append(new_df)

  if args.method == "s-svgd" or args.method == "all":
    for lr in lr_list:
      for lr_g in lr_g_list:
        path = f"{basedir}/{args.kernel}_epoch{args.epochs}_lr{lr}_delta0.01_n{nparticles}/seed0"

        # load results
        res = pickle.load(open(f"{path}/particles_s-svgd_lrg{lr_g}.p", "rb"))
        
        final_test_acc = res["test_accuracy"][-1]
        final_valid_acc = res["valid_accuracy"][-1]

        method_name = "s-svgd"
        new_df = pd.DataFrame(
          {
            "method": method_name,
            "lr": [lr],
            "lr_g": [lr_g],
            "test_accuracy": [final_test_acc],
            "valid_accuracy": [final_valid_acc]
          }
        )
        df_ls.append(new_df)

  if args.method == "gsvgd" or args.method == "all":
    for effdim in eff_dims:
      for lr in lr_list:
        for delta in delta_list:
          path = f"{basedir}/{args.kernel}_epoch{args.epochs}_lr{lr}_delta{delta}_n{nparticles}/seed0"

          # load results
          res = pickle.load(open(f"{path}/particles_gsvgd_effdim{effdim}.p", "rb"))
          
          final_test_acc = res["test_accuracy"][-1]
          final_valid_acc = res["valid_accuracy"][-1]

          method_name = f"gsvgd{effdim}"
          new_df = pd.DataFrame(
            {
              "method": method_name,
              "lr": [lr],
              "delta": [delta],
              "test_accuracy": [final_test_acc],
              "valid_accuracy": [final_valid_acc]
            }
          )
          df_ls.append(new_df)

  metrics_df = pd.concat(df_ls)

  ## plot
  subplot_c = int(np.ceil(len(method_names) / 2))
  fig = plt.figure(figsize=(18, 12))
  for i, s in enumerate(method_names):
    plt.subplot(2, subplot_c, i+1)
    g = sns.heatmap(
      metrics_df.loc[metrics_df.method == s, :].pivot("lr", hyperparams[s], "test_accuracy"), 
      annot=True,
      fmt=".4g"
    )
    g.invert_yaxis()
    plt.title(s)
    g
  fig.savefig(f"{save_dir}/test_accuracy.png")

  # save as csv
  metrics_df.to_csv(
      f"{save_dir}/test_accuracy.csv",
      index=False
  )

  ## plot validation accuracies
  fig = plt.figure(figsize=(18, 12))
  for i, s in enumerate(method_names):
    plt.subplot(2, subplot_c, i+1)
    g = sns.heatmap(
      metrics_df.loc[metrics_df.method == s, :].pivot("lr", hyperparams[s], "valid_accuracy"), 
      annot=True,
      fmt=".4g"
    )
    g.invert_yaxis()
    plt.title(s)
    g
  fig.savefig(f"{save_dir}/valid_accuracy.png")

  print(f"Saved to {save_dir}")
