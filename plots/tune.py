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
device = "cuda:1"

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--dim', type=int, default=50, help='Dimension')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--metric', type=str, default="energy", help='metric')
parser.add_argument('--tune_ssvgd', type=str, default="False", help='metric')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
args = parser.parse_args()

nparticles = args.nparticles
noise = "_noise" if args.noise == "True" else ""

basedir = f"{args.root}/{args.exp}"
basedir_ssvgd = f"{args.root}/{args.exp}_ssvgd"

# dim = 50
lr_list = [0.1, 0.01, 0.001] # [0.1, 0.01, 0.001, 0.00001]
delta_list = [10., 1., 0.1, 0.01, 0.001] # [0.1, 0.01, 0.001, 0.0001, 0.00001]
lr_g_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]

met = args.metric

save_dir = f"{basedir}/tune_{args.kernel}_epoch{args.epochs}_n{nparticles}_dim{args.dim}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

eff_dims = [1, 2, 5]

if __name__ == "__main__":

  print(f"Computing metric {met}")
  svgd_ls = []
  gsvgd_ls = []
  s_svgd_ls = []
  dims_ls = []
  metric_fn_ls = []

  metrics_df_ls = []

  for lr in lr_list:
    for delta in delta_list:
      path = f"{basedir}/{args.kernel}_epoch{args.epochs}_lr{lr}_delta{delta}_n{nparticles}_dim{args.dim}/seed0"

      # load results
      res = pickle.load(open(f"{path}/particles.p", "rb"))
      #target_dist = res["target_dist"]
      eff_dims = res["effdims"]
      target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
      gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
      # get final particles 
      dim = svgd[-1].shape[1]
      dims_ls.append(dim)
      svgd_final = svgd[-1].to(device)
      s_svgd_final = s_svgd[-1].to(device)
      gsvgd_keys = list(gsvgd.keys())
      gsvgd_final = {s: gsvgd[s][-1].to(device) for s in gsvgd_keys}

      del res

      target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
      target = target_dist.sample((10000,))

      # initialize metric
      if met == "cos":
        w = torch.normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)).reshape((1, -1))
        b = 2 * np.pi * torch.rand((1, dim), device=device)
        metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
          target_dist=target_dist, w=w, b=b, device=device)
      else:
        metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
          target_dist=target_dist, device=device)

      # compute metrics
      method_names = ["svgd", "ssvgd"] + gsvgd_keys
      metric_vals = {
        **{"svgd": metric_fn(svgd_final)},
        **{"ssvgd": metric_fn(s_svgd_final)},
        **{s: metric_fn(gsvgd_final[s]) for s in gsvgd_keys}
      }
      new_df = pd.DataFrame(
        {
          "method": method_names,
          "lr": [lr] * len(method_names),
          "delta": [delta] * len(method_names),
          "metric": [metric_vals[s] for s in method_names]
        }
      )
      metrics_df_ls.append(new_df)
      
  metrics_df = pd.concat(metrics_df_ls)


  subplot_c = int(np.ceil(len(method_names) / 2))
  fig = plt.figure(figsize=(18, 12))
  for i, s in enumerate(method_names):
    plt.subplot(2, subplot_c, i+1)
    g = sns.heatmap(
      metrics_df.loc[metrics_df.method == s, ].pivot("lr", "delta", "metric"), 
      annot=True,
      fmt=".4g"
    )
    g.invert_yaxis()
    plt.title(s)
    g
  fig.savefig(f"{save_dir}/{met}.png")

  # save as csv
  metrics_df.to_csv(
      f"{save_dir}/{met}.csv",
      index=False
  )

  print(f"Saved to {save_dir}")

  
  # # processing S-SVGD
  # if args.tune_ssvgd == "True":
  #   metrics_df_ls = []
  #   for lr in lr_list:
  #     for lr_g in lr_g_list:
  #       path = f"{basedir}/rbf_epoch{args.epochs}_lr{lr}_lrg{lr_g}_n{nparticles}_dim{dim}/seed0"

  #       # load results
  #       res = pickle.load(open(f"{path}/particles.p", "rb"))
  #       #target_dist = res["target_dist"]
  #       eff_dims = res["effdims"]
  #       target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
  #       gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
  #       # get final particles 
  #       dim = svgd[-1].shape[1]
  #       dims_ls.append(dim)
  #       svgd_final = svgd[-1].to(device)
  #       s_svgd_final = s_svgd[-1].to(device)
  #       gsvgd_keys = list(gsvgd.keys())
  #       gsvgd_final = {s: gsvgd[s][-1].to(device) for s in gsvgd_keys}

  #       target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
  #       target = target_dist.sample((svgd[0].shape[0],))

  #       # initialize metric
  #       if met == "cos":
  #         w = torch.normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)).reshape((1, -1))
  #         b = 2 * np.pi * torch.rand((1, dim), device=device)
  #         metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
  #           target_dist=target_dist, w=w, b=b, device=device)
  #       else:
  #         metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
  #           target_dist=target_dist, device=device)

  #       # compute metrics
  #       method_names = ["svgd", "ssvgd"] + gsvgd_keys
  #       metric_vals = {
  #         **{"svgd": metric_fn(svgd_final)},
  #         **{"ssvgd": metric_fn(s_svgd_final)},
  #         **{s: metric_fn(gsvgd_final[s]) for s in gsvgd_keys}
  #       }
  #       new_df = pd.DataFrame(
  #         {
  #           "method": method_names,
  #           "lr": [lr] * len(method_names),
  #           "lr_g": [lr_g] * len(method_names),
  #           "metric": [metric_vals[s] for s in method_names]
  #         }
  #       )
  #       metrics_df_ls.append(new_df)
        
  #   metrics_df = pd.concat(metrics_df_ls)


  #   subplot_c = int(np.ceil(len(method_names) / 2))
  #   fig = plt.figure(figsize=(18, 12))
  #   g = sns.heatmap(
  #     metrics_df.loc[metrics_df.method == "ssvgd", ].pivot("lr", "lr_g", "metric"), 
  #     annot=True,
  #     fmt=".4g"
  #   )
  #   g.invert_yaxis()
  #   plt.title("S-SVGD")
  #   g
  #   fig.savefig(f"{save_dir}/{met}_ssvgd.png")

  #   # save as csv
  #   metrics_df.to_csv(
  #       f"{save_dir}/{met}_ssvgd.csv",
  #       index=False
  #   )

  #   print(f"Saved to {save_dir}")
            
    
    
    

