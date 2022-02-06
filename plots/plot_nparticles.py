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

nparticles_ls = [50, 100, 500, 800]
dims = range(10, 110, 10)

metrics = ["energy", "var"]
metrics_ylabs = ["Energy Distance", "Variance"]


save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

eff_dims = [1, 2, 5]

if __name__ == "__main__":

  for met, met_ylabel in zip(metrics, metrics_ylabs):
    print(f"Plotting metric {met}")

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
        target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
        gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}

        target_dist = torch.load(f"{path}/target_dist.p", map_location=device)
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
      
        # compute metric for the final iteration
        svgd_metrics = [metric_fn(x.to(device)) for x in svgd[-1:]]
        s_svgd_metrics = [metric_fn(x.to(device)) for x in s_svgd[-1:]]
        gsvgd_metrics = {
          f"GSVGD{d}": [metric_fn(x.to(device)) for x in gsvgd[f"gsvgd_effdim{d}"]][-1:] for d in eff_dims}
        
        metric_dict = {
          "SVGD": svgd_metrics,
          "S-SVGD": s_svgd_metrics,
          **gsvgd_metrics
        }
        
        method_names = ["SVGD"] + [f"GSVGD{d}" for d in eff_dims] + ["S-SVGD"]
        for method in method_names:
          df_new = pd.DataFrame(
            {
              "Method": [f"{method}-{nparticles}"],
              "N": [nparticles],
              "metric": metric_dict[method],
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

    fig = plt.figure(figsize=(12, 6))
    if met == "var":
      plt.axhline(y=1, linewidth=2, color="k")
    g = sns.lineplot(
      data=metrics_df, 
      x="dim", 
      y="metric", 
      hue="Method", 
      style="Method", 
      markers=True,
      markersize=10,
      palette=palatte,
      hue_order=plot_methods
    )
    g.set_yscale("log")
    plt.xlabel("Dimension", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel(met_ylabel, fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=18, markerscale=2, bbox_to_anchor=(1, 0.5), loc='center left')
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{met}.png")
    fig.savefig(f"{save_dir}/{met}.pdf")
    
    

