import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
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

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, default=10000, help='Num of epochs')
parser.add_argument('--dim', type=int, default=50, help='Dim')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--metric', type=str, default="energy", help='metric')
parser.add_argument('--format', type=str, default="png", help='format of figs')
parser.add_argument('--kernel', type=str, default="rbf", help='format of figs')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if args.noise=="True" else ""
met = args.metric

metrics_ylabs = {
  "energy": "Energy Distance", 
  "energy2": "Energy Distance", 
  "mean": "MSE", 
  "squared": "MSE", 
  "cos": "MSE",
  "wass": "Wasserstein", 
  "var": "Variance", 
  "var_sub": "Variance (other dims)", 
  "energy_sub": "Energy Distance (2D)",
  "pam": "PAM",
  "pam_diff": "Delta PAM",
  "alpha": "Alpha",
  "pamrf": "PAMRF",
}

basedir = f"{args.root}/{args.exp}"
resdirs = [
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim10",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim20",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim30",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim40",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim60",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim70",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim80",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim90",
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim100",
]
seed_list = range(20)

if __name__ == "__main__":


  for res_path in resdirs:
    print(f"Plotting for {res_path}")
    df_list = []

    for seed in seed_list:
      path = f"{basedir}/{res_path}/seed{seed}"
      # load results
      res = pickle.load(open(f"{path}/particles.p", "rb"))
      eff_dims = res["effdims"]
      epochs = res["epochs"]

      target, svgd, maxsvgd = res["target"], res["svgd"], res["maxsvgd"]
      gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
      # save last particles in list
      dim = svgd[-1].shape[1]
      
      epochs = res["epochs"]

      target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
      target = target_dist.sample((20000,))

      #? save dataset used to compute metric
      particles_dic = {
        "epochs": epochs,
        "SVGD": svgd,
        "S-SVGD": maxsvgd,
        **{f"GSVGD{d}": [x.cpu() for x in gsvgd[f"gsvgd_effdim{d}"]] for d in eff_dims},
        "target": target.cpu()
      }
      pickle.dump(particles_dic, open(f"{path}/particles_dict.p", "wb"))

      # initialize metric
      if met in ["pam", "pamrf"]:
        epochs = epochs[1:]
        metric_dict = {
          "SVGD": res[met]["svgd"],
          "S-SVGD": res[met]["maxsvgd"],
          **{f"GSVGD{d}": [x for x in res[met][f"gsvgd_effdim{d}"]] for d in eff_dims}
        }

      elif met == "pam_diff":
        epochs = epochs[2:]
        svgd_metrics = res["pam"]["svgd"]
        maxsvgd_metrics = res["pam"]["maxsvgd"]
        gsvgd_metrics = {}
        for d in eff_dims:
          l = res["pam"][f"gsvgd_effdim{d}"]
          l = [abs(l[i] - l[i-1]) / (l[i-1] + 1e-16) for i in range(1, len(l))]
          gsvgd_metrics[f"GSVGD{d}"] = l

        metric_dict = {
          "SVGD": [abs(svgd_metrics[i] - svgd_metrics[i-1]) / (svgd_metrics[i-1] + 1e-16) for i in range(1, len(svgd_metrics))],
          "S-SVGD": [abs(maxsvgd_metrics[i] - svgd_metrics[i-1]) / (maxsvgd_metrics[i-1] + 1e-16) for i in range(1, len(maxsvgd_metrics))],
          **gsvgd_metrics
        }

      elif met == "alpha":
        epochs = [x[0] for x in res["alpha_tup"]["gsvgd_effdim1"]][100:200]
        metric_dict = {
          "SVGD": [0] * len(epochs),
          "S-SVGD": [0] * len(epochs),
          **{f"GSVGD{d}": [x[1] for x in res["alpha_tup"][f"gsvgd_effdim{d}"]][100:200] for d in eff_dims}
        }
      
      else:
        if met == "cos":
          w = torch.normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)).reshape((1, -1))
          b = 2 * np.pi * torch.rand((1, dim), device=device)
          metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
            target_dist=target_dist, w=w, b=b, device=device)
        else:
          metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
            target_dist=target_dist, device=device)
        
        svgd_metrics = [metric_fn(x.to(device)) for x in svgd]
        maxsvgd_metrics = [metric_fn(x.to(device)) for x in maxsvgd]
        gsvgd_metrics = [[metric_fn(x.to(device)) for x in gsvgd[s]] for s in gsvgd.keys()]

        # compute metric
        metric_dict = {
          "SVGD": svgd_metrics,
          "S-SVGD": maxsvgd_metrics,
          **{f"GSVGD{d}": [metric_fn(x.to(device)) for x in gsvgd[f"gsvgd_effdim{d}"]] for d in eff_dims}
        }

      method_names = ["SVGD"] + [f"GSVGD{d}" for d in eff_dims] + ["S-SVGD"]
      for method in method_names:
        epochs = epochs
        df_new = pd.DataFrame(
          {
            "epochs": epochs,
            "method": [method] * len(epochs),
            "metric": metric_dict[method],
            "seed": [seed] * len(epochs)
          }
        )
        df_list.append(df_new)
        

    metrics_df = pd.concat(df_list)
    metrics_df = metrics_df.reset_index(drop=True)

    # # 90%-CI (wrong!)
    # metrics_df["lower"] = np.nan
    # metrics_df["upper"] = np.nan
    # for method in method_names:
    #   for t in epochs:
    #     select_ind = (metrics_df.method == method) & (metrics_df.epochs == t)
    #     subdf = metrics_df[select_ind]
    #     std = subdf.std() / np.sqrt(len(seed_list))
    #     mean = subdf.mean()
    #     metrics_df.loc[select_ind, "lower"] = mean - 1.96*std
    #     metrics_df.loc[select_ind, "upper"] = mean + 1.96*std


    # fig = plt.figure(figsize=(10, 6))
    # metrics_df = metrics_df.loc[metrics_df.method != "SVGD"]
    fig = plt.figure(figsize=(12, 8))
    g = sns.lineplot(
      data=metrics_df, 
      x="epochs", 
      y="metric", 
      hue="method", 
      style="method", 
      markers=True,
      markersize=14,
      # ci=68
    )

    # # plot 90%-CI (wrong!)
    # for method in method_names:
    #   sub_metrics_df = metrics_df.loc[metrics_df.method == method]
    #   sub_metrics_df = sub_metrics_df.sort_values(["method", "epochs"]).reset_index(drop=True)
    #   plt.fill_between(data=sub_metrics_df, x="epochs", y1="lower", y2="upper", alpha=0.2)

    g.set_yscale("log")
    plt.xlabel("Iterations", fontsize=30)
    plt.xticks(fontsize=22)
    plt.ylabel(metrics_ylabs[met], fontsize=30)
    plt.yticks(fontsize=26)
    # plt.legend(fontsize=25, markerscale=2.5, bbox_to_anchor=(1.01, 0.8), loc='upper left')
    plt.legend(fontsize=25, markerscale=2.5, loc='center right')
    fig.tight_layout()
    save_path = f"{basedir}/{res_path}"
    fig.savefig(f"{save_path}/{met}.png")
    fig.savefig(f"{save_path}/{met}.pdf")
    metrics_df.to_csv(f"{save_path}/{met}.csv", index=False)

    print(f"Saved to {save_path}")

    
  

            
    
    
    

