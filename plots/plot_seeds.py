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
parser.add_argument('--nparticles', type=str, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
parser.add_argument('--metric', type=str, default="energy", help='evaluation metric')
args = parser.parse_args()

nparticles = args.nparticles
lr = args.lr

basedir = f"{args.root}/{args.exp}"

resdirs = [
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim10",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim20",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim30",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim40",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim60",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim70",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim80",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim90",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim100",
]

metrics = [args.metric] # select from the keys of metrics_ylabels below
metrics_ylabels = {
  "energy": "Energy Distance", 
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
  "cov_error": "Covariance Estimation Error"
}

save_dir = f"{basedir}/{args.kernel}_summary_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

seed_list = range(20)

if __name__ == "__main__":

  for met in metrics:
    met_ylabel = metrics_ylabels[met]

    print(f"Plotting metric {met}")
    svgd_ls = []
    gsvgd_ls = []
    s_svgd_ls = []
    metric_fn_ls = []
    df_list = []

    for res_path in resdirs:
      print(f"Computing {met} for {res_path}")
      for seed in seed_list:
        path = f"{basedir}/{res_path}/seed{seed}"

        # load results
        res = pickle.load(open(f"{path}/particles.p", "rb"))
        eff_dims = res["effdims"]
        target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
        gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}

        # save last particles in list
        dim = svgd[-1].shape[1]

        del res

        target_dist = torch.load(f"{path}/target_dist.p", map_location=device)
        target = target_dist.sample((20000, ))

        # initialize metric
        if met == "cos":
          w = torch.normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)).reshape((1, -1))
          b = 2 * np.pi * torch.rand((1, dim), device=device)
          metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
            target_dist=target_dist, w=w, b=b, device=device)
        else:
          metric_fn = Metric(metric=met, x_init=svgd[0].clone(), x_target=target.clone(), 
            target_dist=target_dist, device=device)
        metric_fn_ls.append(metric_fn)

        # compute metric
        plot_ind = -1
        particles = {
          "SVGD": svgd[plot_ind].to(device),
          "S-SVGD": s_svgd[plot_ind].to(device),
          **{f"GSVGD{d} (ours)": gsvgd[f"gsvgd_effdim{d}"][plot_ind].to(device) for d in eff_dims}
        }

        method_names = ["SVGD"] + [f"GSVGD{d} (ours)" for d in eff_dims] + ["S-SVGD"]
        new_df = pd.DataFrame(
          {
            "dim": [dim] * len(method_names),
            "method": method_names,
            "metric": [metric_fn(particles[s]) for s in method_names],
            "seed": [seed] * len(method_names)
          }
        )
        df_list.append(new_df)
      
    metrics_df = pd.concat(df_list)
    metrics_df = metrics_df.reset_index(drop=True)

    # save as csv
    metrics_df.to_csv(
        f"{save_dir}/{met}.csv",
        index=False
    )


    fig = plt.figure(figsize=(10, 8))
    if met == "var":
      plt.axhline(y=1, linewidth=3, color="k")
    g = sns.lineplot(
      data=metrics_df, 
      x="dim", 
      y="metric", 
      hue="method", 
      style="method", 
      markers=True,
      markersize=18,
    )

    if met != "var":
      g.set_yscale("log")
    plt.xlabel("Dimension", fontsize=40)
    plt.xticks(fontsize=32)
    plt.ylabel(met_ylabel, fontsize=40)
    plt.yticks(fontsize=32)
    if met == "var":
      plt.legend(fontsize=28, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center right')
    else:
      plt.legend(fontsize=28, markerscale=2)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{met}.png")
    fig.savefig(f"{save_dir}/{met}.pdf")
    
    # save as csv
    metrics_df.to_csv(
        f"{save_dir}/{met}.csv",
        index=False
    )

    print(f"Saved to {save_dir}")
            
    
    
    

