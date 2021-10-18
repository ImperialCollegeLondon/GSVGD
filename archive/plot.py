import matplotlib.pyplot as plt 
import pickle
import numpy as np
import pandas as pd
import torch
from src.utils import plot_particles, plot_metrics
from src.metrics import Metric
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda:7"

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=str, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, help='no. of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
args = parser.parse_args()

nparticles = args.nparticles
lr = args.lr
noise = "_noise" if args.noise else ""

basedir = f"{args.root}/{args.exp}"

resdirs = [
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim10",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim20",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim30",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim70",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim90",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim100"
]
resdirs = [f"{s}/seed0" for s in resdirs]
# metrics = ["energy", "energy_sub", "wass", "wass_sub",  "mean", "squared", "cov_mat", "cov_mat_sub"]
# metrics_ylabel = ["Energy Distance", "Energy Distance (sub)", "Wasserstein", "Wasserstein (sub)",  
  # "MSE ($x$)", "MSE ($x^2$)", "MSE (Covariance)", "MSE (Covariance 2D)"]
metrics = ["energy", "mean", "squared", "wass", "var", "energy_sub"]
metrics_ylabel = ["Energy", "MSE", "MSE", "Wasserstein", "Variance", "Energy (2D)"]
# metrics = ["energy_sub"]
# metrics_ylabel = ["Energy (2D)"]

save_dir = f"{basedir}/summary_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

eff_dims = [1, 2, 5]

if __name__ == "__main__":

  for met, met_ylabel in zip(metrics, metrics_ylabel):
    print(f"Plotting metric {met}")
    svgd_ls = []
    gsvgd_ls = []
    maxsvgd_ls = []
    dims_ls = []
    metric_fn_ls = []

    for path in resdirs:
      path = f"{basedir}/{path}"
      # load results
      res = pickle.load(open(f"{path}/particles.p", "rb"))
      #target_dist = res["target_dist"]
      eff_dims = res["effdims"]
      target, svgd, maxsvgd = res["target"], res["svgd"], res["maxsvgd"]
      gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
      # save last particles in list
      dim = svgd[-1].shape[1]
      dims_ls.append(dim)
      svgd_ls.append(svgd[-1].to(device))
      gsvgd_ls.append({s: gsvgd[s][-1].to(device) for s in gsvgd.keys()})
      maxsvgd_ls.append(maxsvgd[-1].to(device))

      target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
      target = target_dist.sample((svgd[0].shape[0],))

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

    # compute metrics
    metric_svgd = torch.Tensor([f(x) for x, f in zip(svgd_ls, metric_fn_ls)]).T
    metric_gsvgd = torch.Tensor([[f(x) for x in d.values()] for d, f in zip(gsvgd_ls, metric_fn_ls)])
    metric_maxsvgd = torch.Tensor([f(x) for x, f in zip(maxsvgd_ls, metric_fn_ls)]).T
    
    plot_metrics(
      epochs=dims_ls,
      metric_svgd=metric_svgd,
      metric_gsvgd=metric_gsvgd,
      eff_dims=eff_dims,
      metric_maxsvgd=metric_maxsvgd,
      name=met_ylabel,
      savefile=f"{save_dir}/{met}",
      xlab="Dimension",
      ylog=True
    )

    # save as csv
    metrics_df = pd.DataFrame(
        {
            **{"dim": dims_ls},
            **{"svgd": metric_svgd},
            **{"gsvgd_effdim"+str(eff_dims[i]): [r[i].item() for r in metric_gsvgd] for i in range(len(eff_dims))},
            **{"maxsvgd": metric_maxsvgd}
        }
    )
    metrics_df.to_csv(
        f"{save_dir}/{met}.csv",
        index=False
    )

    print(f"Saved to {save_dir}")
            
    
    
    

