import matplotlib.pyplot as plt 
import pickle
import numpy as np
import pandas as pd
import torch
from src.utils import plot_metrics, plot_metrics_individual
from src.metrics import Metric
import argparse
import os

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, default=10000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--xlim', type=float, default=6, help='learning rate')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if bool(args.noise) else ""

basedir = f"{args.root}/{args.exp}"
resdirs = [
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim3",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim10",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim20",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim30",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim50",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim60",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim80",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_samples{nparticles}_dim100"
]

eff_dims = [1, 2, 5] #[1, 2, 5, 10]

if __name__ == "__main__":


  for path in resdirs:
    path = f"{basedir}/{path}"
    print(f"Plotting for {path}")
    # load results
    res = pickle.load(open(f"{path}/particles.p", "rb"))

    epochs = res["epochs"]

    # extract pam
    pam = res["pam"]
    pam_svgd, pam_maxsvgd = pam["svgd"], pam["maxsvgd"]
    pam_gsvgd = [pam[s] for s in pam.keys() if "gsvgd" in s]
    # #! this is frobenius norm now
    fro = res["fro"]
    fro_gsvgd = [[x.cpu() for x in fro[s]] for s in fro.keys() if "gsvgd" in s]

    plot_metrics(
      epochs=epochs[1:],
      metric_svgd=pam_svgd,
      metric_gsvgd=np.array(pam_gsvgd).T,
      eff_dims=eff_dims,
      metric_maxsvgd=pam_maxsvgd,
      name="PAM",
      savefile=f"{path}/pam",
      xlab="Iterations",
      ylog=True
    )

    # plot frobenius norm of change in proj matrices
    method_names = ["GSVGD" + str(i) for i in eff_dims]
    pam_ls = zip(method_names, fro_gsvgd)
    plot_metrics_individual(
      epochs=epochs[1:],
      metrics_ls=pam_ls,
      name="$|| A_{l} - A_{l-1} ||$",
      savefile=f"{path}/frobenius",
      xlab="Iterations",
      ylog=True
    )

    print(f"Saved to ", path + "/pam.pdf")
            