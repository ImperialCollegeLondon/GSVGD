import sys
sys.path.append(".")
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import pandas as pd
import torch
from src.metrics import Metric
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, default=10000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--xlim', type=float, default=6, help='learning rate')
parser.add_argument('--plot_every', type=int, default=1, help='plot every x checkpoints')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr

noise = "_noise" if args.noise == "True" else ""

basedir = f"{args.root}/{args.exp}"
resdirs = [
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim10",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim20",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim30",
  f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim70",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim90",
  # f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim100"
]
resdirs = [f"{s}/seed0" for s in resdirs]

eff_dims = [1, 2, 5] #[1, 2, 5, 10]

if __name__ == "__main__":


  for path in resdirs:
    path = f"{basedir}/{path}"
    print(f"Plotting for {path}")
    # load results
    res = pickle.load(open(f"{path}/particles.p", "rb"))
    #target_dist = res["target_dist"]
    eff_dims = res["effdims"]
    epochs = res['epochs']
    target, svgd, maxsvgd = res["target"], res["svgd"], res["maxsvgd"]
    gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
    # save last particles in list
    dim = svgd[-1].shape[1]

    device = svgd[-1].device
    target_dist = torch.load(f'{path}/target_dist.p', map_location=device)

    svgd = [x.to(device) for x in svgd]
    maxsvgd = [x.to(device) for x in maxsvgd]
    gsvgd = {s: [x.to(device) for x in gsvgd[s]] for s in gsvgd.keys()}

    # store all results into a dict
    plot_particles = {
      **{"svgd": svgd},
      **{"ssvgd": maxsvgd},
      **gsvgd,
    }

    # trim results
    epochs = epochs[::args.plot_every]
    plot_particles = {s: x[::args.plot_every] for s, x in plot_particles.items()}

    # mean vector for padding
    mix_means = target_dist.sample((10000,)).mean(axis=0)

    # method names for extracting results and plotting
    method_names = ["svgd", "ssvgd"] + [f"gsvgd_effdim{d}" for d in eff_dims]
    plot_names = ["SVGD", "S-SVGD"] + [f"GSVGD{d}" for d in eff_dims]

    for key, method in zip(method_names, plot_names):
      g = plot_particles_hist(
        x_final_ls=plot_particles[key],
        epoch_ls=epochs,
        method=method,
        P=target_dist, 
        d=args.xlim, 
        step=0.1, 
        concat=mix_means[2:],
        savedir=path + f"/{key}_particles_hist.png"
      )

    print(f"Saved to ", path + "/xxx_particles_hist.png")
            
    
    
    

