import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,5,6,7"
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting final particles.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=500, help='Num of particles')
parser.add_argument('--epochs', type=int, default=10000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--xlim', type=float, default=6, help='learning rate')
parser.add_argument('--format', type=str, default="png", help='format of figs')
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if bool(args.noise) else ""

basedir = f"{args.root}/{args.exp}"
resdirs = [
  # f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim2",
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
]
resdirs = [f"{s}/seed0" for s in resdirs]

if __name__ == "__main__":


  for path in resdirs:
    path = f"{basedir}/{path}"
    print(f"Plotting for {path}")
    # load results
    res = pickle.load(open(f"{path}/particles.p", "rb"))
    #target_dist = res["target_dist"]
    eff_dims = res["effdims"]
    target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
    gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}
    dim = svgd[-1].shape[1]

    target_dist = torch.load(f'{path}/target_dist.p', map_location=device)
  
    gsvgd_titles = ["GSVGD" + str(i) for i in eff_dims]
    gsvgd_keys = [f"gsvgd_effdim{i}" for i in eff_dims]

    # mean vector for padding
    mix_means = target_dist.sample((10000,)).mean(axis=0)

    # methods to be plotted
    method_names = ["SVGD", "S-SVGD"] + ["GSVGD1"] # [f"GSVGD{d}" for d in eff_dims]

    target_samples = target_dist.sample((10000,)).cpu().numpy()

    t = -1
    print(len(svgd)-1)
    epochs = [0, 1, 3, 10]
    final_particles_dict = {
      "Target": [target_samples] * len(epochs),
      "SVGD": [svgd[t].cpu().numpy() for t in epochs],
      "S-SVGD": [s_svgd[t].cpu().numpy() for t in epochs],
      **{n: [gsvgd[s][t].cpu().numpy() for t in epochs] for n, s in zip(gsvgd_titles, gsvgd_keys)}
    }

    # plot density
    subplot_c = len(epochs) # int(np.ceil(np.sqrt(len(final_particles_dict))))
    subplot_r = len(method_names) # int(np.ceil(len(final_particles_dict) / subplot_c))

    dim1, dim2 = 0, 1
    fig = plt.figure(figsize=(subplot_c*3, subplot_r*3))
    for i, method in enumerate(method_names):
      print("Plotting", method)
      for t_ind, t in enumerate(epochs):
        x = final_particles_dict[method][t_ind]
        df = pd.DataFrame(x[:, [dim1, dim2]], columns=["dim1", "dim2"])
        target = final_particles_dict["Target"][t_ind]
        df_target = pd.DataFrame(target[:, [dim1, dim2]], columns=["dim1", "dim2"])

        cut = 22 if method == "SVGD" else 22
        
        ## density only
        # sns.kdeplot(
        #   data=df, x="dim1", y="dim2", fill=True,
        #   thresh=0, levels=100, cmap="viridis", cut=cut
        # )
        ## density and particles
        g = plt.subplot(len(method_names), len(epochs), i*len(epochs)+t_ind+1)
        sns.kdeplot(
          data=df_target, 
          x="dim1", y="dim2", fill=True,
          thresh=0, levels=100, cmap="viridis", cut=cut
        )
        if method != "Target":
          sns.scatterplot(
            data=df, x="dim1", y="dim2", alpha=0.4, size=0.05, 
            color="red", legend=False
          )

        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set(ylim=(-args.xlim, args.xlim), xlim=(-args.xlim, args.xlim))
        # if i != len(method_names)-1:
        #   g.set(xticks=[])
        # if t_ind != 0:
        #   g.set(yticks=[])
        g.set(xticks=[])
        g.set(yticks=[])
        if t_ind == 0:
          plt.ylabel(method, fontsize=22)
        
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        plt.title(f"t = {t*5}", fontsize=22)
    
    fig.tight_layout()
    fig.savefig(path + f"/intermed_particles.{args.format}", dpi=500)


    print(f"Saved to ", path + "/intermed_particles.png")
            
    
    
    

