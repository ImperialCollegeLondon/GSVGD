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
parser.add_argument('--kernel', type=str, default="rbf", help='kernel')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if bool(args.noise) else ""

basedir = f"{args.root}/{args.exp}"
resdirs = [
  f"{args.kernel}_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}_dim50",
]
resdirs = [f"{s}/seed0" for s in resdirs]

if __name__ == "__main__":


  for path in resdirs:
    path = f"{basedir}/{path}"
    print(f"Plotting for {path}")
    # load results
    res = pickle.load(open(f"{path}/particles.p", "rb"))

    eff_dims = res["effdims"]
    target, svgd, s_svgd = res["target"], res["svgd"], res["s_svgd"]
    gsvgd = {s: res[s] for s in [f"gsvgd_effdim{d}" for d in eff_dims]}

    dim = svgd[-1].shape[1]

    target_dist = torch.load(f"{path}/target_dist.p", map_location=device)
  
    gsvgd_titles = ["GSVGD" + str(i) for i in eff_dims]
    gsvgd_keys = [f"gsvgd_effdim{i}" for i in eff_dims]
    final_particles_dic = [
      ("SVGD", svgd[-1].to(device)),
      ("S-SVGD", s_svgd[-1].to(device))
    ] + [(n, gsvgd[s][-1].to(device)) for n, s in zip(gsvgd_titles, gsvgd_keys)]

    # mean vector for padding
    mix_means = target_dist.sample((10000,)).mean(axis=0)

    method_names = ["Target", "SVGD", "S-SVGD"] + [f"GSVGD{d}" for d in eff_dims]

    target_samples = target_dist.sample((10000,)).cpu().numpy()

    final_particles_dict = {
      "Target": target_samples,
      "SVGD": svgd[-1].cpu().numpy(),
      "S-SVGD": s_svgd[-1].cpu().numpy(),
      **{n: gsvgd[s][-1].cpu().numpy() for n, s in zip(gsvgd_titles, gsvgd_keys)}
    }

    # plot density
    subplot_c = int(np.ceil(np.sqrt(len(final_particles_dict))))
    subplot_r = int(np.ceil(len(final_particles_dict) / subplot_c))

    dim1, dim2 = 0, 1 # dimensions to plot
    fig = plt.figure(figsize=(subplot_c*3, subplot_r*3))
    for i, method in enumerate(method_names):
      x = final_particles_dict[method]
      df = pd.DataFrame(x[:, [dim1, dim2]], columns=["dim1", "dim2"])
      target = final_particles_dict["Target"]
      df_target = pd.DataFrame(target[:, [dim1, dim2]], columns=["dim1", "dim2"])

      g = plt.subplot(2, subplot_c, i+1)
      cut = 22 if method == "SVGD" else 22 # levels for contour plots
  
      ## density and particles
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
      if i < subplot_c:
        g.set(xticks=[])
      if i != 0 and i != subplot_c:
        g.set(yticks=[])
      
      plt.xticks(fontsize=20)
      plt.yticks(fontsize=20)
      plt.title(method, fontsize=22)
    
    fig.tight_layout()
    fig.savefig(path + f"/final_particles.png", dpi=500)
    fig.savefig(path + f"/final_particles.pdf")

    print(f"Saved to ", path + "/final_particles.png")