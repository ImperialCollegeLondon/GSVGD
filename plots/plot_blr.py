import matplotlib.pyplot as plt 
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Plotting metrics.')
parser.add_argument('--exp', type=str, help='Experiment to run')
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--nparticles', type=int, default=100, help='Num of particles')
parser.add_argument('--epochs', type=int, default=1000, help='Num of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_g', type=float, default=0.001, help='learning rate')
parser.add_argument('--delta', type=float, help='stepsize for projections')
parser.add_argument('--noise', type=str, default="True", help='noise')
parser.add_argument('--format', type=str, default="png", help='format of figs')
args = parser.parse_args()
nparticles = args.nparticles
lr = args.lr
noise = "_noise" if args.noise=="True" else ""

basedir = f"{args.root}/{args.exp}"
resdir = f"rbf_epoch{args.epochs}_lr{lr}_delta{args.delta}_n{nparticles}"
resdir_svgd = f"rbf_epoch{args.epochs}_lr0.1_delta0.01_n{nparticles}"
resdir_ssvgd = f"rbf_epoch{args.epochs}_lr0.1_delta0.01_n{nparticles}"

seeds = range(5)

if __name__ == "__main__":

  df_list = []
  df_early_stop_list = []
  for seed in seeds:
    print(f"loading seed {seed}")
    path = f"{basedir}/{resdir}/seed{seed}"
    path_svgd = f"{basedir}/{resdir_svgd}/seed{seed}"
    path_ssvgd = f"{basedir}/{resdir_ssvgd}/seed{seed}"

    # load results
    svgd_res = pickle.load(open(f"{path_svgd}/particles_svgd.p", "rb"))
    ssvgd_res = pickle.load(open(f"{path_ssvgd}/particles_s-svgd_lrg{args.lr_g}.p", "rb"))

    method_ls = [svgd_res, ssvgd_res]
    method_names = ["SVGD", "S-SVGD"]
    eff_dim_ls = [-1, -1]

    eff_dims = [1, 10, 55] # [1, 2, 5, 10, 20, 30, 40, 50, 55]
    gsvgd_show = "GSVGD5"
    for eff_dim in eff_dims:
      gsvgd_res = pickle.load(open(f"{path}/particles_gsvgd_effdim{eff_dim}.p", "rb"))
      method_ls.append(gsvgd_res)
      method_names.append(f"GSVGD{eff_dim}")
      eff_dim_ls.append(eff_dim)

    num_batches = method_ls[2]["nbatches"]
    nshow = 100 # num of epochs to show in the plot
    for i, (res, method_name) in enumerate(zip(method_ls, method_names)):

      iterations = [x / num_batches for x in res["epochs"][:nshow]]
      if "GSVGD" in method_name:
        rep = 1
        eff_dims_append = [int(method_name.split("GSVGD")[-1])] * len(iterations)
      else:
        rep = len(eff_dims)
        eff_dims_append = np.repeat(eff_dims, len(iterations)).tolist()

      df_new = pd.DataFrame({
        # "iterations": res["epochs"][1:600][::10], # steps
        "iterations": iterations * rep, # epochs
        "test_accuracy": res["test_accuracy"][:nshow] * rep,
        "valid_accuracy": res["valid_accuracy"][:nshow] * rep,
        "test_ll": res["test_ll"][:nshow] * rep,
        "valid_ll": res["valid_ll"][:nshow] * rep,
        "method": method_name,
        "seed": seed,
        "eff_dim": eff_dims_append
      })
      df_list.append(df_new)

      # for early stopping
      df_new = df_new.loc[df_new.iterations <= 8] #? consider first few iterations
      df_new_early_stop = df_new.iloc[[df_new.valid_accuracy.idxmax()]].reset_index(drop=True)
      if "GSVGD" in method_name:
        df_new_early_stop["method"] = "GSVGD"
      else:
        df_new_early_stop = df_new_early_stop.loc[df_new_early_stop.index.tolist() * len(eff_dims)]
        df_new_early_stop["eff_dim"] = eff_dims

      df_early_stop_list.append(df_new_early_stop)

  df = pd.concat(df_list)
  df_early_stop = pd.concat(df_early_stop_list)


  for metric in ["test_accuracy", "valid_accuracy"]:
    ## accuracies
    fig = plt.figure(figsize=(12, 6))
    g = sns.lineplot(
      data=df, 
      x="iterations", 
      y=metric, 
      hue="method", 
      style="method", 
      # markers=True,
      # markersize=8,
      # alpha=1,
      ci=None
    )
    # g.set_yscale("log")
    # g.set_xscale("log")
    # plt.xlabel("Training Steps", fontsize=25)
    plt.xlabel("Epochs", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel("Test Accuracy", fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
    fig.tight_layout()
    fig.savefig(f"{basedir}/{resdir}/{metric}.png")
    fig.savefig(f"{basedir}/{resdir}/{metric}.pdf")
      
    print(f"Saved to {basedir}/{resdir}/{metric}.pdf")

    ## accuracy against effdim
    df_truncated = df.loc[(~df.method.isin(["SVGD", "S-SVGD"])) & (df.iterations >= 8)]
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(
      data=df_truncated,
      x="eff_dim",
      y=metric,
      label="GSVGD",
      ci=None # none since only average accuracy is of interest
    )
    plt.axhline(df.loc[(df.method == "SVGD") & (df.iterations >= 8), metric].mean(), label="SVGD", color="r")
    plt.axhline(df.loc[(df.method == "S-SVGD") & (df.iterations >= 8), metric].mean(), label="S-SVGD", color="orange", ls="--")
    plt.legend(fontsize=20, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
    fig.tight_layout()
    fig.savefig(f"{basedir}/{resdir}/{metric}_average.png")

  ## test accuracy with early stop
  fig = plt.figure(figsize=(12, 6))
  sns.lineplot(
    data=df_early_stop,
    x="eff_dim",
    y="test_accuracy",
    hue="method",
    # ci=68#"sd"
  )
  plt.legend(fontsize=16, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
  fig.tight_layout()
  fig.savefig(f"{basedir}/{resdir}/test_accuracy_early_stop.png")


  ## loglikelihood
  for metric in ["test_ll", "valid_ll"]:
    ## loglikelihood
    fig = plt.figure(figsize=(12, 6))
    g = sns.lineplot(
      data=df, 
      x="iterations", 
      y=metric, 
      hue="method", 
      style="method", 
      ci=None
    )
    # g.set_yscale("log")
    # g.set_xscale("log")
    # plt.xlabel("Training Steps", fontsize=25)
    plt.xlabel("Epochs", fontsize=25)
    plt.xticks(fontsize=20)
    plt.ylabel("Test Loglikelihood", fontsize=25)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, markerscale=2, bbox_to_anchor=(1, 0.4), loc='center left')
    fig.tight_layout()
    fig.savefig(f"{basedir}/{resdir}/{metric}.png")
    fig.savefig(f"{basedir}/{resdir}/{metric}.pdf")
      
    print(f"Saved to {resdir}/{metric}.pdf")
    