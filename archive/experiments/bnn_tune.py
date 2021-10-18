import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

datasets = [
    # "boston_housing",
    # "concrete",
    "energy",
    # "kin8nm",
    # "naval",
    # "protein",
    # "wine",
    # "yacht",
]

df = pd.DataFrame(
    columns=[
        "RMSE SVGD",
        "RMSE MaxSVGD",
        "RMSE GSVGD",
        "LL SVGD",
        "LL MaxSVGD",
        "LL GSVGD",
    ],
    index=[
        "boston_housing",
        "concrete",
        "energy",
        "kin8nm",
        "naval",
        "protein",
        "wine",
        "yacht",
    ],
)

lr_list = [0.1, 0.01, 0.001]
delta_list = [0.1, 0.01, 0.001]
nparticles = 20
df_list = []

for method in ["GSVGD"]:
    for dataset in datasets:
        # try:
        RMSE = 0
        loglik = 0
        # for seed in range(1, 11):
        for seed in [2]:
            for lr in lr_list:
                for delta in delta_list:
                    print(f"res/uci/{dataset}_{method}_nparticles{nparticles}_m10_M2_lr{lr}_delta{delta}_{seed}.json")
                    if method == "GSVGD":
                        with open(
                            f"res/uci/seq_{dataset}_{method}_nparticles{nparticles}_m10_M2_lr{lr}_delta{delta}_{seed}.json"
                        ) as f:
                            res_svgd = json.load(f)    
                    else:
                        with open(
                            f"res/uci/{dataset}_{method}_nparticles{nparticles}_{seed}.json"
                        ) as f:
                            res_svgd = json.load(f)
                    print(res_svgd["RMSE"])
                    new_df = pd.DataFrame(
                        {"lr": [lr], "delta": [delta], "RMSE": [res_svgd["RMSE"]], "LL": [res_svgd["LL"]], "spread": [res_svgd["spread"]]}
                    )
                    df_list.append(new_df)

        # except:
        #     print(f"Experiment not completed for {method} on {dataset}")

df = pd.concat(df_list)
df.to_csv("res/uci/results.csv", index=False)
df.to_latex("res/uci/results.tex")

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
g = sns.heatmap(
    df[["lr", "delta", "RMSE"]].pivot("lr", "delta", "RMSE"), 
    annot=True,
    fmt=".4g"
)
g.invert_yaxis()
plt.title("RMSE")
g
plt.subplot(1, 3, 2)
g = sns.heatmap(
    df[["lr", "delta", "LL"]].pivot("lr", "delta", "LL"), 
    annot=True,
    fmt=".4g"
)
g.invert_yaxis()
plt.title("LL")
g
plt.subplot(1, 3, 3)
g = sns.heatmap(
    df[["lr", "delta", "spread"]].pivot("lr", "delta", "spread"), 
    annot=True,
    fmt=".4g"
)
g.invert_yaxis()
plt.title("spread")
g
fig.savefig(f"res/uci/energy_tune.png")