import pandas as pd
import json
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--root', type=str, default="res", help='Root dir for results')
parser.add_argument('--exp', type=str, default="uci", help='Experiment to run')
args = parser.parse_args()
res_dir = f"{args.root}/{args.exp}"

datasets = [
    "boston_housing",
    "concrete",
    "energy",
    "kin8nm",
    "naval",
    "combined",
    "protein",
    "wine",
    "yacht",
]

df = pd.DataFrame(
    columns=[
        "RMSE SVGD",
        "RMSE S-SVGD",
        "RMSE GSVGD",
        # "RMSE GSVGDcano",
        "LL SVGD",
        "LL S-SVGD",
        "LL GSVGD",
        # "LL GSVGDcano",
        # "Spread SVGD",
        # "Spread S-SVGD",
        # "Spread GSVGD",
    ],
    index=datasets,
)
nparticles = 50
seeds = range(1, 11)

for method in ["SVGD", "S-SVGD", "GSVGD"]: #, "GSVGDcano"]:
    for dataset in datasets:
        try:
            RMSE = 0
            loglik = 0
            spread = 0
            RMSE_se = []
            loglik_se = []
            spread_se = []
            for seed in seeds:

                if method == "GSVGD":
                    with open(
                        # f"{res_dir}/{dataset}_{method}_nparticles{nparticles}_m5_M15_{seed}.json"
                        f"{res_dir}/{dataset}_{method}_nparticles{nparticles}_m10_M5_{seed}.json"
                    ) as f:
                        res_svgd = json.load(f)   
                elif method == "GSVGDcano":
                    with open(
                        f"{res_dir}/{dataset}_{method}_nparticles{nparticles}_m753_M1_{seed}.json"
                    ) as f:
                        res_svgd = json.load(f)   
                else:
                    with open(
                        # f"{res_dir}/{dataset}_{method}_nparticles{nparticles}_{seed}.json"
                        f"res/uci_tune/{dataset}_{method}_nparticles{nparticles}_{seed}.json"
                    ) as f:
                        res_svgd = json.load(f)
                RMSE += res_svgd["RMSE"]
                loglik += res_svgd["LL"]
                spread += res_svgd["spread"]
                RMSE_se.append(res_svgd["RMSE"])
                loglik_se.append(res_svgd["LL"])
                spread_se.append(res_svgd["spread"])
            RMSE_se = stats.sem(RMSE_se)
            loglik_se = stats.sem(loglik_se)
            spread_se = stats.sem(spread_se)
            RMSE /= len(seeds)
            loglik /= len(seeds)
            spread /= len(seeds)
            df.loc[dataset, f"RMSE {method}"] = f"{RMSE:.3g}±{RMSE_se:.3g}"
            df.loc[dataset, f"LL {method}"] = f"{loglik:.3g}±{loglik_se:.3g}"
            # df.loc[dataset, f"Spread {method}"] = f"{spread:.3g}±{spread_se:.3g}"
        except:
            print(f"Experiment not completed for {method} on {dataset}")
df.columns = [
    "RMSE SVGD",
    "RMSE S-SVGD",
    "RMSE GSVGD",
    # "RMSE GSVGDcano",
    "LL SVGD",
    "LL S-SVGD",
    "LL GSVGD",
    # "LL GSVGDcano",
    # "Spread SVGD",
    # "Spread S-SVGD",
    # "Spread GSVGD",
]
df.to_csv(f"{res_dir}/0results.csv", index=False)
df.to_latex(f"{res_dir}/0results.tex")