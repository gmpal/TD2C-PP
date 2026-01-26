# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../../")

# %%
import pickle
import pandas as pd
import numpy as np
import os

# Imports for plotting and statistical tests
import matplotlib

matplotlib.use("agg")  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import operator
import math
from scipy.stats import wilcoxon, friedmanchisquare
import networkx

# Imports for metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
)

# =============================================================================
# 1. SETUP AND CONFIGURATION
# =============================================================================
from src.td2c.benchmark.utils import draw_cd_diagram

THRESHOLD = 0.309

OUTPUT_PATH_CD = "CD_PLOTS/"
os.makedirs(OUTPUT_PATH_CD, exist_ok=True)

ORDER_SAVING_RESULTS = [
    "var",
    "varlingam",
    "pcmci",
    "mvgc",
    "pcmci_gpdc",
    "granger",
    "dynotears",
    "td2c",
]

DISPLAY_NAMES = {
    "var": "VAR",
    "varlingam": "VARLINGAM",
    "pcmci": "PCMCI",
    "mvgc": "MVGC",
    "pcmci_gpdc": "PCMCI-GPDC",
    "granger": "GRANGER",
    "dynotears": "DYNOTEARS",
    "td2c": "TD2C",
}

plt.rcParams["font.family"] = "DejaVu Sans"  # Use available font


datasets_name = ["TEST"]
for dataset_name in datasets_name:
    file_path = f"data/causal_dfs/causal_dfs_{dataset_name}.pkl"
    print(f"--- Creating Original CD Plot for {dataset_name} ---")

    # --- Load Data ---
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    method_results_tuple = loaded_data[:-1]
    true_causal_dfs_dict = loaded_data[-1]

    # --- Calculate Precision scores for each run using pre-computed predictions ---
    run_ids = sorted(true_causal_dfs_dict.keys())

    metrics_function_dict = {
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
    }

    for metric, metric_function in metrics_function_dict.items():
        per_run_scores = []
        for run_id in run_ids:
            y_true_run = true_causal_dfs_dict[run_id]["is_causal"].astype(int)

            for i, method_dfs_dict in enumerate(method_results_tuple):
                internal_name = ORDER_SAVING_RESULTS[i]

                if method_dfs_dict is None or run_id not in method_dfs_dict:
                    continue

                if internal_name == "td2c":
                    y_proba_run = method_dfs_dict[run_id]["probability"].astype(float)
                    y_pred_run = (y_proba_run > THRESHOLD).astype(int)
                else:
                    y_pred_run = method_dfs_dict[run_id]["is_causal"].astype(int)

                if metric == "precision" or metric == "recall" or metric == "f1":
                    score = metric_function(
                        y_true_run, y_pred_run, zero_division=np.nan
                    )
                elif metric == "balanced_accuracy" or metric == "accuracy":
                    score = metric_function(y_true_run, y_pred_run)

                per_run_scores.append(
                    {
                        "Model": DISPLAY_NAMES[internal_name],
                        "dataset_name": f"{dataset_name}_{run_id}",
                        "Score": score,  # The score for this run
                    }
                )

        # --- Create a DataFrame suitable for the CD plot function ---
        scores_df = pd.DataFrame(per_run_scores)

        # --- Generate the CD plot for Precision ---
        print("\nGenerating Precision CD plot...")
        output_path = f"{OUTPUT_PATH_CD}/cd_{dataset_name}_{metric}.png"
        draw_cd_diagram(
            path=output_path,
            df_perf=scores_df,
        )

    print("\nScript finished successfully.")

# %%
