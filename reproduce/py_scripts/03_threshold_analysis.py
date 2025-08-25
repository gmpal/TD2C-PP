# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("./")

# %%
# find_robust_threshold_ALL_METRICS.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import precision_recall_curve, roc_curve
from imblearn.ensemble import BalancedRandomForestClassifier
import os

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
DESCRIPTORS_DIR = Path("data/descriptors/")
TRAIN_DESCRIPTORS_FILE = "descriptors_df_train.pkl"
N_JOBS = 50

# ==============================================================================
# --- 2. DATA LOADING AND PREPARATION ---
# ==============================================================================
print("=" * 60)
print("--- Finding Robust Threshold via Leave-One-Process-Out CV ---")
print("--- Calculating for FOUR different optimization metrics ---")

# ... (same data loading and group assignment code as before) ...
training_data_path = DESCRIPTORS_DIR / TRAIN_DESCRIPTORS_FILE
descriptors_df_train = pd.read_pickle(training_data_path)
descriptors_df_train.fillna(0, inplace=True)
train_process_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 19]


def get_process_num(graph_id):
    process_idx = (graph_id % 1080) // 120
    return train_process_numbers[process_idx]


descriptors_df_train["process_num"] = descriptors_df_train["graph_id"].apply(
    get_process_num
)
print(
    f"Verified {len(descriptors_df_train['process_num'].unique())} unique process groups."
)
X = descriptors_df_train.drop(
    columns=["graph_id", "edge_source", "edge_dest", "is_causal", "process_num"]
)
y = descriptors_df_train["is_causal"]
groups = descriptors_df_train["process_num"]

# ==============================================================================
# --- 3. LEAVE-ONE-GROUP-OUT (PROCESS) CROSS-VALIDATION ---
# ==============================================================================
logo = LeaveOneGroupOut()
# Initialize lists to store thresholds for each metric
thresholds_f1, thresholds_be, thresholds_j, thresholds_dist = [], [], [], []

print(
    f"\nStarting Leave-One-Group-Out CV across {logo.get_n_splits(groups=groups)} processes..."
)

for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
    held_out_process = groups.iloc[val_idx].unique()[0]
    print("-" * 40)
    print(f"Fold {fold+1}: Holding out process number '{held_out_process}'")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    print(f"  - Training classifier...")
    clf_fold = BalancedRandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        random_state=42,
        sampling_strategy="auto",
        replacement=True,
        bootstrap=True,
        n_jobs=N_JOBS,
    )
    clf_fold.fit(X_train_fold, y_train_fold)

    print(f"  - Predicting probabilities and finding optimal thresholds...")
    y_proba_val_fold = clf_fold.predict_proba(X_val_fold)[:, 1]

    # --- METRIC 1: Max F1-Score ---
    precision, recall, pr_thresh = precision_recall_curve(y_val_fold, y_proba_val_fold)
    f1_scores = (
        2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    )
    thresholds_f1.append(pr_thresh[np.argmax(f1_scores)])

    # --- METRIC 2: Break-Even Point (P≈R) ---
    diffs = np.abs(precision[:-1] - recall[:-1])
    thresholds_be.append(pr_thresh[np.argmin(diffs)])

    # --- METRICS 3 & 4 (from ROC curve) ---
    fpr, tpr, roc_thresh = roc_curve(y_val_fold, y_proba_val_fold)

    # --- METRIC 3: Youden's J Statistic ---
    thresholds_j.append(roc_thresh[np.argmax(tpr - fpr)])

    # --- METRIC 4: Closest to (0,1) ---
    distances = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    thresholds_dist.append(roc_thresh[np.argmin(distances)])

    print(
        f"  - Thresholds found: F1={thresholds_f1[-1]:.3f}, BEP={thresholds_be[-1]:.3f}, J={thresholds_j[-1]:.3f}, Dist={thresholds_dist[-1]:.3f}"
    )

# ==============================================================================
# --- 4. FINAL RESULTS AND COMPARISON ---
# ==============================================================================
results = {
    "Metric": ["Max F1-Score", "Break-Even (P≈R)", "Youden's J", "Closest to (0,1)"],
    "Avg. Threshold": [
        np.mean(thresholds_f1),
        np.mean(thresholds_be),
        np.mean(thresholds_j),
        np.mean(thresholds_dist),
    ],
    "Std. Dev.": [
        np.std(thresholds_f1),
        np.std(thresholds_be),
        np.std(thresholds_j),
        np.std(thresholds_dist),
    ],
}
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("--- Cross-Validation Complete: Comparison of Metrics ---")
print(results_df.to_string(index=False))
print("=" * 60)

# --- Interpretation and Recommendation ---
print("\n--- Interpretation ---")
print(
    "This table shows the average 'optimal' threshold according to four different definitions of optimality."
)
print(
    "A low standard deviation for a given metric suggests its result is stable across different processes."
)
if results_df["Avg. Threshold"].std() < 0.05:
    print(
        "The average thresholds are all very similar, indicating the choice of metric is not critical."
    )
else:
    print(
        "The average thresholds differ, indicating your choice of 'optimality' significantly impacts the result."
    )

print("\n--- Recommendation ---")
print(
    "1. For your final paper, choose ONE metric and justify it. Maximizing the F1-Score is the most standard and defensible choice."
)
print(
    "2. You can use this table in an appendix or 'sensitivity analysis' section to show that you explored other options."
)
print(
    "3. If another metric (e.g., Youden's J) gives you a threshold that you believe is more practical for your problem, you can use it, but you must clearly state why you chose it over the F1-score."
)
