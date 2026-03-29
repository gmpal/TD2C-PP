import sys
import argparse
sys.path.append("../../")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import precision_recall_curve, roc_curve
from imblearn.ensemble import BalancedRandomForestClassifier

def main(n_jobs):
    DESCRIPTORS_DIR = Path("../../data/descriptors/")
    TRAIN_DESCRIPTORS_FILE = "descriptors_df_train.pkl"

    print("=" * 60)
    print(f"--- Finding Robust Threshold (n_jobs={n_jobs}) ---")

    training_data_path = DESCRIPTORS_DIR / TRAIN_DESCRIPTORS_FILE
    descriptors_df_train = pd.read_pickle(training_data_path)
    descriptors_df_train.fillna(0, inplace=True)

    train_process_numbers = [1, 3, 5, 7, 9, 11, 13, 15, 19]

    def get_process_num(graph_id):
        process_idx = (graph_id % 1080) // 120
        return train_process_numbers[process_idx]

    descriptors_df_train["process_num"] = descriptors_df_train["graph_id"].apply(get_process_num)

    X = descriptors_df_train.drop(columns=["graph_id", "edge_source", "edge_dest", "is_causal", "process_num"])
    y = descriptors_df_train["is_causal"]
    groups = descriptors_df_train["process_num"]

    logo = LeaveOneGroupOut()
    thresholds_f1 = []
    thresholds_pr_breakeven = []
    thresholds_youden = []
    thresholds_droc = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
        print(f"Processing fold {fold+1}...")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        clf_fold = BalancedRandomForestClassifier(
            n_estimators=50,
            n_jobs=n_jobs,
            random_state=42,
            sampling_strategy="auto",
            replacement=True
        )
        clf_fold.fit(X_train_fold, y_train_fold)
        y_proba_val_fold = clf_fold.predict_proba(X_val_fold)[:, 1]

        # 1. Maximize F1-Score
        precision, recall, pr_thresh = precision_recall_curve(y_val_fold, y_proba_val_fold)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
        thresholds_f1.append(pr_thresh[np.argmax(f1_scores)])

        # 2. Precision-Recall Break-Even (where |precision - recall| is minimized)
        pr_diff = np.abs(precision[:-1] - recall[:-1])
        thresholds_pr_breakeven.append(pr_thresh[np.argmin(pr_diff)])

        # 3. Maximize Youden's J (TPR - FPR) via ROC curve
        fpr, tpr, roc_thresh = roc_curve(y_val_fold, y_proba_val_fold)
        youden_j = tpr - fpr
        thresholds_youden.append(roc_thresh[np.argmax(youden_j)])

        # 4. Minimize D_ROC(0,1) = sqrt(FPR^2 + (1-TPR)^2)
        d_roc = np.sqrt(fpr**2 + (1 - tpr)**2)
        thresholds_droc.append(roc_thresh[np.argmin(d_roc)])

    # Print Table 3
    print("\n" + "=" * 60)
    print("Table 3: Threshold Selection Metrics (LOGO-CV)")
    print("-" * 60)
    print(f"{'Optimization Metric':<30} {'Avg Threshold':>15} {'Std Dev':>12}")
    print("-" * 60)

    metrics = [
        ("Maximize F1-Score", thresholds_f1),
        ("Precision-Recall Break-Even", thresholds_pr_breakeven),
        ("Maximize Youden's J", thresholds_youden),
        ("Minimize D_ROC(0,1)", thresholds_droc),
    ]

    for name, values in metrics:
        avg = np.mean(values)
        std = np.std(values)
        print(f"{name:<30} {avg:>15.3f} {std:>12.3f}")

    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=50)
    args = parser.parse_args()
    main(args.n_jobs)
