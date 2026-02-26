import sys
import argparse
sys.path.append("../../")
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier

def main(n_jobs):
    all_datasets = {
        "NETSIM_5": "../../data/descriptors/descriptors_netsim_5.pkl",
        # ... Add other datasets ...
        "SYNTHETIC": "../../data/descriptors/descriptors_df_test.pkl",
    }
    
    print(f"--- Calculating Feature Importance (n_jobs={n_jobs}) ---")
    
    for name, path in all_datasets.items():
        try:
            df = pd.read_pickle(path)
            df.fillna(df.mean(numeric_only=True), inplace=True)
            
            X = df.drop(columns=["graph_id", "edge_source", "edge_dest", "is_causal"], errors='ignore')
            y = df["is_causal"]

            clf = BalancedRandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=n_jobs # PARAMETRIC
            )
            clf.fit(X, y)
            
            print(f"Dataset: {name}")
            importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print(importances.head(10))
            print("-" * 30)
        except FileNotFoundError:
            print(f"Skipping {name}, file not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=50)
    args = parser.parse_args()
    main(args.n_jobs)