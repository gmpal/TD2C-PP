import sys
import argparse
import os
import pickle
import pandas as pd
from pathlib import Path
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score

# Environment Setup
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.append("../../")
sys.path.append("../src")

# Imports from td2c
from td2c.descriptors import DataLoader
from td2c.benchmark import (
    VARLiNGAM, PCMCI, Granger, DYNOTEARS, D2CWrapper, VAR, MultivariateGranger
)
from td2c.benchmark.utils import prepare_prediction_df_d2c
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def _load_testing_data():
    # ... (Keep original logic of _load_testing_data here) ...
    dataloaders = {}
    original_observations_testing = {}
    true_causal_dfs = {}

    for error_dist in ["gaussian", "uniform", "laplace"]:
        dataloader = DataLoader(n_variables=5, maxlags=3)
        dataloader.from_pickle(f"data/observations/testing_data_{error_dist}.pkl")
        dataloaders[error_dist] = dataloader
        original_observations_testing[error_dist] = dataloader.get_original_observations()
        true_causal_dfs[error_dist] = dataloader.get_true_causal_dfs()

    original_observations_list_testing = []
    for obs_list in original_observations_testing.values():
        original_observations_list_testing.extend(obs_list)

    true_causal_dfs_list_testing = []
    for causal_df in true_causal_dfs.values():
        true_causal_dfs_list_testing.extend(causal_df)

    return original_observations_list_testing, true_causal_dfs_list_testing

def main(n_jobs):
    MAXLAGS = 3
    THRESHOLD = 0.309
    DESCRIPTORS_DIR = Path("data/descriptors/")
    PRE_RESULTS_DIR = Path("data/before_d2c/")
    RESULTS_DIR = Path("data/causal_dfs/")

    PRE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    DATASETS_TO_PROCESS = [
        {"name": "DREAM3_10", "n_vars": 10, "input_file": "data/realistic/dream3/dream3_10.pkl", "d2c_descriptors_file": "descriptors_dream3_10.pkl"},
        {"name": "DREAM3_50", "n_vars": 50, "input_file": "data/realistic/dream3/dream3_50.pkl", "d2c_descriptors_file": "descriptors_dream3_50.pkl"},
        {"name": "NETSIM_5", "n_vars": 5, "input_file": "data/realistic/netsym/netsym_5.pkl", "d2c_descriptors_file": "descriptors_netsim_5.pkl"},
        {"name": "NETSIM_10", "n_vars": 10, "input_file": "data/realistic/netsym/netsym_10.pkl", "d2c_descriptors_file": "descriptors_netsim_10.pkl"},
        {"name": "TEST", "n_vars": 5, "input_file": None, "d2c_descriptors_file": "descriptors_df_test.pkl"},
    ]

    # --- Train Classifier ---
    print(f"--- Training D2C Classifier (n_jobs={n_jobs}) ---")
    training_data_path = DESCRIPTORS_DIR / "descriptors_df_train.pkl"
    descriptors_df_train = pd.read_pickle(training_data_path)
    descriptors_df_train.fillna(0, inplace=True)
    X_train = descriptors_df_train.drop(columns=["graph_id", "edge_source", "edge_dest", "is_causal"])
    y_train = descriptors_df_train["is_causal"]

    clf = BalancedRandomForestClassifier(
        n_estimators=500, n_jobs=n_jobs, random_state=42, 
        sampling_strategy="auto", replacement=True
    )
    clf.fit(X_train, y_train)

    # --- Benchmark Loop ---
    for config in DATASETS_TO_PROCESS:
        dataset_name = config["name"]
        print(f"\n--- PROCESSING DATASET: {dataset_name} ---")

        if dataset_name == "TEST":
            original_observations_testing, true_causal_dfs = _load_testing_data()
        else:
            dataloader = DataLoader(n_variables=config["n_vars"], maxlags=MAXLAGS)
            dataloader.from_pickle(config["input_file"])
            original_observations_testing = dataloader.get_original_observations()
            true_causal_dfs = dataloader.get_true_causal_dfs()

        # Run Competitors
        competitors_cache_path = PRE_RESULTS_DIR / f"causal_dfs_before_d2c_{dataset_name}.pkl"
        
        if competitors_cache_path.exists():
            print("Loading cached competitor results.")
            with open(competitors_cache_path, "rb") as f:
                all_competitors = pickle.load(f)
            # Unpack...
            causal_dfs_var = all_competitors["causal_dfs_var"]
            causal_dfs_varlingam = all_competitors["causal_dfs_varlingam"]
            causal_dfs_pcmci = all_competitors["causal_dfs_pcmci"]
            causal_dfs_pcmci_gpdc = all_competitors["causal_dfs_pcmci_gpdc"]
            causal_dfs_granger = all_competitors["causal_dfs_granger"]
            causal_dfs_mvgc = all_competitors["causal_dfs_mvgc"]
            causal_dfs_dynotears = all_competitors["causal_dfs_dynotears"]
        else:
            print("Running competitors...")
            # Instantiate and run
            var = VAR(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            var.run()
            varlingam = VARLiNGAM(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            varlingam.run()
            pcmci = PCMCI(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            pcmci.run()
            pcmci_gpdc = PCMCI(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs, ci="GPDC")
            pcmci_gpdc.run()
            granger = Granger(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            granger.run()
            dynotears = DYNOTEARS(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            dynotears.run()
            mvgc = MultivariateGranger(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=n_jobs)
            mvgc.run()

            # Collect results
            causal_dfs_var = var.get_causal_dfs()
            causal_dfs_varlingam = varlingam.get_causal_dfs()
            causal_dfs_pcmci = pcmci.get_causal_dfs()
            causal_dfs_pcmci_gpdc = pcmci_gpdc.get_causal_dfs()
            causal_dfs_granger = granger.get_causal_dfs()
            causal_dfs_mvgc = mvgc.get_causal_dfs()
            causal_dfs_dynotears = dynotears.get_causal_dfs()
            
            # Cache
            all_to_cache = {
                "causal_dfs_var": causal_dfs_var,
                "causal_dfs_varlingam": causal_dfs_varlingam,
                "causal_dfs_pcmci": causal_dfs_pcmci,
                "causal_dfs_pcmci_gpdc": causal_dfs_pcmci_gpdc,
                "causal_dfs_granger": causal_dfs_granger,
                "causal_dfs_mvgc": causal_dfs_mvgc,
                "causal_dfs_dynotears": causal_dfs_dynotears,
            }
            with open(competitors_cache_path, "wb") as f:
                pickle.dump(all_to_cache, f)

        # --- Run D2C ---
        print(f"Running D2C for {dataset_name}...")
        d2c_descriptors_path = DESCRIPTORS_DIR / config["d2c_descriptors_file"]
        d2c_args = {
            "ts_list": original_observations_testing,
            "model": clf,
            "threshold": THRESHOLD,
            "n_variables": config["n_vars"],
            "maxlags": MAXLAGS,
            "mb_estimator": "ts",
            "manages_own_parallelism": True,
        }

        if d2c_descriptors_path.exists():
            d2c_args["precomputed_descriptors_path"] = str(d2c_descriptors_path)
            d2c_args["n_jobs"] = 1 # Already computed
        else:
            d2c_args["n_jobs"] = n_jobs

        d2cwrapper = D2CWrapper(**d2c_args)
        d2cwrapper.run()
        causal_dfs_d2c = d2cwrapper.get_causal_dfs()

        # Save Final Results
        def sort_dict_by_key(d): return {k: d[k] for k in sorted(d)} if isinstance(d, dict) else d
        true_causal_dfs_cleaned = {k: v for k, v in enumerate(true_causal_dfs) if v is not None}
        
        final_results = (
            sort_dict_by_key(causal_dfs_var),
            sort_dict_by_key(causal_dfs_varlingam),
            sort_dict_by_key(causal_dfs_pcmci),
            sort_dict_by_key(causal_dfs_mvgc),
            sort_dict_by_key(causal_dfs_pcmci_gpdc),
            sort_dict_by_key(causal_dfs_granger),
            sort_dict_by_key(causal_dfs_dynotears),
            sort_dict_by_key(causal_dfs_d2c),
            true_causal_dfs_cleaned,
        )

        with open(RESULTS_DIR / f"causal_dfs_{dataset_name}.pkl", "wb") as f:
            pickle.dump(final_results, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=50)
    args = parser.parse_args()
    main(args.n_jobs)