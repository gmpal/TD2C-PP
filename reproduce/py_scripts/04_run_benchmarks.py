# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("./")

# %%
import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from imblearn.ensemble import BalancedRandomForestClassifier

# --- Environment Setup for Performance ---
# Limit threads to prevent over-subscription, especially by libraries like MKL used in VARLiNGAM
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Add Project Source to Path ---
# Assumes the script is run from a directory where '../src' is the correct path
sys.path.append("../src")

# --- Import Custom Libraries ---
from d2c.descriptors import DataLoader
from d2c.benchmark import (
    VARLiNGAM,
    PCMCI,
    Granger,
    DYNOTEARS,
    D2CWrapper,
    VAR,
    MultivariateGranger,
)

# --- Suppress Warnings ---
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from d2c.benchmark.utils import prepare_prediction_df_d2c

from sklearn.metrics import roc_auc_score


# %%
def _load_testing_data():
    """
    !!! Special Helper Function for testing data loading.
    !!! It requires special treatment because it's split into 3 files because of the noise distributions.
    Loads testing data from pickle files for different error distributions.

    This function initializes data loaders for three types of error distributions:
    'gaussian', 'uniform', and 'laplace'. It retrieves original observations,
    lagged flattened observations, flattened directed acyclic graphs (DAGs),
    and true causal dataframes from the data loaders. The function aggregates
    these observations and returns them as lists.

    Returns:
        tuple: A tuple containing:
            - original_observations_list_testing (list): A list of original observations from the testing data.
            - true_causal_dfs_list_testing (list): A list of true causal dataframes from the testing data.
    """
    dataloaders = {}
    original_observations_testing = {}
    lagged_flattened_observations_testing = {}
    flattened_dags_testing = {}
    true_causal_dfs = {}

    for error_dist in ["gaussian", "uniform", "laplace"]:
        dataloader = DataLoader(n_variables=5, maxlags=3)
        dataloader.from_pickle(f"data/observations/testing_data_{error_dist}.pkl")

        dataloaders[error_dist] = dataloader
        original_observations_testing[error_dist] = (
            dataloader.get_original_observations()
        )
        lagged_flattened_observations_testing[error_dist] = (
            dataloader.get_observations()
        )
        flattened_dags_testing[error_dist] = dataloader.get_dags()
        true_causal_dfs[error_dist] = dataloader.get_true_causal_dfs()

    original_observations_list_testing = []
    for obs_list in original_observations_testing.values():
        original_observations_list_testing.extend(obs_list)

    lagged_flattened_observations_list_testing = []
    for obs_list in lagged_flattened_observations_testing.values():
        lagged_flattened_observations_list_testing.extend(obs_list)

    flattened_dags_list_testing = []
    for dags_list in flattened_dags_testing.values():
        flattened_dags_list_testing.extend(dags_list)

    true_causal_dfs_list_testing = []
    for causal_df in true_causal_dfs.values():
        true_causal_dfs_list_testing.extend(causal_df)

    return original_observations_list_testing, true_causal_dfs_list_testing


# %%
# ==============================================================================
# --- 1. SCRIPT CONFIGURATION ---
# ==============================================================================
N_JOBS = 50  # Max jobs for parallelizable tasks
MAXLAGS = 3  # Max lags for all models
DESCRIPTORS_DIR = Path("data/descriptors/")
PRE_RESULTS_DIR = Path("data/before_d2c/")
RESULTS_DIR = Path("data/causal_dfs/")
THRESHOLD = 0.309

# Ensure directories exist
PRE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define all datasets to be processed in a list of dictionaries
DATASETS_TO_PROCESS = [
    {
        "name": "DREAM3_10",
        "n_vars": 10,
        "input_file": "data/realistic/dream3/dream3_10.pkl",
        "d2c_descriptors_file": "descriptors_dream3_10.pkl",
    },
    {
        "name": "DREAM3_50",
        "n_vars": 50,
        "input_file": "data/realistic/dream3/dream3_50.pkl",
        "d2c_descriptors_file": "descriptors_dream3_50.pkl",
    },
    {
        "name": "NETSIM_5",
        "n_vars": 5,
        "input_file": "data/realistic/netsym/netsym_5.pkl",
        "d2c_descriptors_file": "descriptors_netsim_5.pkl",
    },
    {
        "name": "NETSIM_10",
        "n_vars": 10,
        "input_file": "data/realistic/netsym/netsym_10.pkl",
        "d2c_descriptors_file": "descriptors_netsim_10.pkl",
    },
    {
        "name": "TEST",
        "n_vars": 5,
        "input_file": None,  # requires special treatment
        "d2c_descriptors_file": "descriptors_df_test.pkl",
    },
]

# %%
# ==============================================================================
# --- 2. TRAIN THE TD2C Classifier (ONCE) ---
# ==============================================================================
print("=" * 60)
print("--- Training D2C Classifier (once for all benchmarks) ---")

# Use the 'no_copula' version of the training data as per our decision
training_data_path = DESCRIPTORS_DIR / "descriptors_df_train.pkl"
print(f"Loading training data from: {training_data_path}")

try:
    descriptors_df_train = pd.read_pickle(training_data_path)
    # Using 0 imputation for cases when MB members don't exist (childless/parentless nodes)
    descriptors_df_train.fillna(0, inplace=True)

    X_train = descriptors_df_train.drop(
        columns=["graph_id", "edge_source", "edge_dest", "is_causal"]
    )
    y_train = descriptors_df_train["is_causal"]

    # Define the classifier
    clf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        sampling_strategy="auto",
        replacement=True,
        bootstrap=True,
        n_jobs=N_JOBS,
    )

    print("Fitting the classifier...")
    clf.fit(X_train, y_train)
    print("Classifier training complete.")

except FileNotFoundError:
    print(
        f"FATAL ERROR: Training file not found at {training_data_path}. Cannot proceed."
    )
    sys.exit(1)


# %%
# ==============================================================================
# --- 3. MAIN BENCHMARKING LOOP ---
# ==============================================================================
for config in DATASETS_TO_PROCESS:
    dataset_name = config["name"]
    n_vars = config["n_vars"]

    print("\n" + "=" * 60)
    print(f"--- PROCESSING DATASET: {dataset_name} ---")
    print(f"Variables: {n_vars}, Max Lags: {MAXLAGS}")

    if dataset_name == "TEST":
        # special loading helper: it's split in multiple files
        # one per kind of error, and we need to make sure the loading
        # order and merging are exactly the same as when we have computed
        # descriptors (notebook 01_descriptors_computation.ipynb)
        original_observations_testing, true_causal_dfs = _load_testing_data()
    else:
        dataloader = DataLoader(n_variables=n_vars, maxlags=MAXLAGS)
        dataloader.from_pickle(config["input_file"])
        original_observations_testing = dataloader.get_original_observations()
        true_causal_dfs = dataloader.get_true_causal_dfs()

    # --- Run Competitors (or load from cache) ---
    competitors_cache_path = (
        PRE_RESULTS_DIR / f"causal_dfs_before_d2c_{dataset_name}.pkl"
    )

    if competitors_cache_path.exists():
        print(
            f"Found cached competitor results. Loading from: {competitors_cache_path}"
        )
        with open(competitors_cache_path, "rb") as f:
            all_competitors = pickle.load(f)
        causal_dfs_var = all_competitors["causal_dfs_var"]
        causal_dfs_varlingam = all_competitors["causal_dfs_varlingam"]
        causal_dfs_pcmci = all_competitors["causal_dfs_pcmci"]
        causal_dfs_pcmci_gpdc = all_competitors["causal_dfs_pcmci_gpdc"]
        causal_dfs_granger = all_competitors["causal_dfs_granger"]
        causal_dfs_mvgc = all_competitors["causal_dfs_mvgc"]
        causal_dfs_dynotears = all_competitors["causal_dfs_dynotears"]
    else:
        print(
            f"No cached results found. Running all competitor benchmarks for {dataset_name}..."
        )

        # Instantiate and run all competitor models
        var = VAR(ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS)
        var.run()
        varlingam = VARLiNGAM(
            ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS
        )
        varlingam.run()
        pcmci = PCMCI(
            ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS
        )
        pcmci.run()
        pcmci_gpdc = PCMCI(
            ts_list=original_observations_testing,
            maxlags=MAXLAGS,
            n_jobs=N_JOBS,
            ci="GPDC",
        )
        pcmci_gpdc.run()
        granger = Granger(
            ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS
        )
        granger.run()
        dynotears = DYNOTEARS(
            ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS
        )
        dynotears.run()
        mvgc = MultivariateGranger(
            ts_list=original_observations_testing, maxlags=MAXLAGS, n_jobs=N_JOBS
        )
        mvgc.run()

        # Get results
        causal_dfs_var = var.get_causal_dfs()
        causal_dfs_varlingam = varlingam.get_causal_dfs()
        causal_dfs_pcmci = pcmci.get_causal_dfs()
        causal_dfs_pcmci_gpdc = pcmci_gpdc.get_causal_dfs()
        causal_dfs_granger = granger.get_causal_dfs()
        causal_dfs_mvgc = mvgc.get_causal_dfs()
        causal_dfs_dynotears = dynotears.get_causal_dfs()

        # Save to cache for next time
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
        print(f"Competitor results saved to cache: {competitors_cache_path}")

    # --- Run D2C ---
    print(f"Running D2C for {dataset_name}...")
    d2c_descriptors_path = DESCRIPTORS_DIR / config["d2c_descriptors_file"]
    d2c_args = {
        "ts_list": original_observations_testing,
        "model": clf,
        "threshold": THRESHOLD,
        "n_variables": n_vars,
        "maxlags": MAXLAGS,
        "mb_estimator": "ts",
        "manages_own_parallelism": True,  # False to paralellelize per observation, True to parallelize per couple
    }

    # Conditionally add the precomputed descriptors path
    if d2c_descriptors_path.exists():
        print(f"  - Using precomputed descriptors from: {d2c_descriptors_path}")
        d2c_args["precomputed_descriptors_path"] = str(d2c_descriptors_path)
        d2c_args["n_jobs"] = 1
    else:
        print(
            f"  - WARNING: Precomputed descriptors not found at {d2c_descriptors_path}."
        )
        print("  - D2C will compute descriptors on the fly. This may be slow.")
        d2c_args["n_jobs"] = N_JOBS

    test = pd.read_pickle(d2c_args["precomputed_descriptors_path"])
    test.fillna(0, inplace=True)
    X_test = test.drop(
        columns=["graph_id", "edge_source", "edge_dest", "is_causal"], errors="ignore"
    )
    y_test = test["is_causal"]
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    print(f"ROC AUC BEFORE: {roc_auc_score(y_test, y_pred_proba)}")

    d2cwrapper = D2CWrapper(**d2c_args)
    d2cwrapper.run()
    causal_dfs_d2c = d2cwrapper.get_causal_dfs()
    print("D2C run complete.")

    # --- Save Final Results ---
    # Helper to sort dictionaries by key for consistent ordering
    def sort_dict_by_key(d):
        return {k: d[k] for k in sorted(d)} if isinstance(d, dict) else d

    final_output_path = RESULTS_DIR / f"causal_dfs_{dataset_name}.pkl"

    # Add key to true_causal_dfs for uniformity
    true_causal_dfs_cleaned = {
        k: v for k, v in enumerate(true_causal_dfs) if v is not None
    }

    # Collect all results into a tuple
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

    prediction_truth_df = prepare_prediction_df_d2c(
        causal_dfs_d2c, true_causal_dfs_cleaned
    )

    # ADD THIS DEBUG LINE:
    print(
        f"NaNs found in the final y_true vector: {prediction_truth_df.y_true.isna().sum()}"
    )

    print(
        "ROC AUC AFTER",
        roc_auc_score(prediction_truth_df.y_true, prediction_truth_df.y_pred),
    )

    with open(final_output_path, "wb") as f:
        pickle.dump(final_results, f)

    print(
        f"Successfully saved all final results for {dataset_name} to: {final_output_path}"
    )

print("\n" + "=" * 60)
print("--- All benchmark runs completed successfully! ---")
