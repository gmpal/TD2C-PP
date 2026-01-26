import sys
import argparse
import os
import time
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
sys.path.append("../../")
from src.td2c.data_generation.builder import TSBuilder
from src.td2c.descriptors import DataLoader
from src.td2c.benchmark import VAR, VARLiNGAM, PCMCI, Granger, DYNOTEARS, MultivariateGranger, D2CWrapper

# Set env vars to 1 thread for fair comparison before importing libraries that use MKL
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def main(n_jobs_high):
    N_VARS_LIST = [3, 5, 10, 15, 20, 25]
    MAXLAGS = 3
    N_JOBS_D2C_LOW = 1
    # Use the passed argument for high jobs
    N_JOBS_D2C_HIGH = n_jobs_high 
    TRAINING_DESCRIPTORS_PATH = "data/descriptors/descriptors_df_train.pkl" # Check path
    RESULTS_SAVE_PATH = "data/benchmark_times_by_nvars.csv"

    # Load Model
    print(f"Loading training data from {TRAINING_DESCRIPTORS_PATH}...")
    descriptors_df_train = pd.read_pickle(TRAINING_DESCRIPTORS_PATH)
    X_train = descriptors_df_train.drop(columns=["graph_id", "edge_source", "edge_dest", "is_causal"])
    y_train = descriptors_df_train["is_causal"]
    
    clf = BalancedRandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
    clf.fit(X_train, y_train)

    results_list = []
    benchmark_methods = {
        "VAR": VAR, "VARLiNGAM": VARLiNGAM, "Granger": Granger,
        "DYNOTEARS": DYNOTEARS, "MultivariateGranger": MultivariateGranger
    }

    for n_vars in N_VARS_LIST:
        print(f"Benchmarking: {n_vars} variables")
        tsbuilder = TSBuilder(
            observations_per_time_series=250, maxlags=MAXLAGS, n_variables=n_vars,
            time_series_per_process=1, processes_to_use=[2], verbose=False
        )
        tsbuilder.build()
        dataloader = DataLoader(n_variables=n_vars, maxlags=MAXLAGS)
        dataloader.from_tsbuilder(tsbuilder)
        
        orig_obs = dataloader.get_original_observations()
        lagged_obs = dataloader.get_observations()

        # Run Standard Benchmarks (Single Threaded for fair baseline)
        for name, method in benchmark_methods.items():
            start = time.time()
            try:
                method(ts_list=orig_obs, maxlags=MAXLAGS, n_jobs=1).run()
                results_list.append({"n_variables": n_vars, "method": name, "execution_time": time.time() - start})
            except Exception as e: print(e)

        # Run PCMCI
        start = time.time()
        PCMCI(ts_list=orig_obs, maxlags=MAXLAGS, n_jobs=1, ci="ParCorr").run()
        results_list.append({"n_variables": n_vars, "method": "PCMCI (ParCorr)", "execution_time": time.time() - start})

        # Run D2C Low
        start = time.time()
        D2CWrapper(ts_list=lagged_obs, model=clf, n_variables=n_vars, maxlags=MAXLAGS, n_jobs=N_JOBS_D2C_LOW, manages_own_parallelism=True).run()
        results_list.append({"n_variables": n_vars, "method": f"TD2C ({N_JOBS_D2C_LOW} job)", "execution_time": time.time() - start})

        # Run D2C High
        start = time.time()
        D2CWrapper(ts_list=lagged_obs, model=clf, n_variables=n_vars, maxlags=MAXLAGS, n_jobs=N_JOBS_D2C_HIGH, manages_own_parallelism=True).run()
        results_list.append({"n_variables": n_vars, "method": f"TD2C ({N_JOBS_D2C_HIGH} jobs)", "execution_time": time.time() - start})

    pd.DataFrame(results_list).to_csv(RESULTS_SAVE_PATH, index=False)
    print("Benchmark complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=50)
    args = parser.parse_args()
    main(args.n_jobs)