import sys
import argparse
sys.path.append("../../")
from src.td2c.descriptors import D2C, DataLoader

def compute_descriptors(data_type, input_prefix, output_file, n_vars, maxlags, n_jobs):
    print(f"--- Computing {data_type} descriptors ({output_file}) ---")
    
    if data_type == "synthetic_split":
        # Handle the special case of split files for synthetic data
        dataloaders = {}
        original_observations = {}
        lagged_observations = {}
        flattened_dags = {}
        
        for error_dist in ["gaussian", "uniform", "laplace"]:
            dl = DataLoader(n_variables=n_vars, maxlags=maxlags)
            dl.from_pickle(f"../../data/{input_prefix}_{error_dist}.pkl")
            dataloaders[error_dist] = dl
            lagged_observations[error_dist] = dl.get_observations()
            flattened_dags[error_dist] = dl.get_dags()

        # Merge lists
        obs_list = []
        for obs in lagged_observations.values(): obs_list.extend(obs)
        dags_list = []
        for dags in flattened_dags.values(): dags_list.extend(dags)
        
        final_obs = obs_list
        final_dags = dags_list

    else:
        # Standard single file loading
        dl = DataLoader(n_variables=n_vars, maxlags=maxlags)
        dl.from_pickle(f"realdata/{input_prefix}.pkl" if "realdata" in input_prefix else input_prefix)
        final_obs = dl.get_observations()
        final_dags = dl.get_dags()

    # Compute
    d2c = D2C(
        observations=final_obs,
        dags=final_dags,
        couples_to_consider_per_dag=-1,
        n_variables=n_vars,
        maxlags=maxlags,
        seed=42,
        n_jobs=n_jobs,  # PARAMETRIC
        full=True,
        dynamic=True,
        mb_estimator="ts",
    )
    d2c.initialize()
    d2c.get_descriptors_df().to_pickle(f"../../data/descriptors/{output_file}")
    print(f"Saved to data/descriptors/{output_file}")


def main(n_jobs):
    MAXLAGS = 3
    
    # 1. Train Data
    compute_descriptors("synthetic_split", "training_data", "descriptors_df_train.pkl", 5, MAXLAGS, n_jobs)
    
    # 2. Test Data
    compute_descriptors("synthetic_split", "testing_data", "descriptors_df_test.pkl", 5, MAXLAGS, n_jobs)
    
    # 3. Netsim 5
    compute_descriptors("single", "netsym/netsym_5", "descriptors_netsim_5.pkl", 5, MAXLAGS, n_jobs)
    
    # 4. Netsim 10
    compute_descriptors("single", "netsym/netsym_10", "descriptors_netsim_10.pkl", 10, MAXLAGS, n_jobs)
    
    # 5. Dream 10
    compute_descriptors("single", "dream3/dream3_10", "descriptors_dream3_10.pkl", 10, MAXLAGS, n_jobs)
    
    # 6. Dream 50
    compute_descriptors("single", "dream3/dream3_50", "descriptors_dream3_50.pkl", 50, MAXLAGS, n_jobs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=50)
    args = parser.parse_args()
    main(args.n_jobs)