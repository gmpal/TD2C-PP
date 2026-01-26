# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../../")

# %%
from td2c.data_generation.builder import TSBuilder
from src import config

# %%
for error in ["gaussian", "uniform", "laplace"]:
    print(f"Generating synthetic data with {error} noise...")

    tsbuilder = TSBuilder(
        observations_per_time_series=250,
        maxlags=config.MAXLAGS,
        n_variables=config.N_VARS,
        time_series_per_process=120,
        processes_to_use=[1, 3, 5, 7, 9, 11, 13, 15, 19],
        noise_dist=error,
        noise_scale=0.1,
        max_neighborhood_size=2,
        seed=42,
        max_attempts=10,
        verbose=True,
    )

    tsbuilder.build()

    tsbuilder.to_pickle(f"{config.OBSERVATION_PATH}training_data_{error}.pkl")

# %%
error_process_map = {
    "gaussian": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    "uniform": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    "laplace": [2, 4, 6, 8, 10, 12, 14, 16, 18],
}

for error in ["gaussian", "uniform", "laplace"]:
    print(f"Generating synthetic data with {error} noise...")

    tsbuilder = TSBuilder(
        observations_per_time_series=250,
        maxlags=config.MAXLAGS,
        n_variables=config.N_VARS,
        time_series_per_process=40,
        processes_to_use=error_process_map[error],
        noise_dist=error,
        noise_scale=0.1,
        max_neighborhood_size=2,
        seed=42,
        max_attempts=10,
        verbose=True,
    )

    tsbuilder.build()

    tsbuilder.to_pickle(f"{config.OBSERVATION_PATH}testing_data_{error}.pkl")

# %%
# we can check one of the generated time series
tsbuilder.get_generated_observations()

# %%
# we can check one of the generated dags
from td2c.data_generation.utils import show_DAG

dag = tsbuilder.get_generated_dags()[1][2]
show_DAG(dag, config.N_VARS, config.MAXLAGS)
