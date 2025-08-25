# %%
# avoids the need for users to install TD2C as a package
import sys

sys.path.append("../..")

# %% [markdown]
# Actual notebook to generate all descriptors for all possible pairs of variables for each of the datasets involved - Train, Test, Dream3-10, Dream3-50, Netsim-5, Netsim-10. <br><br>
# N.B. The true labels are present in these descriptor (`is_causal`) but when they will be passed to `D2CWrapper`, they will be replaced with the prediction from the fitted model. (`D2CWrapper` is the actual benchmark wrapper that would recompute all descriptors starting from observations only, but if we have computed them before, no need to repeat).
#
# <br>
# <b> !!! - This is computationally intense - !!! </b>

# %%
from d2c.descriptors import D2C, DataLoader

N_VARS = 5
MAXLAGS = 3
N_JOBS = 50

dataloaders = {}
original_observations_training = {}
lagged_flattened_observations_training = {}
flattened_dags_training = {}

for error_dist in ["gaussian", "uniform", "laplace"]:
    dataloader = DataLoader(n_variables=N_VARS, maxlags=MAXLAGS)
    dataloader.from_pickle(f"data/training_data_{error_dist}.pkl")

    dataloaders[error_dist] = dataloader
    original_observations_training[error_dist] = dataloader.get_original_observations()
    lagged_flattened_observations_training[error_dist] = dataloader.get_observations()
    flattened_dags_training[error_dist] = dataloader.get_dags()

original_observations_list_training = []
for obs_list in original_observations_training.values():
    original_observations_list_training.extend(obs_list)

lagged_flattened_observations_list_training = []
for obs_list in lagged_flattened_observations_training.values():
    lagged_flattened_observations_list_training.extend(obs_list)

flattened_dags_list_training = []
for dags_list in flattened_dags_training.values():
    flattened_dags_list_training.extend(dags_list)

d2c_new = D2C(
    observations=lagged_flattened_observations_list_training,
    dags=flattened_dags_list_training,
    couples_to_consider_per_dag=-1,
    n_variables=N_VARS,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=50,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()

d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_df_train.pkl")

# %%
dataloaders = {}
original_observations_testing = {}
lagged_flattened_observations_testing = {}
flattened_dags_testing = {}
true_causal_dfs = {}

for error_dist in ["gaussian", "uniform", "laplace"]:
    dataloader = DataLoader(n_variables=N_VARS, maxlags=MAXLAGS)
    dataloader.from_pickle(f"data/testing_data_{error_dist}.pkl")

    dataloaders[error_dist] = dataloader
    original_observations_testing[error_dist] = dataloader.get_original_observations()
    lagged_flattened_observations_testing[error_dist] = dataloader.get_observations()
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

from d2c.descriptors import D2C, DataLoader

d2c_new = D2C(
    observations=lagged_flattened_observations_list_testing,
    dags=flattened_dags_list_testing,
    couples_to_consider_per_dag=-1,
    n_variables=N_VARS,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=N_JOBS,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()
d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_df_test.pkl")

# %%
from d2c.descriptors import DataLoader

N_JOBS = 50
MAXLAGS = 3
N_VARS = 5
dataloader = DataLoader(n_variables=N_VARS, maxlags=MAXLAGS)
dataloader.from_pickle(f"realdata/netsym/netsym_5.pkl")

original_observations_testing = dataloader.get_original_observations()
lagged_flattened_observations_testing = dataloader.get_observations()
flattened_dags_testing = dataloader.get_dags()
true_causal_dfs = dataloader.get_true_causal_dfs()

from d2c.descriptors import D2C, DataLoader

d2c_new = D2C(
    observations=lagged_flattened_observations_testing,
    dags=flattened_dags_testing,
    couples_to_consider_per_dag=-1,
    n_variables=N_VARS,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=N_JOBS,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()
d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_netsim_5.pkl")

# %%
from d2c.descriptors import DataLoader

N_JOBS = 50
MAXLAGS = 3
N_VARS = 5
dataloader = DataLoader(n_variables=10, maxlags=MAXLAGS)
dataloader.from_pickle(f"realdata/netsym/netsym_10.pkl")

original_observations_testing = dataloader.get_original_observations()
lagged_flattened_observations_testing = dataloader.get_observations()
flattened_dags_testing = dataloader.get_dags()
true_causal_dfs = dataloader.get_true_causal_dfs()

from d2c.descriptors import D2C, DataLoader

d2c_new = D2C(
    observations=lagged_flattened_observations_testing,
    dags=flattened_dags_testing,
    couples_to_consider_per_dag=-1,
    n_variables=10,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=N_JOBS,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()
d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_netsim_10.pkl")

# %%
from d2c.descriptors import DataLoader

N_JOBS = 50
MAXLAGS = 3
N_VARS = 5
dataloader = DataLoader(n_variables=10, maxlags=MAXLAGS)
dataloader.from_pickle(f"realdata/dream3/dream3_10.pkl")

original_observations_testing = dataloader.get_original_observations()
lagged_flattened_observations_testing = dataloader.get_observations()
flattened_dags_testing = dataloader.get_dags()
true_causal_dfs = dataloader.get_true_causal_dfs()

from d2c.descriptors import D2C, DataLoader

d2c_new = D2C(
    observations=lagged_flattened_observations_testing,
    dags=flattened_dags_testing,
    couples_to_consider_per_dag=-1,
    n_variables=10,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=N_JOBS,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()
d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_dream3_10.pkl")

# %%
from d2c.descriptors import DataLoader

N_JOBS = 50
MAXLAGS = 3
N_VARS = 5
dataloader = DataLoader(n_variables=50, maxlags=MAXLAGS)
dataloader.from_pickle(f"realdata/dream3/dream3_50.pkl")

original_observations_testing = dataloader.get_original_observations()
lagged_flattened_observations_testing = dataloader.get_observations()
flattened_dags_testing = dataloader.get_dags()
true_causal_dfs = dataloader.get_true_causal_dfs()

from d2c.descriptors import D2C, DataLoader

d2c_new = D2C(
    observations=lagged_flattened_observations_testing,
    dags=flattened_dags_testing,
    couples_to_consider_per_dag=-1,
    n_variables=50,
    maxlags=MAXLAGS,
    seed=42,
    n_jobs=N_JOBS,
    full=True,
    dynamic=True,
    mb_estimator="ts",
)

d2c_new.initialize()
d2c_new.get_descriptors_df().to_pickle(f"data/descriptors/descriptors_dream3_50.pkl")
