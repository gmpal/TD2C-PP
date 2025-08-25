import pickle
import numpy as np
import networkx as nx
import pandas as pd


class DataLoader:
    """
    DataLoader class for handling time series data and directed acyclic graphs (DAGs) to prepare for the following stage: the descriptors computation.
    The preparation includes
    1. creating lagged time series,
    2. flattening the original dictionaries into coherent lists,
    3. renaming the nodes of the DAGs.

    Methods:
        __init__(self, maxlags=3, n_variables=3):
            Initializes the DataLoader object with specified maximum lags and number of variables.
        _flatten(self, dict_of_dicts):
            Converts a dictionary of dictionaries to a single list.
        _rename_dags(dags, n_variables):
            Renames the nodes of the DAGs to use the same convention as the descriptors.
        _create_lagged_multiple_ts(observations, maxlags):
            Creates lagged multiple time series from the given observations.
        _create_lagged_single_ts(obs, maxlags):
            Creates lagged single time series from the given observations.
        from_pickle(self, data_path):
            Loads data from a pickle file.
        from_tsbuilder(self, ts_builder):
            Loads data from a TimeSeriesBuilder object.
        get_observations(self):
            Returns the observations after creating the lagged time series.
        get_original_observations(self):
            Returns the original observations without creating the lagged time series.
        get_dags(self):
            Returns the DAGs after renaming the nodes.
        get_true_causal_dfs(self):
            Returns dataframes indicating true causal relationships in the DAGs.
    """

    def __init__(self, maxlags=3, n_variables=3):
        """
        Initializes the DataLoader object.

        Parameters:
        - maxlags (int): The maximum number of lags to create for the lagged time series.
        - n_variables (int): The number of variables in the data.
        """
        self.observations = None
        self.dags = None
        self.n_variables = n_variables
        self.maxlags = maxlags

    def _flatten(self, dict_of_dicts):
        """
        Convert a dictionary of dictionaries to a single list.
        This is useful because data is stored in different files, according to the generative process.
        When we load data, we keep track of the generative process as index of the dictionary.

        Parameters:
        - dict_of_dicts (dict): The dictionary of dictionaries to be flattened.

        Returns:
        - list_of_arrays (np.ndarray): The flattened array.
        """
        list_of_arrays = [
            dict_of_dicts[process][ts]
            for process in sorted(dict_of_dicts.keys())
            for ts in sorted(dict_of_dicts[process].keys())
        ]
        return list_of_arrays

    @staticmethod
    def _rename_dags(dags, n_variables):
        """
        Rename the nodes of the DAGs to use the same convention as the descriptors.
        Specifically, we rename the nodes from x_(t-y) to x + y*n_variables.
        We move from string to integer and we consider the variables from the past as different.

        Example:
        if n = 3
        - 3_t-0 -> 3
        - 1_t-1 -> 4
        - 3_t-1 -> 6

        Parameters:
        - dags (list): The list of DAGs to be renamed.
        - n_variables (int): The number of variables in the data.

        Returns:
        - updated_dags (list): The list of renamed DAGs.
        """
        updated_dags = []
        for dag in dags:
            mapping = {
                node: int(node.split("_")[0]) + int(node.split("-")[1]) * n_variables
                for node in dag.nodes()
            }  # from x_(t-y) to x + y*n_variables
            dag = nx.relabel_nodes(dag, mapping)
            updated_dags.append(dag)
        return updated_dags

    @staticmethod
    def _create_lagged_multiple_ts(observations, maxlags):
        """
        Create lagged multiple time series from the given observations.

        Parameters:
        - observations (list): A list of numpy arrays representing the time series observations.
        - maxlags (int): The maximum number of lags to create.

        Returns:
        - lagged_observations (list): A list of numpy arrays representing the lagged time series observations.
        """
        lagged_observations = []
        for obs in observations:
            lagged = obs.copy()
            for i in range(1, maxlags + 1):
                lagged = np.concatenate(
                    (lagged, np.roll(obs, i, axis=0)), axis=1
                )  # np roll brings last values to the top
            lagged_observations.append(
                lagged[maxlags:]
            )  # we need to drop the first maxlags rows
        return lagged_observations

    @staticmethod
    def _create_lagged_single_ts(obs, maxlags):
        """
        Create lagged single time series from the given observations.
        """
        lagged = obs.copy()
        for i in range(1, maxlags + 1):
            lagged = np.concatenate(
                (lagged, np.roll(obs, i, axis=0)), axis=1
            )  # np roll brings last values to the top
        return lagged[maxlags:]

    def load_netsym(self, data_path):
        #load .mat files
        import scipy.io as sio
        import numpy as np

        # load data
        for file in os.listdir(data_path):

            data = sio.loadmat(data_path + file)
            observations = data['ts']
            dag = nx.DiGraph(data['dag'])

            
        with open(data_path, "rb") as f:
            loaded_data = pickle.load(f)
        self.observations = self._flatten(loaded_data["observations"])
        self.dags = self._flatten(loaded_data["dags"])


    def from_pickle(self, data_path):
        """
        Data loader from a data file.

        Parameters:
        - data_path (str): The path to the data file.
        """
        with open(data_path, "rb") as f:
            loaded_observations, loaded_dags, _ = pickle.load(
                f
            )  # third element is neighbors, not used from this point on
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def from_tsbuilder(self, ts_builder):
        """
        Data loader from a TimeSeriesBuilder object.
        This prevents storing data on disk and allows a unique flow from data generation to descriptors computation to benchmark.

        Parameters:
        - ts_builder (TimeSeriesBuilder): The TimeSeriesBuilder object containing the generated data.
        """
        loaded_observations = ts_builder.get_generated_observations()
        loaded_dags = ts_builder.get_generated_dags()
        self.observations = self._flatten(loaded_observations)
        self.dags = self._flatten(loaded_dags)

    def get_observations(self):
        """
        Get the observations after having created the lagged time series.

        Returns:
        - lagged_observations (list): A list of numpy arrays representing the lagged time series observations.
        """
        return self._create_lagged_multiple_ts(self.observations, self.maxlags)

    def get_original_observations(self):
        """
        Get the observations WITHOUT creating the lagged time series.
        This is useful for methods that do not need the lagged time series.

        Returns:
        - observations (list): A list of numpy arrays representing the time series observations.
        """
        return self.observations

    def get_dags(self):
        """
        Get the DAGs after having renamed the nodes.

        Returns:
        - updated_dags (list): The list of renamed DAGs.
        """
        return self._rename_dags(self.dags, self.n_variables)

    def get_true_causal_dfs(self):
        causal_dataframes = []
        for dag in self._rename_dags(self.dags, self.n_variables):
            pairs = [
                (source, effect)
                for source in range(
                    self.n_variables, self.n_variables * self.maxlags + self.n_variables
                )
                for effect in range(self.n_variables)
            ]
            multi_index = pd.MultiIndex.from_tuples(pairs, names=["from", "to"])
            causal_dataframe = pd.DataFrame(index=multi_index, columns=["is_causal"])
            causal_dataframe["is_causal"] = 0
            for edge in dag.edges:
                source = edge[0]
                effect = edge[1]
                causal_dataframe.loc[(source, effect), "is_causal"] = 1

            causal_dataframe.reset_index(inplace=True)
            causal_dataframe = causal_dataframe.loc[
                causal_dataframe.to < self.n_variables
            ]
            causal_dataframes.append(causal_dataframe)

        return causal_dataframes
