from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from tqdm.auto import tqdm

from src.td2c.descriptors.utils import (
    coeff,
    HOC,
    compute_partial_correlation_using_residuals,
    compute_residuals_correlation_with_inputs,
    node_to_var_lag,
    var_lag_to_node,
)

from src.td2c.descriptors.estimators import MarkovBlanketEstimator, OptimizedMIEstimator


def _d2c_worker_with_dag(args):
    """Helper function to allow instance methods to be used by pool.imap_unordered."""
    # Unpack the instance of the D2C class and the rest of the arguments
    d2c_instance, dag_idx, dag, n_vars, maxlags, num_samples = args
    return d2c_instance.compute_descriptors_with_dag(
        dag_idx, dag, n_vars, maxlags, num_samples
    )


def _d2c_worker_for_couple(args):
    """
    Top-level worker function for parallelizing by couple.
    Accepts pre-computed Markov Blankets.
    """
    d2c_instance, dag_idx, ca, ef, label, MBca, MBef = args
    return d2c_instance.compute_descriptors_for_couple(
        dag_idx, ca, ef, label, MBca, MBef
    )


# Add this small helper to the top level of d2c.py for the MB parallelization
def _run_mb_func(func, dataset, node):
    return func(dataset, node)


class D2C:
    """
    D2C class for computing descriptors in a time series dataset.

    Args:
        dags (list): List of directed acyclic graphs (DAGs) representing causal relationships.
        observations (list): List of observations (pd.DataFrame) corresponding to each DAG.
        n_variables (int, optional): Number of variables in the time series. Defaults to 3.
        maxlags (int, optional): Maximum number of lags in the time series. Defaults to 3.
        mutual_information_proxy (str, optional): Method to use for mutual information computation. Defaults to "Ridge".
        proxy_params (dict, optional): Parameters for the mutual information computation. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1.

    Attributes:
        DAGs (list): List of DAGs representing causal relationships.
        dag_to_observation (dict): Mapping of DAG index to corresponding observation.
        x_y (None): Placeholder for computed descriptors.
        n_variables (int): Number of variables in the time series.
        maxlags (int): Maximum number of lags in the time series.
        test_couples (list): List of couples for which descriptors have been computed.
        mutual_information_proxy (str): Method used for mutual information computation.
        proxy_params (dict): Parameters for the mutual information computation.
        family (dict): Family of descriptors to compute.
        verbose (bool): Whether to print verbose output.
        n_jobs (int): Number of parallel jobs to run.
        seed (int): Random seed for reproducibility.

    Methods:

        initialize(self):
            Initialize the D2C object by computing descriptors in parallel for all observations.
        compute_descriptors_without_dag(self, n_variables, maxlags):
            Compute all descriptors when a DAG is not available.
        compute_descriptors_with_dag(self, dag_idx, dag, n_variables, maxlags, num_samples=20):
            Compute all descriptors associated to a DAG.
        get_markov_blanket(self, dag, node):
            Compute the REAL Markov Blanket of a node in a specific DAG.
        standardize_data(self, observations):
            Standardize the observation DataFrame.
        check_data_validity(self, observations):
            Check the validity of the data.
        update_dictionary_quantiles(self, dictionary, name, quantiles):
            Update the dictionary with quantiles.
        update_dictionary_distribution(self, dictionary, name, values):
            Update the dictionary with distribution moments.
        update_dictionary_actual_values(self, dictionary, name, values):
            Update the dictionary with actual values.
        compute_descriptors_for_couple(self, dag_idx, ca, ef, label):
            Compute descriptors for a given couple of nodes in a DAG.
        get_descriptors_df(self):
            Get the concatenated DataFrame of X and Y.
        get_test_couples(self):
            Get the test couples.
    """

    def __init__(
        self,
        dags,
        observations,
        couples_to_consider_per_dag=20,
        MB_size=5,
        n_variables=3,
        maxlags=3,
        mutual_information_proxy="Ridge",
        proxy_params=None,
        full=False,
        dynamic=False,
        verbose=False,
        cmi="ksg",
        mb_estimator="original",
        seed=42,
        n_jobs=1,
        new_descriptors_only=False,
        errors_and_parcorr=False,
    ) -> None:

        self.DAGs = dags
        self.observations = observations
        self.couples_to_consider_per_dag = couples_to_consider_per_dag
        self.n_variables = n_variables
        self.maxlags = maxlags
        self.mutual_information_proxy = mutual_information_proxy
        self.proxy_params = proxy_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed
        self.dynamic = dynamic

        self.new_descriptors_only = new_descriptors_only
        self.errors_and_parcorr = errors_and_parcorr
        self.dynamic_k_lag = 15

        self.x_y = None  # Placeholder for computed descriptors, list of dictionaries
        self.test_couples = (
            []
        )  # List of couples for which descriptors have been computed

        self.markov_blanket_estimator = MarkovBlanketEstimator(
            size=min(MB_size, n_variables - 2), n_variables=n_variables, maxlags=maxlags
        )

        self.mb_estimator = mb_estimator
        self.mutual_information_estimator = OptimizedMIEstimator(method=cmi)

        self.full = full

        np.random.seed(seed)

    def initialize(self) -> None:
        """
        Initialize the D2C object by computing descriptors in parallel for all observations.

        """
        if self.couples_to_consider_per_dag == -1:
            num_samples = -1
        else:
            num_samples = (
                self.couples_to_consider_per_dag // 2
            )  # half causal half non causal

        if self.n_jobs == 1:
            results = [
                self.compute_descriptors_with_dag(
                    dag_idx,
                    dag,
                    self.n_variables,
                    self.maxlags,
                    num_samples=num_samples,
                )
                for dag_idx, dag in tqdm(
                    enumerate(self.DAGs),
                    total=len(self.DAGs),
                    desc="Computing Descriptors",
                )
            ]

        else:
            args = [
                (self, dag_idx, dag, self.n_variables, self.maxlags, num_samples)
                for dag_idx, dag in enumerate(self.DAGs)
            ]

            with Pool(processes=self.n_jobs) as pool:
                # Call the top-level worker function
                results_iterator = pool.imap_unordered(_d2c_worker_with_dag, args)
                results = list(
                    tqdm(results_iterator, total=len(args), desc="Processing DAGs")
                )

        # merge lists into a single list
        results = [item for sublist in results for item in sublist]
        self.x_y = results

    def compute_descriptors_without_dag(self, n_variables, maxlags) -> list:
        """
        Compute all descriptors when a directed acyclic graph (DAG) is not available.
        This is useful for real testing data, but not for synthetic training data.
        So far it's one D2C object per synthetic dataset, so we don't need to pass the DAGs.
        We only have one set of observations, so we place them as if the dag index was 0.
        Synthetic labeled data is handled by the compute_descriptors_with_dag method.
        TODO: not clear, refactor, remove the dag_idx=0
        """

        # We assume there is only one observation set at index 0
        if not self.observations or len(self.observations) == 0:
            raise ValueError("Observations are required for this method.")
        observations = self.observations[0]

        all_possible_links = list(
            {  # Using a list for stable order
                (i, j)
                for i in range(n_variables, n_variables + n_variables * maxlags)
                for j in range(n_variables)
                if i != j
            }
        )

        # --- PRE-COMPUTATION OF MARKOV BLANKETS ---
        # This is CRITICAL for performance when parallelizing by couple.
        all_involved_nodes = set(
            node for couple in all_possible_links for node in couple
        )

        precomputed_markov_blankets = {}
        if self.verbose:
            print(f"Pre-computing MBs for {len(all_involved_nodes)} unique nodes...")

        mb_func = (
            self.markov_blanket_estimator.estimate
            if self.mb_estimator == "original"
            else self.markov_blanket_estimator.estimate_time_series
        )

        # We can parallelize the MB computation itself!
        mb_nodes = list(all_involved_nodes)
        if self.n_jobs > 1 and len(mb_nodes) > self.n_jobs:
            with Pool(processes=self.n_jobs) as pool:
                # Create tuples of (function, dataset, node)
                mb_args = [(mb_func, observations, node) for node in mb_nodes]
                # A simple starmap to run in parallel
                mb_results = pool.starmap(_run_mb_func, mb_args)
            precomputed_markov_blankets = dict(zip(mb_nodes, mb_results))
        else:  # Or run sequentially
            for node in mb_nodes:
                precomputed_markov_blankets[node] = mb_func(observations, node=node)
        # --- END OF PRE-COMPUTATION ---

        # --- PARALLEL COMPUTATION OF DESCRIPTORS (BY COUPLE) ---
        # Prepare arguments for each couple, including the pre-computed MBs
        args_for_couples = [
            (
                self,
                0,
                ca,
                ef,
                np.nan,
                precomputed_markov_blankets.get(ca),
                precomputed_markov_blankets.get(ef),
            )
            for ca, ef in all_possible_links
        ]

        if self.n_jobs == 1:
            results = [
                self.compute_descriptors_for_couple(*arg[1:])  # Unpack args, skip self
                for arg in args_for_couples
            ]
        else:
            print(
                f"Running descriptor computation in parallel with {self.n_jobs} jobs..."
            )
            with Pool(processes=self.n_jobs) as pool:
                results_iterator = pool.imap_unordered(
                    _d2c_worker_for_couple, args_for_couples
                )
                results = list(
                    tqdm(
                        results_iterator,
                        total=len(args_for_couples),
                        desc="Computing Descriptors",
                    )
                )

        return pd.DataFrame(results)

    def compute_descriptors_with_dag(
        self, dag_idx, dag, n_variables, maxlags, num_samples=20
    ) -> list:
        """
        Compute all descriptors associated to a directed acyclic graph (DAG).
        This is useful for synthetic training data, but not for real testing data.
        In this method, we can select specific couples based on their nature (causal or non-causal).
        Real unlabeled data is handled by the compute_descriptors_without_dags method.

        Args:
            dag (networkx.DiGraph): The directed acyclic graph.
            n_variables (int): The number of variables in the graph.
            maxlags (int): The maximum number of lags.
            num_samples (int, optional): The number of samples to consider. Defaults to 20.

        Returns:
            List of couples contains the computed descriptors, and the labels (1 for causal links, 0 for non-causal links).
        """

        if not maxlags > 0:
            raise ValueError(
                "maxlags must be greater than 0 for the current version of TD2C."
            )

        x_y_couples = []

        observations = self.observations[dag_idx]

        all_possible_links = {
            (i, j)
            for i in range(n_variables, n_variables + n_variables * maxlags)
            for j in range(n_variables)
            if i != j
        }

        causal_links = list(
            set(
                [(int(parent), int(child)) for parent, child in dag.edges]
            ).intersection(all_possible_links)
        )
        non_causal_links = list(all_possible_links - set(causal_links))

        all_involved_nodes = set()

        if num_samples != -1:
            # Sample the links first to know which nodes we need
            subset_causal_links = np.random.permutation(causal_links)[
                : min(len(causal_links), num_samples)
            ].astype(int)
            subset_non_causal_links = np.random.permutation(non_causal_links)[
                : min(len(non_causal_links), num_samples)
            ].astype(int)

            # Collect all unique nodes from the sampled couples
            for parent, child in subset_causal_links:
                all_involved_nodes.add(parent)
                all_involved_nodes.add(child)
            for node_a, node_b in subset_non_causal_links:
                all_involved_nodes.add(node_a)
                all_involved_nodes.add(node_b)
            nodes_to_compute_mb_for = list(all_involved_nodes)

        # 2. Pre-compute Markov Blankets for only the necessary nodes
        # This avoids computing MBs for nodes that aren't part of any sampled couple.
        precomputed_markov_blankets = {}

        if self.verbose:
            print(
                f"DAG {dag_idx}: Pre-computing MBs for {len(nodes_to_compute_mb_for)} unique nodes..."
            )

        # Choose the correct estimator based on the setting
        if self.mb_estimator == "original":
            mb_func = self.markov_blanket_estimator.estimate
        elif self.mb_estimator == "ts":
            mb_func = self.markov_blanket_estimator.estimate_time_series
        else:
            raise ValueError("Invalid Markov Blanket estimator")

        for node in nodes_to_compute_mb_for:
            precomputed_markov_blankets[node] = mb_func(observations, node=node)

        # --- End of Pre-computation Logic ---

        if num_samples == -1:
            for parent, child in causal_links:
                x_y_couples.append(
                    self.compute_descriptors_for_couple(
                        dag_idx,
                        parent,
                        child,
                        label=1,
                        MBca=precomputed_markov_blankets.get(
                            parent
                        ),  # Use .get() for safety
                        MBef=precomputed_markov_blankets.get(child),
                    )
                )  # causal
            for node_a, node_b in non_causal_links:
                x_y_couples.append(
                    self.compute_descriptors_for_couple(
                        dag_idx,
                        node_a,
                        node_b,
                        label=0,
                        MBca=precomputed_markov_blankets.get(node_a),
                        MBef=precomputed_markov_blankets.get(node_b),
                    )
                )  # noncausal

            self.test_couples.extend(causal_links)
            self.test_couples.extend(non_causal_links)

        else:

            for parent, child in subset_causal_links:
                x_y_couples.append(
                    self.compute_descriptors_for_couple(
                        dag_idx,
                        parent,
                        child,
                        label=1,
                        MBca=precomputed_markov_blankets[parent],
                        MBef=precomputed_markov_blankets[child],
                    )
                )  # causal
            for node_a, node_b in subset_non_causal_links:
                x_y_couples.append(
                    self.compute_descriptors_for_couple(
                        dag_idx,
                        node_a,
                        node_b,
                        label=0,
                        MBca=precomputed_markov_blankets[node_a],
                        MBef=precomputed_markov_blankets[node_b],
                    )
                )  # noncausal

            self.test_couples.extend(subset_causal_links)
            self.test_couples.extend(subset_non_causal_links)

        return x_y_couples

    def update_dictionary_from_population(self, dictionary, name, population):
        """
        Updates the dictionary with features derived from a descriptor population.

        If the population is small (<= 4), it creates specific, interpretable features
        from the raw values (e.g., _parent, _child).

        If the population is larger, it falls back to summary statistics (mean, std).
        """
        pop_size = len(population)

        if pop_size == 0:
            # Handle the case of an empty population
            dictionary[f"{name}_parent"] = 0.0  # Or np.nan
            dictionary[f"{name}_child"] = 0.0  # Or np.nan
            return

        if pop_size == 1:
            # This handles cases like mca_mca_cau
            # We give it a more generic name since it's not strictly parent/child
            dictionary[f"{name}_interaction"] = population[0]

        elif pop_size == 2:
            # This handles cases like m_cau, eff_m_cau, etc.
            # It relies on the canonical [parent, child] ordering.
            dictionary[f"{name}_parent"] = population[0]
            dictionary[f"{name}_child"] = population[1]

        elif pop_size == 4:
            # This handles cases like mca_mef_cau.
            # The order is [P_ca-P_ef, P_ca-C_ef, C_ca-P_ef, C_ca-C_ef]
            dictionary[f"{name}_pp"] = population[0]  # Parent-Parent
            dictionary[f"{name}_pc"] = population[1]  # Parent-Child
            dictionary[f"{name}_cp"] = population[2]  # Child-Parent
            dictionary[f"{name}_cc"] = population[3]  # Child-Child

        dictionary[f"{name}_mean"] = np.mean(population)
        dictionary[f"{name}_std"] = np.std(population)

    def compute_dynamic_descriptors(self, observations, ca, ef, CMI, min_k=1, max_k=15):
        """
        Computes a set of dynamic descriptors based on a generalized Transfer Entropy.
        This tests I(Cause(t-1); Effect(t) | Effect(t-k)) for a range of k.
        This respects the no-contemporaneous-links assumption.
        """
        te_values = {}

        # --- Step 1: Build the temporary, extended observation matrix up to max_k ---
        ts_data = observations[:, : self.n_variables]
        n_samples = ts_data.shape[0]

        if n_samples <= max_k:
            return {}

        all_lagged_blocks = [ts_data]
        for lag in range(1, max_k + 1):
            lagged_block = ts_data[: n_samples - lag, :]
            padding = np.full((lag, self.n_variables), np.nan)
            padded_block = np.vstack([padding, lagged_block])
            all_lagged_blocks.append(padded_block)

        extended_obs = np.concatenate(all_lagged_blocks, axis=1)
        # handle nans
        extended_obs = np.nan_to_num(extended_obs, nan=0.0)  # Replace NaNs with 0.0

        # --- Step 2: Calculate the generalized Transfer Entropy ---
        ca_var_idx, _ = node_to_var_lag(ca, self.n_variables)
        ef_var_idx, _ = node_to_var_lag(ef, self.n_variables)

        forward_te_variants = []
        backward_te_variants = []

        temp_maxlags = max_k

        for k in range(min_k, max_k + 1):
            # --- Forward Test: I(Z_i(t-1); Z_j(t) | Z_j(t-k)) ---
            source_fw_node = var_lag_to_node(
                ca_var_idx, 1, self.n_variables, temp_maxlags
            )  # Fixed to lag 1
            target_fw_node = var_lag_to_node(
                ef_var_idx, 0, self.n_variables, temp_maxlags
            )  # Fixed to lag 0
            cond_fw_node = var_lag_to_node(
                ef_var_idx, k, self.n_variables, temp_maxlags
            )  # Variable lag k

            fw_cmi = CMI(extended_obs, source_fw_node, target_fw_node, [cond_fw_node])
            forward_te_variants.append(fw_cmi)

            # --- Backward Test: I(Z_j(t-1); Z_i(t) | Z_i(t-k)) ---
            source_bw_node = var_lag_to_node(
                ef_var_idx, 1, self.n_variables, temp_maxlags
            )  # Fixed to lag 1
            target_bw_node = var_lag_to_node(
                ca_var_idx, 0, self.n_variables, temp_maxlags
            )  # Fixed to lag 0
            cond_bw_node = var_lag_to_node(
                ca_var_idx, k, self.n_variables, temp_maxlags
            )  # Variable lag k

            bw_cmi = CMI(extended_obs, source_bw_node, target_bw_node, [cond_bw_node])
            backward_te_variants.append(bw_cmi)

        # --- Step 3: Create highly-focused summary features ---
        if forward_te_variants and backward_te_variants:
            fwd_mean = np.mean(forward_te_variants)
            bwd_mean = np.mean(backward_te_variants)
            fwd_std = np.std(forward_te_variants)
            bwd_std = np.std(backward_te_variants)

            # The main feature: overall asymmetry in the window
            te_values[f"te_asymmetry_diff_{min_k}_{max_k}"] = fwd_mean - bwd_mean

            # Also include the value for k=1, which is the standard Transfer Entropy
            # This assumes min_k is 1
            if min_k == 1:
                te_values["transfer_entropy_fwd"] = forward_te_variants[0]
                te_values["transfer_entropy_bwd"] = backward_te_variants[0]
                te_values["transfer_entropy_diff"] = (
                    forward_te_variants[0] - backward_te_variants[0]
                )

        return te_values

    def compute_descriptors_for_couple(
        self, dag_idx, ca, ef, label, MBca=None, MBef=None
    ):
        """
        Compute descriptors for a given couple of nodes in a directed acyclic graph (DAG).

        Args:
            dag_idx (int): The index of the DAG.
            ca (int): The index of the cause node.
            ef (int): The index of the effect node.
            label (bool): The label indicating whether the edge between the cause and effect nodes is causal.

        Returns:
            dict: A dictionary containing the computed descriptors.

        """

        observations = self.observations[dag_idx]

        # If Markov Blankets are not provided, compute them.
        if MBca is None or MBef is None:
            raise ValueError("Markov Blankets must be provided or computed.")

        values = {}
        values["graph_id"] = dag_idx
        values["edge_source"] = ca
        values["edge_dest"] = ef
        values["is_causal"] = label

        # e, c = observations[:, ef], observations[:, ca] #aliases 'e' and 'c' for brevity
        e, c = ef, ca
        CMI = self.mutual_information_estimator.estimate

        values["parcorr_errors"] = compute_partial_correlation_using_residuals(
            observations, ca, ef, MBca, MBef
        )

        values["errors_correlation_with_inputs"] = (
            compute_residuals_correlation_with_inputs(observations, ca, ef, MBca)
        )

        common_causes = list(set(MBca).intersection(MBef))

        mbca_mbef_couples = [(i, j) for i in MBca for j in MBef]

        mbca_mbca_couples = [(i, j) for i in MBca for j in MBca if i != j]

        mbef_mbef_couples = [(i, j) for i in MBef for j in MBef if i != j]

        # b: ef = b * (ca + mbef)
        values["coeff_cause"] = coeff(
            observations[:, e], observations[:, c], observations[:, MBef]
        )

        # b: ca = b * (ef + mbca)
        values["coeff_eff"] = coeff(
            observations[:, c], observations[:, e], observations[:, MBca]
        )

        values["HOC_3_1"] = HOC(observations[:, c], observations[:, e], 3, 1)
        values["HOC_1_2"] = HOC(observations[:, c], observations[:, e], 1, 2)
        values["HOC_2_1"] = HOC(observations[:, c], observations[:, e], 2, 1)
        values["HOC_1_3"] = HOC(observations[:, c], observations[:, e], 1, 3)

        values["kurtosis_ca"] = kurtosis(observations[:, c])
        values["kurtosis_ef"] = kurtosis(observations[:, e])

        if self.dynamic:
            # Compute dynamic descriptors using helper function
            dynamic_values = self.compute_dynamic_descriptors(observations, ca, ef, CMI)
            # Merge dynamic descriptors into main values dictionary
            values.update(dynamic_values)

        # I(mca ; mef | cause) for (mca,mef) in mbca_mbef_couples
        # mca_mef_cau = [0] if not len(mbca_mbef_couples) else [CMI(observations[:,i], observations[:,j], c) for i, j in mbca_mbef_couples]
        mca_mef_cau = (
            [0]
            if not len(mbca_mbef_couples)
            else [CMI(observations, i, j, c) for i, j in mbca_mbef_couples]
        )

        # I(mca ; mef| effect) for (mca,mef) in mbca_mbef_couples
        # mca_mef_eff = [0] if not len(mbca_mbef_couples) else [CMI(observations[:,i], observations[:,j], e) for i, j in mbca_mbef_couples]
        mca_mef_eff = (
            [0]
            if not len(mbca_mbef_couples)
            else [CMI(observations, i, j, e) for i, j in mbca_mbef_couples]
        )

        # I(cause; m | effect) for m in MBef
        # cau_m_eff = [0] if not len(MBef) else [CMI(c, observations[:, m], e) for m in MBef]
        cau_m_eff = [0] if not len(MBef) else [CMI(observations, c, m, e) for m in MBef]

        # I(effect; m | cause) for m in MBca
        # eff_m_cau = [0] if not len(MBca) else [CMI(e, observations[:, m], c) for m in MBca]
        eff_m_cau = [0] if not len(MBca) else [CMI(observations, e, m, c) for m in MBca]

        # I(m; cause) for m in MBef
        # m_cau = [0] if not len(MBef) else [CMI(c, observations[:, m]) for m in MBef]
        m_cau = [0] if not len(MBef) else [CMI(observations, c, m) for m in MBef]

        # I(cause; effect | common_causes)
        # values['com_cau'] = CMI(e, c, observations[:, common_causes])
        values["com_cau"] = CMI(observations, e, c, common_causes)

        # I(cause; effect)
        # values['cau_eff'] = CMI(e, c)
        values["cau_eff"] = CMI(observations, e, c)

        # I(effect; cause)
        # values['eff_cau'] = CMI(c, e)
        values["eff_cau"] = CMI(observations, c, e)

        # I(effect; cause | MBeffect)
        # values['eff_cau_mbeff'] = CMI(c, e, observations[:, MBef])
        values["eff_cau_mbeff"] = CMI(observations, c, e, MBef)

        # I(cause; effect | MBcause)
        # values['cau_eff_mbcau'] = CMI(e, c, observations[:, MBca])
        values["cau_eff_mbcau"] = CMI(observations, e, c, MBca)

        # I(effect; cause | arrays_m_plus_MBca)
        # eff_cau_mbcau_plus = [0] if not len(MBef) else [CMI(c, e, observations[:,np.unique(np.concatenate(([m], MBca)))]) for m in MBef]
        eff_cau_mbcau_plus = (
            [0]
            if not len(MBef)
            else [
                CMI(observations, c, e, np.unique(np.concatenate(([m], MBca))))
                for m in MBef
            ]
        )

        # I(cause; effect | arrays_m_plus_MBef)
        # cau_eff_mbeff_plus = [0] if not len(MBca) else [CMI(e, c, observations[:,np.unique(np.concatenate(([m], MBef)))]) for m in MBca]
        cau_eff_mbeff_plus = (
            [0]
            if not len(MBca)
            else [
                CMI(observations, e, c, np.unique(np.concatenate(([m], MBef))))
                for m in MBca
            ]
        )

        # I(m; effect) for m in MBca
        # m_eff = [0] if not len(MBca) else [CMI(e, observations[:, m]) for m in MBca]
        m_eff = [0] if not len(MBca) else [CMI(observations, e, m) for m in MBca]

        # I(mca ; mca| cause) - I(mca ; mca) for (mca,mca) in mbca_couples
        # mca_mca_cau = [0] if not len(mbca_mbca_couples) else [CMI(observations[:,i], observations[:,j], c) - CMI(observations[:,i], observations[:,j]) for i, j in mbca_mbca_couples]
        mca_mca_cau = (
            [0]
            if not len(mbca_mbca_couples)
            else [
                CMI(observations, i, j, c) - CMI(observations, i, j)
                for i, j in mbca_mbca_couples
            ]
        )

        # I(mbe ; mbe| effect) - I(mbe ; mbe) for (mbe,mbe) in mbef_couples
        # mbe_mbe_eff = [0] if not len(mbef_mbef_couples) else [CMI(observations[:,i], observations[:,j], e) - CMI(observations[:,i], observations[:,j]) for i, j in mbef_mbef_couples]
        mbe_mbe_eff = (
            [0]
            if not len(mbef_mbef_couples)
            else [
                CMI(observations, i, j, e) - CMI(observations, i, j)
                for i, j in mbef_mbef_couples
            ]
        )

        values["skewness_ca"] = skew(observations[:, c])
        values["skewness_ef"] = skew(observations[:, e])

        self.update_dictionary_from_population(values, "mca_mef_cau", mca_mef_cau)
        self.update_dictionary_from_population(values, "mca_mef_eff", mca_mef_eff)
        self.update_dictionary_from_population(values, "cau_m_eff", cau_m_eff)
        self.update_dictionary_from_population(values, "eff_m_cau", eff_m_cau)
        self.update_dictionary_from_population(values, "m_cau", m_cau)
        self.update_dictionary_from_population(
            values, "eff_cau_mbcau_plus", eff_cau_mbcau_plus
        )
        self.update_dictionary_from_population(
            values, "cau_eff_mbeff_plus", cau_eff_mbeff_plus
        )
        self.update_dictionary_from_population(values, "m_eff", m_eff)

        self.update_dictionary_from_population(values, "mca_mca_cau", mca_mca_cau)
        self.update_dictionary_from_population(values, "mbe_mbe_eff", mbe_mbe_eff)

        return values

    def get_descriptors_df(self) -> pd.DataFrame:
        """
        Get the concatenated DataFrame of X and Y.

        Returns:
            pd.DataFrame: The concatenated DataFrame of X and Y.

        """
        return pd.DataFrame(self.x_y)

    def get_test_couples(self):
        return self.test_couples
