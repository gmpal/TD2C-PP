import pandas as pd
from td2c.benchmark.base import BaseCausalInference
from td2c.descriptors.loader import DataLoader
from td2c.descriptors.d2c import D2C as D2C_
from sklearn.impute import SimpleImputer
import numpy as np
import os  # Import the os module to check for file existence


class D2CWrapper(BaseCausalInference):
    """
    D2C class wrapper for causal inference using the D2C algorithm.

    This wrapper is designed for causal discovery on a single time series.
    It computes descriptors for all possible time-lagged edges and uses a
    pre-trained model to predict causal relationships.

    This class manages its own parallelization for computing descriptors,
    making it suitable for use within a sequential benchmarking loop.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the D2CWrapper with specific configurations.

        Args:
            model: A pre-trained classifier model with a .predict_proba method.
            n_variables (int): The number of variables (columns) in the raw time series.
            n_jobs (int): The number of parallel jobs to use for descriptor computation. Defaults to 1.
            cmi (str): The mutual information estimator to use (e.g., 'ksg', 'fast_proxy'). Defaults to 'ksg'.
            mb_estimator (str): The Markov Blanket estimator ('original' or 'ts'). Defaults to 'original'.
            normalize (bool): Whether to standardize the time series data. Defaults to False.
            dynamic (bool): Whether to compute dynamic (temporal) descriptors. Defaults to True.
            filename (str, optional): A base name for saving intermediate descriptors. Defaults to None.
            *args, **kwargs: Arguments passed to the BaseCausalInference parent class.
        """

        self.model = kwargs.pop("model", None)
        self.cmi = kwargs.pop("cmi", "ksg")
        self.mb_estimator = kwargs.pop("mb_estimator", "original")
        self.filename = kwargs.pop("filename", None)
        self.n_variables = kwargs.pop("n_variables", 6)
        self.threshold = kwargs.pop("threshold", 0.5)
        # Add a path to load pre-computed descriptors instead of calculating them.
        self.descriptors_path = kwargs.pop("precomputed_descriptors_path", None)

        super().__init__(*args, **kwargs)

        if self.model is None:
            raise ValueError("A pre-trained 'model' is required for D2CWrapper.")

        self.returns_proba = True

        # Pre-compute predictions if descriptors file is provided
        self._precomputed_predictions = None
        if self.descriptors_path and os.path.exists(self.descriptors_path):
            print(
                f"Loading and pre-computing predictions from: {self.descriptors_path}"
            )
            self._precompute_predictions()

    def _precompute_predictions(self):
        print("--- DEBUGGING D2CWRAPPER ---")
        descriptors_df = pd.read_pickle(self.descriptors_path)

        # Let's see the initial columns
        print(
            f"Initial descriptor columns ({len(descriptors_df.columns)}): {sorted(descriptors_df.columns)}"
        )

        X_test_initial = descriptors_df.drop(
            columns=["graph_id", "edge_source", "edge_dest", "is_causal"],
            errors="ignore",
        )

        training_features = self.model.feature_names_in_
        print(
            f"Model feature names ({len(training_features)}): {sorted(training_features)}"
        )

        # Find the difference
        missing_in_test = set(training_features) - set(X_test_initial.columns)
        extra_in_test = set(X_test_initial.columns) - set(training_features)

        if missing_in_test:
            print(
                f"\nFATAL MISMATCH: The model needs these features, but they were dropped or are missing:"
            )
            print(sorted(list(missing_in_test)))

        if extra_in_test:
            print(
                f"\nWARNING: These features are in the test data but the model wasn't trained on them:"
            )
            print(sorted(list(extra_in_test)))

        X_test_initial.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_initial.fillna(0, inplace=True)

        X_test_reindexed = X_test_initial.reindex(
            columns=training_features, fill_value=0
        )  # Use fill_value for safety

        # Check if reindexing created a bunch of zeros
        if missing_in_test:
            print(
                "\nBecause of the mismatch, the reindexed data for missing columns will be all zeros."
            )
            print(
                f"Example of reindexed data for a missing column '{list(missing_in_test)[0]}':"
            )
            print(X_test_reindexed[list(missing_in_test)[0]].describe())

        y_pred_proba = self.model.predict_proba(X_test_reindexed)[:, 1]

        # print(f"Prediction shape: {y_pred_proba.shape}")
        # print(f"Prediction sample: {y_pred_proba[:5]}")
        # Option 1: Store only predictions, not full descriptors
        self._precomputed_predictions = pd.DataFrame(
            {
                "graph_id": descriptors_df["graph_id"],
                "edge_source": descriptors_df["edge_source"],
                "edge_dest": descriptors_df["edge_dest"],
                "probability": y_pred_proba,
                "is_causal": (y_pred_proba > self.threshold).astype(int),
            }
        )

    def _validate_feature_compatibility(
        self, training_features, test_features, context="d2c versions"
    ):
        """
        Validate that training and test features match exactly.

        Args:
            training_features: Features used during model training
            test_features: Features available in test data
            context: Description of what might cause the mismatch (default: "d2c versions")

        Raises:
            ValueError: If features don't match exactly
        """
        missing_features = set(training_features) - set(test_features)
        extra_features = set(test_features) - set(training_features)

        if missing_features or extra_features:
            error_msg = f"Feature mismatch detected - likely due to different {context} used for training vs testing descriptors.\n"

            if missing_features:
                error_msg += (
                    f"Missing features in test data: {sorted(missing_features)}\n"
                )

            if extra_features:
                error_msg += f"Extra features in test data: {sorted(extra_features)}\n"

            error_msg += f"Training features count: {len(training_features)}, Test features count: {len(test_features)}"

            raise ValueError(error_msg)

    def infer(self, single_ts: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ts_index = kwargs.get("ts_index", "unknown")
        # print(f"Looking for ts_index: {ts_index}")

        # Use pre-computed predictions if available
        if self._precomputed_predictions is not None:
            # print(f"Available graph_ids: {self._precomputed_predictions['graph_id'].unique()}")
            # print(f"Total pre-computed predictions: {len(self._precomputed_predictions)}")

            # Filter predictions for this specific graph_id/ts_index
            results = self._precomputed_predictions[
                self._precomputed_predictions["graph_id"] == ts_index
            ].copy()

            # print(f"Filtered results shape: {results.shape}")
            # print(f"Results head:\n{results.head()}")

            if results.empty:
                raise ValueError(
                    f"No pre-computed predictions found for graph_id/ts_index: {ts_index}"
                )

            # Drop graph_id as it's not needed in the output
            results = results.drop(columns=["graph_id"])

            return results

        # Otherwise, compute from scratch (original implementation)
        print("Computing descriptors from scratch...")

        # Step 1: Prepare the time-lagged data
        data_for_d2c = DataLoader._create_lagged_single_ts(single_ts, self.maxlags)

        # Step 2: Compute descriptors using the D2C engine
        if not self.manages_own_parallelism:
            jobs_for_d2c = 1
        else:
            jobs_for_d2c = self.n_jobs

        d2c_engine = D2C_(
            dags=None,
            observations=[data_for_d2c],
            maxlags=self.maxlags,
            n_variables=self.n_variables,
            n_jobs=jobs_for_d2c,
            cmi=self.cmi,
            mb_estimator=self.mb_estimator,
        )

        descriptors_df = d2c_engine.compute_descriptors_without_dag(
            n_variables=self.n_variables, maxlags=self.maxlags
        )

        if self.filename is not None:
            output_path = f"{self.filename}_descriptors_{ts_index}.csv"
            descriptors_df.to_csv(output_path, index=False)

        # Step 3: Use the pre-trained model to predict causality
        X_test = descriptors_df.drop(
            columns=["graph_id", "edge_source", "edge_dest", "is_causal"],
            errors="ignore",
        )

        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        if X_test.isnull().values.any():
            print(
                f"Warning: Found {X_test.isnull().sum().sum()} NaN/inf values. Imputing with mean."
            )
            imputer = SimpleImputer(strategy="mean")
            X_test_imputed = imputer.fit_transform(X_test)
            X_test = pd.DataFrame(
                X_test_imputed, columns=X_test.columns, index=X_test.index
            )

        training_features = self.model.feature_names_in_
        self._validate_feature_compatibility(training_features, X_test.columns)
        X_test = X_test.reindex(columns=training_features)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Build the results DataFrame
        results = descriptors_df[["edge_source", "edge_dest"]].copy()
        results["probability"] = y_pred_proba
        results["is_causal"] = (y_pred_proba > self.threshold).astype(int)

        return results

    def build_causal_df(self, results: pd.DataFrame, n_variables: int) -> pd.DataFrame:
        """
        Formats the inference results into the standard causal DataFrame structure.
        """
        causal_df = results.rename(
            columns={
                "edge_source": "from",
                "edge_dest": "to",
            }
        )

        causal_df["p_value"] = 1 - causal_df["probability"]
        causal_df["effect"] = None

        # Define the columns we want in the final output
        final_columns = ["from", "to", "effect", "p_value", "probability", "is_causal"]

        # Include the true label if it exists, for convenience in evaluation
        if "is_causal_true" in causal_df.columns:
            final_columns.append("is_causal_true")

        final_df = causal_df[final_columns]

        return final_df.sort_values(by=["from", "to"]).reset_index(drop=True)
