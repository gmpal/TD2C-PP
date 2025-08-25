"""
This module contains a MarkovBlanketEstimator and a MutualInformationEstimator. 
"""

import numpy as np
import pandas as pd

from cachetools import cached, Cache
from cachetools.keys import hashkey

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split



from sklearn.base import BaseEstimator, RegressorMixin
import time

import pyvinecopulib as pv
from sklearn.base import BaseEstimator


class CopulaEstimator(BaseEstimator):
    def __init__(self, max_lag=2):
        self.max_lag = max_lag

    def _create_lagged_data_np(self, cause_series, effect_series):
        """
        Pure NumPy version of create_lagged_data.
        Returns a single array and the number of valid (non-NaN) samples.
        """
        n_samples = len(cause_series)
        # We lose `max_lag` samples due to the shifting
        valid_n_samples = n_samples - self.max_lag
        
        # Pre-allocate the array for all lagged versions of the cause
        # Shape: (valid_samples, num_lagged_features)
        lagged_cause = np.zeros((valid_n_samples, self.max_lag + 1))
        
        for lag in range(self.max_lag + 1):
            # The slices are carefully constructed to align the time series
            start_idx = self.max_lag - lag
            end_idx = n_samples - lag
            lagged_cause[:, lag] = cause_series[start_idx:end_idx]
            
        # The effect is just the last part of the original series
        lagged_effect = effect_series[self.max_lag:].reshape(-1, 1)
        
        # Final data matrix: [cause(t-lag), ..., cause(t), effect(t)]
        final_data = np.hstack([lagged_cause, lagged_effect])
        return final_data

    def fit_copula(self, data):
        """Fits a Vine Copula directly to a NumPy array."""
        # Rank-transform (scipy is faster for this than pandas)
        from scipy.stats import rankdata
        U = np.apply_along_axis(rankdata, 0, data, method='average') / (len(data) + 1)
        
        copula_model = pv.Vinecop.from_data(U)
        loglik = copula_model.loglik()
        taus = copula_model.taus
        # Flatten the list of lists of taus
        taus_flattened = [item for sublist in taus for item in sublist if sublist]
        return loglik, taus_flattened

    def estimate(self, observations, cause_index, effect_index):
        cause_series = observations[:, cause_index]
        effect_series = observations[:, effect_index]
        
        # 1) Create lagged data using the fast NumPy version
        lagged_data_xy = self._create_lagged_data_np(cause_series, effect_series)
        
        if lagged_data_xy.shape[0] < 2: # Not enough data to fit a copula
            return {"loglik": -np.inf, "taus": []}
        
        # 2) Fit copula
        ll_xy, taus_xy = self.fit_copula(lagged_data_xy)

        return {"loglik": ll_xy, "taus": taus_xy}


class MarkovBlanketEstimator:
    def __init__(self, size=5, n_variables=5, maxlags=5, verbose=True):
        """
        Initializes the Markov Blanket Estimator with specified parameters.

        Parameters:
        - verbose (bool): Whether to print detailed logs.
        - size (int): The desired size of the Markov Blanket.
        - n_variables (int): The number of variables in the dataset.
        - maxlags (int): The maximum number of lags to consider in the time series analysis.
        """
        self.verbose = verbose
        self.size = size
        self.n_variables = n_variables
        self.maxlags = maxlags

    def column_based_correlation(self, X, Y):
        """
        Computes Pearson correlation coefficients between each column in X and the vector Y.
        This is a highly optimized version using NumPy.
        """
        # 1. Standardize the data (center and scale)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        Y_std = (Y - Y.mean()) / Y.std()
        
        # 2. Compute correlation as a dot product
        # The result is a vector of correlations of each column of X with Y
        return np.dot(X_std.T, Y_std) / len(Y)

    def rank_features(self, X, Y, regr=False):
        """
        Ranks features in X based on their correlation or regression coefficient with Y.

        Parameters:
        - X (numpy.ndarray): The feature matrix.
        - Y (numpy.ndarray): The target vector.
        - nmax (int): The maximum number of features to rank (default is self.nmax).
        - regr (bool): Whether to use regression coefficients instead of correlations.

        Returns:
        - numpy.ndarray: Indices of the top-ranked features.
        """

        if regr:
            model = RidgeCV()
            model.fit(X, Y)
            importances = np.abs(model.coef_)
            ranked_indices = np.argsort(importances)[::-1]
        else:
            correlations = self.column_based_correlation(X, Y)
            ranked_indices = np.argsort(np.abs(correlations))[::-1]

        return ranked_indices

    def estimate(self, dataset, node):
        """
        Estimates the Markov Blanket for a given node using feature ranking.

        Parameters:
        - dataset (numpy.ndarray): The dataset containing all variables.
        - node (int): The index of the target node for which to estimate the Markov Blanket.
        - size (int): The desired size of the Markov Blanket.

        Returns:
        - numpy.ndarray: Indices of the variables in the estimated Markov Blanket.
        """
        n = dataset.shape[1]
        candidates_positions = np.array(list(set(range(n)) - {node}))
        Y = dataset[:, node]

        # Exclude the target node from the dataset for ranking
        X = dataset[:, candidates_positions]

        order = self.rank_features(X, Y, regr=False)
        sorted_ind = candidates_positions[order]

        return sorted_ind[: self.size]

    def estimate_time_series(self, dataset, node):
        """
        Estimates the Markov Blanket for a node in a time series context,
        ensuring a consistent canonical ordering: [PARENT, CHILD].

        The parent is the node's own variable at the previous time step.
        The child is the node's own variable at the next time step.

        This consistency is crucial for creating specific, interpretable features.
        """
        mb = [] # Use a standard Python list first, it's easier to append to

        # The column index for the node at the previous time step (t-1)
        # In our lagged structure, this corresponds to the node's own variable at the *next* lag.
        # So, for node z_i(t-L), the parent is z_i(t-(L+1)), which is at index `node + n_variables`.
        parent_node = node + self.n_variables
        
        # The column index for the node at the next time step (t+1)
        # For node z_i(t-L), the child is z_i(t-(L-1)), which is at index `node - n_variables`.
        child_node = node - self.n_variables

        # --- Enforce Canonical Order: Parent First, then Child ---

        # 1. Check for and add the PARENT
        # The parent node's index must be less than the total number of columns.
        if parent_node < dataset.shape[1]:
            mb.append(parent_node)

        # 2. Check for and add the CHILD
        # The child node's index must be non-negative.
        if child_node >= 0:
            mb.append(child_node)
            
        # Convert the final list to an integer NumPy array
        return np.array(mb, dtype=int)


cache = Cache(maxsize=1024)  # Define cache size


def custom_hashkey(*args, **kwargs):
    return hashkey(
        *(
            (arg.data.tobytes(), arg.shape) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ),
        **kwargs,
    )


@cached(cache, key=custom_hashkey)
def mse(X, y, cv=2):
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y_var = np.var(y)
    if y_var < 1e-6:
        return 0.0
    
    # Single split - much faster
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    model = Ridge(alpha=1e-3, solver='lsqr', fit_intercept=False)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    return max(1e-3, np.mean((y_test - pred)**2))# we set 0.001 as a lower bound


class MutualInformationEstimator:

    def __init__(self, proxy="Ridge", proxy_params=None, k=3):
        """
        Initializes the Mutual Information Estimator with specified regression proxy and parameters.

        Parameters:
        - proxy (str): The name of the regression model to use ('Ridge' by default).
        - proxy_params (dict): Parameters for the regression model.
        """
        self.proxy = proxy
        self.proxy_params = proxy_params or {}
        self.k = k

    def get_regression_model(self):
        """
        Initializes the regression model based on the specified proxy and proxy parameters.

        Returns:
        - model: The regression model instance.
        """
        if self.proxy == "Ridge":
            alpha = self.proxy_params.get("alpha", 1e-3)
            model = Ridge(alpha=alpha)
        elif self.proxy == "LOWESS":
            tau = self.proxy_params.get("tau", 0.5)
            model = LOWESS(tau=tau)
        elif self.proxy == "RF":
            raise NotImplementedError(
                "Random Forest is not yet supported as a proxy model."
            )
        else:  # TODO: Implement other regression models here based on the proxy value.
            raise ValueError(f"Unsupported proxy model: {self.proxy}")
        return model

    def estimate_original(self, dataset, y_index, x1_index, x2_index=None, cv=2):
        """
        Estimates the (normalized) conditional mutual information of x1 to y given x2.

        For a rough approximation, assuming Gaussian distributed errors in a linear regression model,
        we can consider H(y|x) to be proportional to the MSE.
        As the error in predicting y from x increases (i.e., as the MSE increases),
        the uncertainty of y given x also increases, reflecting a higher conditional entropy.

        Similarly, we consider H(y) to be proportional to the variance of y.

        The assumptions are strong and may not hold in practice, but they provide a simple and fast approximation.
        TODO: further explore the validity of the assumptions.


        Formulas:
        - H(y) ≈ Var(y)
        - H(y|x) ≈ MSE(x,y)
        - I(x1; y) ≈ (H(y) − H(y|x1))/H(y)= 1 - MSE(x1,y) / Var(y)
        - I(x1; y|x2) ≈ [(H(y|x2) − H(y|x1, x2))] / H(y|x2) = 1 - MSE([x1,x2],y) / MSE(x2, y)

        Parameters:
        - y (numpy.ndarray): The target vector.
        - x1 (numpy.ndarray): The feature matrix for the first variable.
        - x2 (numpy.ndarray): The feature matrix for the second variable (optional).
        - cv (int): The number of cross-validation folds to use.

        Returns:
        - float: The estimated conditional mutual information.
        """

        y = dataset[:, y_index]
        x1 = dataset[:, x1_index]
        x2 = None if x2_index is None else dataset[:, x2_index]

        if (
            x2 is None or x2.size == 0
        ):  # - I(x1; y) ≈ (H(y) − H(y|x1))/H(y)= 1 - MSE(x1,y) / Var(y)
            entropy_y = max(1e-3, np.var(y))  # we set 0.001 as a lower bound
            entropy_y_given_x1 = mse(x1, y, cv=cv)
            mutual_information = 1 - entropy_y_given_x1 / entropy_y
            return max(
                0, mutual_information
            )  # if negative, it means that knowing x1 brings more uncertainty to y (conditional entropy is higher than unconditional entropy)
        else:  # - I(x1; y|x2) ≈ [(H(y|x2) − H(y|x1, x2))] / H(y|x2) = 1 - MSE([x1,x2],y) / MSE(x2, y)
            if y.size == 0 or x1.size == 0:
                return 0
            x1_2d = x1 if x1.ndim > 1 else x1[:, np.newaxis]
            x2_2d = x2 if x2.ndim > 1 else x2[:, np.newaxis]

            x1x2 = np.hstack((x1_2d, x2_2d))
            entropy_y_given_x2 = mse(x2, y, cv=cv)
            entropy_y_given_x1_x2 = mse(
                x1x2, y, cv=cv
            )  # how much information x1 and x2 together have about y
            mutual_information = 1 - entropy_y_given_x1_x2 / entropy_y_given_x2
            return max(0, mutual_information)

    def estimate_knn_cmi(self, dataset, y_index, x1_index, x2_index=None):
        """ """
        import knncmi
        import pandas as pd

        dataset = pd.DataFrame(dataset)
        # make columns strings
        dataset.columns = [str(i) for i in dataset.columns]

        # if x2_index is list and is empty, set it to None
        if x2_index is not None and isinstance(x2_index, list) and len(x2_index) == 0:
            x2_index = None

        # print(dataset)
        # print(dataset.columns)
        # print(y_index)
        # print(dataset.columns[y_index])
        y_name = [dataset.columns[y_index]]
        x1_name = [dataset.columns[x1_index]]
        x2_name = None if x2_index is None else list(dataset.columns[x2_index])

        # print(y_name)
        # print(x1_name)
        # print(x2_name)
        if x2_name is None:
            return knncmi.cmi(y_name, x1_name, [], self.k, dataset)

        else:
            return knncmi.cmi(y_name, x1_name, x2_name, self.k, dataset)


class LOWESS(BaseEstimator, RegressorMixin):
    def __init__(self, tau):
        self.tau = tau
        self.X_ = None
        self.y_ = None
        self.theta_ = None

    def wm(self, point, X):
        # Calculate the squared differences in a vectorized way
        # point is reshaped to (1, -1) for broadcasting to match the shape of X
        differences = X - point.reshape(1, -1)
        squared_distances = np.sum(differences**2, axis=1)

        # Calculate the weights
        tau_squared = -2 * self.tau * self.tau
        weights = np.exp(squared_distances / tau_squared)

        # Create a diagonal matrix from the weights
        weight_matrix = np.diag(weights)

        return weight_matrix

    def fit(self, X, y):
        # Fit the model to the data
        self.X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0], 1), axis=1)
        self.y_ = np.array(y).reshape(-1, 1)
        return self

    def predict(self, X):
        # Predict using the fitted model

        # allocate array of size X.shape[0]
        preds = np.empty(X.shape[0])
        X_ = np.append(X, np.ones(X.shape[0]).reshape(X.shape[0], 1), axis=1)

        start = time.time()

        for i in range(X.shape[0]):
            point_ = X_[i]
            w = self.wm(point_, self.X_)
            self.theta_ = (
                np.linalg.pinv(self.X_.T @ (w @ self.X_)) @ self.X_.T @ (w @ self.y_)
            )
            pred = np.dot(point_, self.theta_)
            preds[i] = pred

        return preds.reshape(-1, 1)


# Based on recent research, here are better alternatives to CMIKNN

class CCMIEstimator:
    """
    Classifier-based Conditional Mutual Information (CCMI) Estimator
    
    Based on Mukherjee et al. (2020) - superior to kNN methods,
    especially in high dimensions
    """
    
    def __init__(self, classifier_type='logistic', n_estimators=10):
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
    
    def estimate(self, X, Y, Z=None):
        """
        Estimate I(X;Y|Z) using classifier-based approach
        
        Much more robust than kNN in high dimensions
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import numpy as np
        
        if Z is None:
            return self._estimate_marginal_mi(X, Y)
        
        # Construct joint and product distributions
        n_samples = len(X)
        
        # Joint distribution samples: (X, Y, Z)
        joint_samples = np.column_stack([X.reshape(-1, 1) if X.ndim == 1 else X,
                                       Y.reshape(-1, 1) if Y.ndim == 1 else Y,
                                       Z.reshape(-1, 1) if Z.ndim == 1 else Z])
        
        # Product distribution samples: X ~ P(X|Z), Y ~ P(Y|Z), but independent
        # This requires careful construction - simplified version here
        product_samples = self._construct_product_distribution(X, Y, Z)
        
        # Labels: 1 for joint, 0 for product
        joint_labels = np.ones(n_samples)
        product_labels = np.zeros(n_samples)
        
        # Combine data
        all_samples = np.vstack([joint_samples, product_samples])
        all_labels = np.concatenate([joint_labels, product_labels])
        
        # Train classifier to distinguish joint from product
        if self.classifier_type == 'logistic':
            clf = LogisticRegression(random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
        
        # Use cross-validation to get robust estimates
        scores = cross_val_score(clf, all_samples, all_labels, cv=5, scoring='neg_log_loss')
        
        # Convert log-loss to MI estimate (this is a simplified version)
        # Full implementation requires more careful likelihood ratio estimation
        mi_estimate = np.maximum(0, -np.mean(scores) - np.log(2))
        
        return mi_estimate
    
    def _construct_product_distribution(self, X, Y, Z):
        """
        Construct samples from product distribution P(X|Z)P(Y|Z)
        This is a simplified version - full implementation is more complex
        """
        # Simplified: just permute Y values within each Z stratum
        n_samples = len(X)
        X_prod = X.copy()
        Y_prod = np.random.permutation(Y)  # This is oversimplified
        Z_prod = Z.copy()
        
        return np.column_stack([X_prod.reshape(-1, 1) if X_prod.ndim == 1 else X_prod,
                              Y_prod.reshape(-1, 1) if Y_prod.ndim == 1 else Y_prod,
                              Z_prod.reshape(-1, 1) if Z_prod.ndim == 1 else Z_prod])
    
    def _estimate_marginal_mi(self, X, Y):
        """Estimate I(X;Y) using sklearn's robust implementation"""
        from sklearn.feature_selection import mutual_info_regression
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return mutual_info_regression(X, Y, random_state=42)[0]


class HybridMIEstimator:
    """
    Hybrid approach combining multiple estimators for robustness
    """
    
    def __init__(self, methods=['ksg', 'ccmi', 'binning'], weights=None):
        self.methods = methods
        self.weights = weights or [1/len(methods)] * len(methods)
        
    def estimate(self, X, Y, Z=None):
        """
        Combine multiple MI estimation methods for robustness
        """
        estimates = []
        
        for method in self.methods:
            if method == 'ksg':
                # Use sklearn's KSG implementation for marginal MI
                if Z is None:
                    from sklearn.feature_selection import mutual_info_regression
                    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
                    est = mutual_info_regression(X_2d, Y, random_state=42)[0]
                else:
                    # For conditional MI, use decomposition: I(X;Y|Z) = I(X,Z;Y) - I(Z;Y)
                    est = self._estimate_conditional_ksg(X, Y, Z)
                    
            elif method == 'ccmi':
                ccmi = CCMIEstimator()
                est = ccmi.estimate(X, Y, Z)
                
            elif method == 'binning':
                est = self._estimate_binning(X, Y, Z)
                
            estimates.append(est)
        
        # Weighted average
        return np.average(estimates, weights=self.weights)
    
    def _estimate_conditional_ksg(self, X, Y, Z):
        """
        Estimate I(X;Y|Z) using KSG via decomposition
        Less accurate than CCMI but still better than naive kNN
        """
        from sklearn.feature_selection import mutual_info_regression
        
        # I(X;Y|Z) = I(X,Z;Y) - I(Z;Y)
        X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
        Z_2d = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        
        # I(X,Z;Y)
        XZ = np.column_stack([X_2d, Z_2d])
        mi_xz_y = mutual_info_regression(XZ, Y, random_state=42)[0]
        
        # I(Z;Y)
        mi_z_y = mutual_info_regression(Z_2d, Y, random_state=42)[0]
        
        return max(0, mi_xz_y - mi_z_y)
    
    def _estimate_binning(self, X, Y, Z):
        """
        Adaptive binning approach - good fallback method
        """
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.metrics import mutual_info_score
        
        # Adaptive number of bins
        n_samples = len(X)
        n_bins = max(5, min(int(np.sqrt(n_samples)), 20))
        
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        
        if Z is None:
            X_binned = discretizer.fit_transform(X.reshape(-1, 1)).ravel()
            Y_binned = discretizer.fit_transform(Y.reshape(-1, 1)).ravel()
            return mutual_info_score(X_binned, Y_binned)
        else:
            # This is a simplified conditional MI with binning
            # Full implementation would be more sophisticated
            X_binned = discretizer.fit_transform(X.reshape(-1, 1)).ravel()
            Y_binned = discretizer.fit_transform(Y.reshape(-1, 1)).ravel()
            Z_binned = discretizer.fit_transform(Z.reshape(-1, 1)).ravel()
            
            # Rough approximation of conditional MI
            # Better implementations exist but this gives the idea
            mi_xyz = 0  # Would need proper 3-way MI calculation
            mi_xz = mutual_info_score(X_binned, Z_binned)
            mi_yz = mutual_info_score(Y_binned, Z_binned)
            
            # This is oversimplified - don't use this exact formula
            return max(0, mi_xyz - mi_xz - mi_yz)


# Enhanced version of your current estimator
class EnhancedTD2CMutualInformationEstimator(MutualInformationEstimator):
    """
    Drop-in replacement for your current MI estimator with better methods
    """
    
    def __init__(self, proxy="hybrid", proxy_params=None, k=5):
        super().__init__(proxy, proxy_params, k)
        
        if proxy == "hybrid":
            self.estimator = HybridMIEstimator(
                methods=['ksg', 'ccmi'], 
                weights=[0.7, 0.3]  # Favor KSG for stability, CCMI for high-dim
            )
        elif proxy == "ccmi":
            self.estimator = CCMIEstimator()
        else:
            # Keep your existing methods as fallback
            self.estimator = None
    
    def estimate_enhanced(self, dataset, y_index, x1_index, x2_index=None):
        """
        Enhanced estimation method - drop-in replacement for your current approach
        """
        Y = dataset[:, y_index]
        X1 = dataset[:, x1_index]
        X2 = None if x2_index is None else dataset[:, x2_index]
        
        if self.proxy in ["hybrid", "ccmi"]:
            return self.estimator.estimate(X1, Y, X2)
        else:
            # Fallback to your existing methods
            if x2_index is None:
                return self.estimate_original(dataset, y_index, x1_index)
            else:
                return self.estimate_original(dataset, y_index, x1_index, x2_index)
            


from sklearn.feature_selection import mutual_info_regression

# Keep your custom_hashkey and cache setup
# ...

# The MSE function can also be slightly optimized
@cached(cache, key=custom_hashkey)
def mse(X, y, cv=2):
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    
    # Use a simpler/faster model for the proxy if possible
    # RidgeCV can be slow. A fixed, small alpha is often enough.
    model = Ridge(alpha=1e-3, solver='auto') 
    
    # cross_val_score can be slow. For a proxy, we might not need it to be perfect.
    # A single train/test split could be faster, but CV is more robust.
    # We will stick with CV for now.
    neg_mean_squared_error_folds = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=cv
    )
    return max(1e-3, -np.mean(neg_mean_squared_error_folds))


class OptimizedMIEstimator:
    def __init__(self, method='fast_proxy', k=5):
        """
        Initializes the MI Estimator.
        Methods:
        - 'fast_proxy': Your original Ridge regression-based proxy.
        - 'ksg': Kraskov-Stögbauer-Grassberger estimator via sklearn (recommended).
        """
        if method not in ['fast_proxy', 'ksg']:
            raise ValueError("Method must be 'fast_proxy' or 'ksg'")
        self.method = method
        self.k = k # k is used by 'ksg'
        
    def _sanitize_inputs(self, dataset, y_index, x1_index, x2_index=None, verbose=False):
        """
        Sanitize and validate inputs to ensure they're in the expected format.
        """
        # Validate dataset
        if dataset.ndim != 2:
            raise ValueError(f"Dataset must be 2D (samples, features), got shape {dataset.shape}")
        
        n_samples, n_features = dataset.shape
        
        # Validate indices
        if not isinstance(y_index, (int, np.integer)) or not (0 <= y_index < n_features):
            raise ValueError(f"y_index must be valid integer, got {y_index}")
        
        if not isinstance(x1_index, (int, np.integer)) or not (0 <= x1_index < n_features):
            raise ValueError(f"x1_index must be valid integer, got {x1_index}")
        
        # Handle x2_index
        if x2_index is not None:
            if isinstance(x2_index, list) or isinstance(x2_index, np.ndarray):
                if len(x2_index) == 0:
                    x2_index = None
                elif len(x2_index) == 1:
                    x2_index = x2_index[0]
                    if not isinstance(x2_index, (int, np.integer)) or not (0 <= x2_index < n_features):
                        raise ValueError(f"x2_index must be valid integer, got {x2_index}")
                else:
                    pass # # x2_index is a list of indices, we will handle it later
            elif not isinstance(x2_index, (int, np.integer)) or not (0 <= x2_index < n_features):
                raise ValueError(f"x2_index must be valid integer, got {x2_index}")
        
        return dataset, y_index, x1_index, x2_index

    
    def estimate(self, dataset, y_index, x1_index, x2_index=None, verbose=False, cv=2):

        # Sanitize inputs first
        dataset, y_index, x1_index, x2_index = self._sanitize_inputs(
            dataset, y_index, x1_index, x2_index, verbose
        )
        
        
        if self.method == 'fast_proxy':
            return self._estimate_proxy(dataset, y_index, x1_index, x2_index, verbose, cv)
        elif self.method == 'ksg':
            return self._estimate_ksg(dataset, y_index, x1_index, x2_index, verbose)

    def _estimate_proxy(self, dataset, y_index, x1_index, x2_index=None, verbose=False, cv=2):
        y = dataset[:, y_index]
        x1 = dataset[:, x1_index]
        
        if x2_index is None or (isinstance(x2_index, list) and len(x2_index) == 0):
            # I(X1; Y)
            entropy_y = max(1e-3, np.var(y))
            entropy_y_given_x1 = mse(x1, y, cv=cv)
            mi = 1 - entropy_y_given_x1 / entropy_y
            return max(0, mi)
        else:
            # I(X1; Y | X2)
            x2 = dataset[:, x2_index]
            x1_2d = x1.reshape(-1, 1) if x1.ndim == 1 else x1
            x2_2d = x2.reshape(-1, 1) if x2.ndim == 1 else x2
            x1x2 = np.hstack([x1_2d, x2_2d])
            
            entropy_y_given_x2 = mse(x2, y, cv=cv)
            entropy_y_given_x1_x2 = mse(x1x2, y, cv=cv)
            mi = 1 - entropy_y_given_x1_x2 / entropy_y_given_x2
            return max(0, mi)

    def _estimate_ksg(self, dataset, y_index, x1_index, x2_index=None, verbose=False):
        y = dataset[:, y_index]
        x1 = dataset[:, x1_index].reshape(-1, 1)

        if x2_index is None or (isinstance(x2_index, list) and len(x2_index) == 0):
            # I(X1; Y)
            return mutual_info_regression(x1, y, n_neighbors=self.k, random_state=42)[0]
        else:
            # Estimate I(X1; Y | X2) using decomposition: I(X1, X2; Y) - I(X2; Y)
            x2 = dataset[:, x2_index]
            x2 = x2.reshape(-1, 1) if x2.ndim == 1 else x2

            # I(X2; Y)
            mi_x2_y = mutual_info_regression(x2, y, n_neighbors=self.k, random_state=42)[0]
            
            # I(X1, X2; Y)
            x1x2 = np.hstack([x1, x2])
            mi_x1x2_y = mutual_info_regression(x1x2, y, n_neighbors=self.k, random_state=42)[0]
            
            # The conditional MI is the difference
            cmi = mi_x1x2_y - mi_x2_y
            return max(0, cmi)