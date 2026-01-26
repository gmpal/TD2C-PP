import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import os
import networkx as nx
import pickle


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from scipy.stats import percentileofscore
import statsmodels.api as sm
from datetime import datetime


def from_descriptors_to_true_causal_dfs(descriptors_df):
    """
    Converts a DataFrame of descriptors into a dictionary of causal DataFrames.
    -----------
    descriptors_df : pandas.DataFrame
        DataFrame containing the descriptors with columns ['graph_id', 'edge_source', 'edge_dest'].
    Y_pred_proba : array-like
        Array-like object containing the predicted probabilities.
    Returns:
    --------
    dict
        A dictionary where the keys are graph IDs and the values are DataFrames with columns:
        ['from', 'to', 'effect', 'p_value', 'probability', 'is_causal'].
    """

    # from	to	effect	p_value	probability	is_causal
    causal_dfs = descriptors_df[
        ["graph_id", "edge_source", "edge_dest", "is_causal"]
    ].copy()
    # rename columns
    causal_dfs.columns = ["graph_id", "from", "to", "is_causal"]

    # add columns
    causal_dfs["effect"] = None
    causal_dfs["p_value"] = None
    causal_dfs["probability"] = None

    # sort by from to
    causal_dfs = causal_dfs.sort_values(by=["from", "to"])

    # groupby graph id
    groups = causal_dfs.groupby("graph_id")

    # make a list
    causal_df = {}
    for name, group in groups:
        causal_df[name] = group.drop(columns=["graph_id"]).reset_index(drop=True)

    # TODO: verificare che l'ordine dei gruppi sia lo stesso dell'ordine iniziale dei grafi
    # Il problema é che gli altri competitors hanno delle liste e non un unico grande df

    return causal_df


def from_descriptors_to_causal_dfs(descriptors_df, Y_pred_proba):
    """
    Converts a DataFrame of descriptors and predicted probabilities into a dictionary of causal DataFrames.
    Parameters:
    -----------
    descriptors_df : pandas.DataFrame
        DataFrame containing the descriptors with columns ['graph_id', 'edge_source', 'edge_dest'].
    Y_pred_proba : array-like
        Array-like object containing the predicted probabilities.
    Returns:
    --------
    dict
        A dictionary where the keys are graph IDs and the values are DataFrames with columns:
        ['from', 'to', 'effect', 'p_value', 'probability', 'is_causal'].
    """

    # from	to	effect	p_value	probability	is_causal
    causal_dfs = descriptors_df[["graph_id", "edge_source", "edge_dest"]].copy()
    # rename columns
    causal_dfs.columns = ["graph_id", "from", "to"]

    # add columns
    causal_dfs["effect"] = None
    causal_dfs["p_value"] = None
    causal_dfs["probability"] = Y_pred_proba
    causal_dfs["is_causal"] = Y_pred_proba > 0.5

    # sort by from to
    causal_dfs = causal_dfs.sort_values(by=["from", "to"])

    # groupby graph id
    groups = causal_dfs.groupby("graph_id")

    # make a list
    causal_df = {}
    for name, group in groups:
        causal_df[name] = group.drop(columns=["graph_id"]).reset_index(drop=True)
        print(causal_df[name])
    # TODO: verificare che l'ordine dei gruppi sia lo stesso dell'ordine iniziale dei grafi
    # Il problema é che gli altri competitors hanno delle liste e non un unico grande df

    return causal_df


def compute_partial_correlation_using_precision_matrix(
    df: pd.DataFrame, x1_name: str, x2_name: str, mb1_names: list, mb2_names: list
) -> float:
    """
    Compute the partial correlation between two variables given their respective Markov blankets. The equivalence between the residual-based method and the precision matrix method holds precisely when the data are multivariate Gaussian. This is because, in Gaussian distributions, conditional independence is fully captured by zero partial correlations, which directly relate to the concentration matrix.

    https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        x1_name (str): The name of the first variable.
        x2_name (str): The name of the second variable.
        mb1_names (list): The Markov blanket of the first variable.
        mb2_names (list): The Markov blanket of the second variable.

    Returns:
        float: The partial correlation between X1 and X2 given their Markov blankets.
    """
    # Combine into Z
    # print("x1_name")
    # print(x1_name)
    # print("x2_name")
    # print(x2_name)
    # print("mb1_names")
    # print(mb1_names)
    # print("mb2_names")
    # print(mb2_names)
    combined = []
    combined.append(x1_name)
    combined.append(x2_name)
    combined.extend(mb1_names)
    combined.extend(mb2_names)
    combined = list(dict.fromkeys(combined))  # Remove duplicates while preserving order
    # print(combined)
    Z = df[:, combined]
    # print(Z.shape)
    covariance_matrix = np.cov(Z, rowvar=False)
    # print(covariance_matrix.shape)

    # print("#########################")
    # print(Z.std(axis=0))
    # corr_matrix = np.corrcoef(Z, rowvar=False)
    # print(corr_matrix)
    # print("#########################")

    precision_matrix = np.linalg.inv(covariance_matrix)

    kappa_12 = precision_matrix[0, 1]
    kappa_11 = precision_matrix[0, 0]
    kappa_22 = precision_matrix[1, 1]

    partial_corr = -kappa_12 / np.sqrt(kappa_11 * kappa_22)
    return partial_corr


def compute_partial_correlation_using_residuals(
    df: pd.DataFrame, x1_name: str, x2_name: str, mb1_names: list, mb2_names: list
) -> float:
    """
    Compute the partial correlation between two variables given their respective Markov blankets using the residuals-based method. The partial correlation between $A$ and $B$, controlling for some set of variables $C$, is essentially the correlation of the residuals of $A$ and $B$ once those residuals have been purged of any linear dependence on $C$.

    Mathematically, if $C=\mathrm{MB}_A \cup \mathrm{MB}_B$ :
    1. Regress $A$ on $C$ to get residuals $\hat{r}_A$.
    2. Regress $B$ on $C$ to get residuals $\hat{r}_B$.
    3. Compute $\operatorname{corr}\left(\hat{r}_A, \hat{r}_B\right)$.

    That correlation of the residuals is the partial correlation between $A$ and $B$ given $C$.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        x1_name (str): The name of the first variable.
        x2_name (str): The name of the second variable.
        mb1_names (list): The Markov blanket of the first variable.
        mb2_names (list): The Markov blanket of the second variable.

    Returns:
        float: The partial correlation between X1 and X2 given their Markov blankets.
    """

    conditioning_set = []
    conditioning_set.extend(mb1_names)
    conditioning_set.extend(mb2_names)
    conditioning_set = list(
        dict.fromkeys(conditioning_set)
    )  # Remove duplicates while preserving order

    # print("conditioning_set")
    # print(conditioning_set)
    # print("x1")
    # print(x1_name)
    # print("x2")
    # print(x2_name)
    # print("df")
    # print(df.shape)
    # print("df[conditioning_set].shape")
    # print(df[:, conditioning_set].shape)
    # print("df[x1_name].shape")
    # print(df[:, x2_name].reshape(-1, 1).shape)

    # 2. Regress X1 on the Markov blankets
    X_A = sm.add_constant(df[:, conditioning_set])  # add intercept
    model_A = sm.OLS(df[:, x1_name].reshape(-1, 1), X_A).fit()
    resid_A = model_A.resid  # residuals of A

    # 3. Regress B on the Markov blankets
    X_B = sm.add_constant(df[:, conditioning_set])
    model_B = sm.OLS(df[:, x2_name].reshape(-1, 1), X_B).fit()
    resid_B = model_B.resid  # residuals of B

    resid_A_flat = resid_A.ravel()  # shape: (n_samples,)
    resid_B_flat = resid_B.ravel()  # shape: (n_samples,)

    # 4. Compute correlation of residuals
    partial_corr = np.corrcoef(resid_A_flat, resid_B_flat)[0, 1]

    return partial_corr


def compute_residuals_correlation_with_inputs(
    df: pd.DataFrame, x1_name: str, x2_name: str, mb1_names: list
) -> tuple:
    """
    Compute the correlation of the residuals of two variables with their respective inputs. This is a measure of the direct influence of the inputs on the variables, controlling for their Markov blankets.

    1. Model $Y=f\left(X, \mathrm{MB}_X, \ldots\right)$ :
        - Implies a causal direction from $X$ and its Markov blanket to $Y$.
        - Residual $E_Y$ represents the variation in $Y$ unexplained by $X$ and its MB.
    3. Residual Correlations:
        - Correlation of $E_Y$ with $X$ : If significant, suggests that $X$ still has an influence on $Y$ beyond what is captured by the MB.

    """

    concat = []
    concat.append(x1_name)
    concat.extend(mb1_names)

    # corr_matrix = np.corrcoef(df[concat])  # or np.corrcoef() if df is a NumPy array

    # print(corr_matrix)

    # X_Y = sm.add_constant(df[:, concat])  # Add intercept
    # model_Y = sm.OLS(df[:, x2_name].reshape(-1, 1), X_Y).fit()
    # resid_Y = model_Y.resid.reshape(-1, 1)

    from sklearn.linear_model import Ridge

    X = df[:, concat]
    Y = df[:, x2_name].reshape(-1, 1)

    model = Ridge(alpha=1.0).fit(X, Y)
    resid_Y = Y - model.predict(X)

    # print("resid_Y")
    # print(resid_Y.shape)
    # print("df[:, x1_name].reshape(-1, 1)")
    # print(df[:, x1_name].reshape(-1, 1).shape)
    # check if resid_Y is constant
    # if np.std(resid_Y) < 1e-10:
    #     return print("resid_Y is constant")

    # if np.std(df[:, x1_name].reshape(-1, 1)) < 1e-10:
    #     return print("df[:, x1_name].reshape(-1, 1) is constant")

    # # check for nans
    # if np.isnan(resid_Y).any():
    #     return print("resid_Y has nans")

    # if np.isnan(df[:, x1_name].reshape(-1, 1)).any():
    #     return print("df[:, x1_name].reshape(-1, 1) has nans")

    # Correlation of E_Y with X
    resid_flat = resid_Y.ravel()  # shape: (n_samples,)
    x1_flat = df[:, x1_name].ravel()  # shape: (n_samples,)

    corr_EY_X = np.corrcoef(resid_flat, x1_flat)[0, 1]

    return corr_EY_X


############################################################################################################
#######   COMMENTED OUT FOR NOW because it is compacted in the above function  ############################
#               ############################################################################################################
def ridge_regression(X_train, Y_train, X_test=None, lambda_val=1e-3):
    """
    Perform ridge regression and returns the trained model, predictions, and metrics.

    Args:
        X_train (np.ndarray): The training design matrix.
        Y_train (np.ndarray): The training response vector.
        X_test (np.ndarray, optional): The test design matrix. Defaults to None.
        lambda_val (float, optional): The regularization parameter. Defaults to 1e-3.

    Returns:
        dict: Dictionary containing the trained model, predictions, and computed metrics.
    """

    X_train = pd.DataFrame(X_train)

    model = Ridge(alpha=lambda_val)
    model.fit(X_train, Y_train)

    Y_train_hat = model.predict(X_train)
    e_train = Y_train - Y_train_hat
    MSE_emp = mean_squared_error(Y_train, Y_train_hat)
    NMSE = MSE_emp / (np.var(Y_train) ** 2)

    e_loo = cross_val_score(
        model, X_train, Y_train, scoring="neg_mean_squared_error", cv=2
    )
    MSE_loo = -np.mean(e_loo)

    Y_test_hat = None
    if X_test is not None:
        Y_test_hat = model.predict(X_test)

    return {
        "e_train": e_train,
        "beta_hat": [model.intercept_, model.coef_[0]],
        "MSE_emp": MSE_emp,
        "NMSE": NMSE,
        "MSE_loo": MSE_loo,
        "Y_train_hat": Y_train_hat,
        "Y_test_hat": Y_test_hat,
        "model": model,
    }


# def ridge_regression(X_train, Y_train, X_test=None, lambda_val=1e-3, verbose=True):
#     """
#         This function performs ridge regression, a variant of linear regression that
#         includes a regularization term. This method is used to prevent overfitting and to
#         handle multicollinearity (high correlation among predictors) in data. The regularization
#         term is controlled by a parameter lambda, which shrinks the coefficients towards zero.

#         The function is based on the R code from the original D2C package.

#         Args:
#             X (np.ndarray): The design matrix.
#             Y (np.ndarray): The response vector.
#             X_ts (np.ndarray, optional): The test design matrix. Defaults to None.
#             lambda_val (float, optional): The regularization parameter. Defaults to 1e-3.

#         Returns:
#             dict: Dictionary containing the computed metrics.

#     """
#     if verbose: print(datetime.now().strftime('%H:%M:%S'),'ridge_regression')

#     n = X_train.shape[1]  # Number of predictors
#     p = n + 1
#     N = X_train.shape[0]  # Number of observations

#     # Prepare the design matrix by adding a column of ones for the intercept
#     XX = np.c_[np.ones((N, 1)), X_train]

#     if lambda_val < 0:
#         if verbose: print(datetime.now().strftime('%H:%M:%S'),'lambda_val < 0')
#         min_MSE_loo = np.inf
#         for lambda_current in np.arange(1e-3, 5, 0.5):
#             H1 = pinv(XX.T @ XX + lambda_current * np.eye(p))
#             beta_hat = H1 @ XX.T @ Y_train
#             H = XX @ H1 @ XX.T
#             Y_hat = XX @ beta_hat
#             e = Y_train - Y_hat
#             e_loo = e / (1 - np.diag(H))
#             MSE_loo = np.mean(e_loo**2)
#             if MSE_loo < min_MSE_loo:
#                 lambda_val = lambda_current
#                 min_MSE_loo = MSE_loo

#     H1 = pinv(XX.T @ XX + lambda_val * np.eye(p))
#     beta_hat = H1 @ XX.T @ Y_train
#     H = XX @ H1 @ XX.T
#     Y_hat = XX @ beta_hat
#     e = Y_train - Y_hat
#     var_hat_w = e.T @ e / (N - p)
#     MSE_emp = np.mean(e**2)
#     e_loo = e / (1 - np.diag(H))
#     MSE_loo = np.mean(e_loo**2)
#     NMSE = np.mean(e_loo**2) / (np.var(Y_train)**2)
#     if verbose: print(datetime.now().strftime('%H:%M:%S'),'NMSE: ', NMSE)
#     Y_hat_ts = None
#     if X_test is not None:
#         if verbose: print(datetime.now().strftime('%H:%M:%S'),'X_test is not None')
#         N_ts = X_test.shape[0]
#         if np.isscalar(X_test) and n > 1:
#             Y_hat_ts = np.r_[1, X_test] @ beta_hat
#         else:
#             XX_ts = np.c_[np.ones((N_ts, 1)), X_test]
#             Y_hat_ts = XX_ts @ beta_hat

#     return {
#         'e': e,
#         'beta_hat': beta_hat,
#         'MSE_emp': MSE_emp,
#         'sdse_emp': tstd(e**2),
#         'var_hat': var_hat_w,
#         'MSE_loo': MSE_loo,
#         'sdse_loo': tstd(e_loo**2),
#         'Y_hat': Y_hat,
#         'Y_hat_ts': Y_hat_ts,
#         'e_loo': e_loo,
#         'NMSE': NMSE
#     }


def column_based_correlation(X, Y, verbose=True):
    # TODO: multidimensional Y
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "column_based_correlation")
    columns_of_X = X.shape[1]  # Number of columns in X

    correlation_vector = np.zeros(columns_of_X)  # Initialize correlation vector

    for i in range(columns_of_X):
        correlation_matrix = np.corrcoef(X.iloc[:, i], Y.iloc[:, 0])
        correlation_value = correlation_matrix[0, 1]
        correlation_vector[i] = correlation_value

    correlation_array = correlation_vector.reshape(1, -1)

    # Print the correlation vector
    return correlation_array[0]


def co2i(X, Y, verbose=True):

    # check if Y is a pd.series and make it dataframe
    if isinstance(Y, pd.Series):
        Y = pd.DataFrame(Y)

    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "co2i")

    correlation_vector = column_based_correlation(X, Y, verbose=verbose)
    corr_sq = np.square(correlation_vector)

    epsilon = 1e-10  # a small positive value to prevent log(0)
    clamped_values = np.clip(1 - corr_sq, epsilon, None)
    I = -0.5 * np.log(clamped_values)
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "I: ", I)

    return I


def rankrho(X, Y, nmax=5, regr=False, verbose=False):
    """
    Perform mutual information ranking between two arrays.

    Parameters:
        X (array-like): Input array with shape (N, n), representing N samples and n features.
        Y (array-like): Input array with shape (N,). Target variable.
        nmax (int, optional): Number of top-ranked features to return. Defaults to 5.
        regr (bool, optional): Flag indicating whether to use ridge regression for ranking. Defaults to False.
        verbose (bool, optional): Flag indicating whether to display progress information. Defaults to True.

    Returns:
        list: Indices of the top-ranked features in X based on mutual information with Y.

    Notes:
        The function calculates the mutual information between each column of X and Y, and returns the indices of the
        columns in X that have the highest mutual information with Y. The number of indices returned is determined by the
        nmax parameter.

        If the variance of Y is less than 0.01, the function returns a list of indices ranging from 1 to nmax.

        If regr is False, the function uses the co2i function to calculate the mutual information. If regr is True, ridge
        regression is performed for each column of X with Y as the target variable, and the maximum coefficient value is
        used as the mutual information.

        The input arrays X and Y are expected to have compatible shapes, where X has shape (N, n) and Y has shape (N,).
        The function assumes that the columns of X and Y correspond to the same samples.

    Example:
        X = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]

        Y = [10, 20, 30]

        top_features = rankrho(X, Y, nmax=2, regr=True)

        # Output: [3, 2]
        # The third column of X has the highest mutual information with Y, followed by the second column.

    """
    # Number of columns in X and Y
    n = X.shape[1]

    Iy = np.zeros(n)

    if not regr:
        Iy = co2i(X, Y, verbose=verbose)
    else:
        for i in range(n):
            Iy[i] = abs(ridge_regression(X.iloc[:, i], Y)["beta_hat"][1])

    argsort = np.argsort(Iy)
    reverse = argsort[::-1]
    to_return = reverse[:nmax]
    return to_return


def mRMR(X, Y, nmax, verbose=False):
    """
    Max-Relevance Min-Redundancy (mRMR) feature selection method.

    This function selects features based on maximizing mutual information (MI) with
    the target variable Y and minimizing the average mutual information among the
    selected features.

    Parameters:
    - X (pd.DataFrame): Feature matrix with rows as samples and columns as features.
    - Y (np.array or pd.Series): Target variable.
    - nmax (int): Maximum number of features to select.
    - verbose (bool, optional): If True, prints progress and intermediate results. Default is True.

    Returns:
    - list[int]: List of indices for the selected features.
    """

    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "mRMR")
    num_features = X.shape[1]

    # Calculate mutual information between each feature in X and Y
    mi_XY = mutual_info_regression(X, Y)
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "mi_XY: ", mi_XY)

    # Start with the feature with maximum MI with Y
    indices = [np.argmax(mi_XY)]

    for _ in range(nmax - 1):
        remaining_indices = list(set(range(num_features)) - set(indices))
        if verbose:
            print(
                datetime.now().strftime("%H:%M:%S"),
                "remaining_indices: ",
                remaining_indices,
            )
        mi_XX = np.zeros(len(remaining_indices))

        # Calculate mutual information between selected features and remaining features
        for i in range(len(remaining_indices)):
            mi_XX[i] = mutual_info_regression(
                X.iloc[:, indices], X.iloc[:, remaining_indices[i]]
            )[0]

        # Calculate MRMR score for each remaining feature
        mrmr_scores = mi_XY[remaining_indices] - np.mean(mi_XX)
        if verbose:
            print(datetime.now().strftime("%H:%M:%S"), "mrmr_scores: ", mrmr_scores)
        # Select feature with maximum MRMR score
        indices.append(remaining_indices[np.argmax(mrmr_scores)])

    return indices


def ecdf(data, verbose=False):
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "ecdf")

    def _ecdf(x):
        return percentileofscore(data, x) / 100

    return _ecdf


def coeff(y, x1, x2=None, verbose=False):
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "coeff")
    if x2 is not None:
        X = np.column_stack((x1, x2))
    else:
        X = np.array(x1).reshape(-1, 1)

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]  # return the coefficient of x1


# High Order Correlation
def HOC(x, y, i, j):
    return np.mean((x - np.mean(x)) ** i * (y - np.mean(y)) ** j) / (
        np.std(x) ** i * np.std(y) ** j
    )


def stab(X, Y, lin=True, R=10):
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-4)
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-4)

    Xhat = []
    Yhat = []

    for r in range(R):
        m1 = np.random.rand()
        m2 = np.random.rand()
        prob = np.exp(-((X - m1) ** 2) / (2 * 0.25**2)) + np.exp(
            -((X - m2) ** 2) / (2 * 0.25**2)
        )
        prob /= np.sum(prob)
        indices = np.random.choice(np.arange(len(X)), 100, p=prob)  # Get indices
        rX = X[indices]
        rY = Y[indices]  # Directly use the indices to get rY values

        model_Y = sm.OLS(rY, sm.add_constant(rX)).fit()
        model_X = sm.OLS(rX, sm.add_constant(rY)).fit()

        Xts = np.linspace(0, 1, 100)
        Yts = np.linspace(0, 1, 100)
        pY = model_Y.predict(sm.add_constant(Xts))
        pX = model_X.predict(sm.add_constant(Yts))
        Yhat.append(pY)
        Xhat.append(pX)

    return np.sign(np.mean(np.std(Yhat, axis=0)) - np.mean(np.std(Xhat, axis=0)))


def print_dag(dag, part="all"):
    if part == "all":
        print("#" * 20)
        for node, attr in dag.nodes(data=True):
            print(f"Node {node} has attributes {attr}")
        for edge_source, edge_dest, attrs in dag.edges(data=True):
            print(f"Edge {edge_source} -> {edge_dest} has attributes {attrs}")
    elif part == "nodes":
        print("#" * 20)
        for node in dag.nodes():
            print(f"Node {node}")
    elif part == "edges":
        print("#" * 20)
        for edge_source, edge_dest in dag.edges():
            print(f"Edge {edge_source} -> {edge_dest}")


def dag_to_formula(dag):
    import networkx as nx

    formula = ""
    for node in nx.topological_sort(dag):
        if f"_t-" not in str(node):
            formula += f"{node} = "
            bias = dag.nodes[node]["bias"]
            parents = list(dag.predecessors(node))
            for parent in parents:
                edge = dag.edges[parent, node]
                weight = edge["weight"]
                formula += f"{weight}*{parent} + "
            formula += f"{bias}\n"
    print(formula)


def custom_layout(G, n_nodes, t_lag):
    """
    Create a custom layout for the graph where nodes with the same identifier
    are aligned in the same column, regardless of their connections.
    """
    pos = {}
    width = 1.0 / (n_nodes - 1)
    height = 1.0 / (t_lag - 1)

    for node in G.nodes():
        if "_t-" in node:
            i, t = map(int, node.split("_t-"))
        else:
            i, t = int(node), 0
        pos[node] = (i * width, t * height)

    # Scale and center the positions
    pos = {node: (x * 10, y * 3) for node, (x, y) in pos.items()}
    return pos


def show_DAG(G):
    import networkx as nx
    import matplotlib.pyplot as plt

    # Using the custom layout for plotting
    plt.figure(figsize=(10, 6))
    pos_custom = custom_layout(G, 3, 3)
    nx.draw(
        G,
        pos_custom,
        with_labels=True,
        node_size=1000,
        node_color="lightpink",
        font_size=10,
        arrowsize=10,
    )
    plt.title("Time Series DAG with Custom Layout")
    plt.show()


def epred(X, Y):
    """
    Returns the predicted values for Y using linear regression.

    Parameters:
    - X: A 2D numpy array or matrix representing predictor variables.
    - Y: A 1D numpy array representing the response variable.

    Returns:
    Predicted values for Y.
    """
    # Get the number of rows and columns of the matrix X
    N, n = X.shape

    # Check for columns with almost constant values
    sds = np.std(X, axis=0)
    non_const_cols = sds > 0.01
    X = X.loc[:, non_const_cols]

    # Scale the predictors
    scaler = StandardScaler()
    XX = scaler.fit_transform(X)

    # Check for insufficient rows or NaN values
    if N < 5 or np.isnan(XX).any():
        raise ValueError("Error in epred")

    # Linear regression prediction
    reg = LinearRegression().fit(XX, Y)
    Y_hat = reg.predict(XX)

    return Y_hat


def make_name(node_idx):
    return "v" + str(node_idx)



def node_to_var_lag(node_idx, n_variables):
    """Converts a flat node index to a (variable_id, lag) tuple."""
    if node_idx < n_variables:
        return (node_idx, 0)  # It's a variable at time t (present)
    else:
        # Nodes are ordered: [vars_t, vars_t-1, vars_t-2, ...]
        offset_idx = node_idx - n_variables
        lag = (offset_idx // n_variables) + 1
        var_id = offset_idx % n_variables
        return (var_id, lag)


def var_lag_to_node(var_id, lag, n_variables, maxlags):
    """Converts a (variable_id, lag) tuple back to a flat node index."""
    if lag == 0:
        return var_id
    if lag > maxlags:
        return None  # This lag doesn't exist in our data

    # Calculate the index based on the ordering
    index_ = n_variables + (lag - 1) * n_variables + var_id
    return int(index_)




