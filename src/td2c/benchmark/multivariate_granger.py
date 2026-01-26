from src.td2c.benchmark.base import BaseCausalInference
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR


class MultivariateGranger(BaseCausalInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, single_ts, **kwargs):
        """
        Performs multivariate granger causality by fitting a VAR and then calling the test_causality method for each pair of variables.
        We apply t-tests to the coefficients of the VAR model to determine if the relationship is causal.
        Notice that this implementation of MVGC is limited:
        - Linearity Assumption: Relies on linear VAR models, potentially missing nonlinear causal relationships.
        - Computational Constraints: May not be optimized for very high-dimensional systems or large datasets.
        - Advanced Features: Lacks some advanced functionalities available in specialized toolboxes, such as frequency-domain causality or network visualizations.

        Returns a dictionary with the causal structure of the time series.
        The causal structure is as follows:
        {
        0: {0: {1: 1, 2: 0}, 1: {1: 1, 2: 1}, 2: {1: 0, 2: 0}},
        1: {0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}},
        2: {0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}, 2: {1: 1, 2: 0}}
        }

        and it's [from][to][lag] = 1 if the relationship is causal, 0 otherwise.

        """
        model = VAR(single_ts)
        results = model.fit(self.maxlags)

        names = ["const"]
        for lag in range(1, self.maxlags + 1):
            names.extend([f"{lag}." + str(var) for var in range(single_ts.shape[1])])
        names = np.array(names)

        causal_structure = {}
        for y_from in range(single_ts.shape[1]):
            causal_structure[y_from] = {}
            for y_to in range(single_ts.shape[1]):
                causal_structure[y_from][y_to] = {}
                test_result = results.test_causality(y_to, [y_from], kind="f")
                for lag in range(1, self.maxlags + 1):
                    causal_structure[y_from][y_to][lag] = 0
                if test_result.pvalue < 0.05:
                    y_to_pvalues = results.pvalues[:, y_to]

                    significant_lags = names[y_to_pvalues < 0.05]

                    # if the constant p-value is significant, we should ignore it
                    if "const" in significant_lags:
                        significant_lags = significant_lags[1:]

                    significant_lags = [
                        int(name.split(".")[0]) for name in significant_lags
                    ]

                    for lag in significant_lags:
                        causal_structure[y_from][y_to][lag] = 1

        return causal_structure

    def build_causal_df(self, results, n_variables):
        """
        This implementation of MVGS returns a dictionary as follows
        {
        0: {0: {1: 1, 2: 0}, 1: {1: 1, 2: 1}, 2: {1: 0, 2: 0}},
        1: {0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}, 2: {1: 0, 2: 0}},
        2: {0: {1: 0, 2: 0}, 1: {1: 0, 2: 0}, 2: {1: 1, 2: 0}}
        }

        and it's [from][to][lag] = 1 if the relationship is causal, 0 otherwise.
        """
        # initialization
        pairs = [
            (source, effect)
            for source in range(n_variables, n_variables * self.maxlags + n_variables)
            for effect in range(n_variables)
        ]
        multi_index = pd.MultiIndex.from_tuples(pairs, names=["from", "to"])
        causal_dataframe = pd.DataFrame(
            index=multi_index, columns=["effect", "p_value", "probability", "is_causal"]
        )

        for lag in range(self.maxlags):
            for source in range(n_variables):
                for effect in range(n_variables):
                    current_causal = results[source][effect][lag + 1]
                    causal_dataframe.loc[
                        (n_variables + source + lag * n_variables, effect)
                    ] = (None, None, None, current_causal)

        # break the multiindex into columns (from and to)
        causal_dataframe.reset_index(inplace=True)

        return causal_dataframe
