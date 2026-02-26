# Reproducibility Package — IJF-D-24-00683R2

**Title:** Causal Discovery in Multivariate Time Series through Mutual Information Featurization
**Authors:** Gian Marco Paldino, Gianluca Bontempi
**Journal:** International Journal of Forecasting

---

The full reproducibility package is available at:

**https://github.com/gmpal/IJF-TD2C**

The repository contains all code, pre-computed data, and a detailed `README.md` covering the complete reproduction procedure, table/figure mapping, hardware requirements, and expected runtimes.

For reviewers who wish to verify the results without re-running the computationally intensive steps (descriptor computation and benchmark execution, which require a cluster and take 2–3 days), all intermediary data is included. The pre-computed data folder (`data.zip`, ~500 MB) can be downloaded from the link provided in the README, and the full pipeline can then be executed in approximately 5 minutes on a standard laptop using the provided skip flags:

```bash
cd reproduce/py_scripts
python pipeline.py --n_jobs 4 --skip_data --skip_descriptors --skip_benchmark
```

This reproduces all tables and figures in the paper.
