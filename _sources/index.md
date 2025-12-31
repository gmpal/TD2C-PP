# Causal Discovery in Multivariate Time Series

**A Supervised Learning Approach using Mutual Information Featurization**

---

### 🚀 The TD2C Framework
**TD2C (Temporal Dependency to Causality)** represents a paradigm shift in causal discovery. Rather than relying on restrictive linear assumptions (like Granger Causality) or binary conditional independence tests (like PC/PCMCI) which can fail in complex non-linear settings, TD2C reframes the problem as **pattern recognition**.

We hypothesize that a causal link creates a persistent and learnable **asymmetry** in the flow of information through a system. TD2C operationalizes this by learning to recognize these complex signatures from a rich set of information-theoretic and statistical descriptors.

### 📄 Abstract
Discovering causal relationships in complex multivariate time series is a fundamental scientific challenge. Traditional methods often falter, either by relying on restrictive linear assumptions or on conditional independence tests that become uninformative in the presence of intricate, non-linear dynamics.

This project proposes **TD2C**, a supervised learning framework that:
1.  **Featurizes** candidate links using Information Theory (Mutual Information, Transfer Entropy) and error-based statistics.
2.  **Learns** a model of causality from diverse synthetic time series.
3.  **Generalizes** zero-shot to unseen dynamics and real-world benchmarks (NetSim, DREAM3).

Our results show that TD2C achieves state-of-the-art performance, particularly in high-dimensional and non-linear settings.

---

### 👥 Authors & Affiliation

**Gian Marco Paldino** & **Gianluca Bontempi**
*Machine Learning Group, Computer Science Department*
*Université Libre de Bruxelles (ULB), Belgium*

---

### 📚 Structure of this Book

This documentation is organized to guide you from the theoretical foundations to the implementation details of the pipeline.

1.  **Theory & Background**
    *   [Background](02_theory/background.ipynb): An overview of Causal Discovery, Information Theory, and the Markov Blanket.
    *   [The Hypothesis](02_theory/hypothesis.ipynb): Defining the "Temporally-Aware Markov Blanket" and the hypothesis of Information Asymmetry.

2.  **Methodology**
    *   [Feature Engineering](03_pipeline/feature_engineering.ipynb): Detailed mathematical formulations of the TD2C descriptor set (Transfer Entropy, Residuals, Higher-Order Moments).
    *   **Pipeline**: (Coming soon) Training the classifier and running inference.

3.  **Experiments**
    *   **Results**: (Coming soon) Benchmarking against PCMCI, VarLiNGAM, and DYNOTEARS.

---

### 🔗 Resources
*   **Source Code:** [GitHub Repository](https://github.com/gmpal/TD2C-PP)
*   **Paper:** *Causal Discovery in Multivariate Time Series through Mutual Information Featurization* (Preprint)