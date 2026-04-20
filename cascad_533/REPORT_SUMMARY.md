# CASCAD Verification Report #533 — Summary

**Report date:** April 13, 2026
**Manuscript:** IJF-D-24-00683R3
**Title:** Causal Discovery in Multivariate Time Series through Mutual Information Featurization
**Authors:** Gian Marco Paldino, Gianluca Bontempi

**Verifier environment:** Windows 11 Enterprise, 32-core Intel CPU, 512 GB RAM, Python 3.10
**Package downloaded:** April 7, 2026 from GitHub

---

## Section 3 — Verification Steps

No execution errors encountered. Pipeline ran successfully end-to-end.

---

## Section 4 — Findings Overview

| Item | Status | Details |
|---|---|---|
| Figure 1–2 | Not code-generated | Theoretical/explanatory |
| Figure 3 | ✅ Accurate | F1-Score boxplot on realistic benchmarks |
| Figure 4 | ✅ Accurate | CD diagrams (Precision & Recall) |
| Figure I.6 | ⚠️ Discrepancies | Panel A (F1-Score CD) has grouping differences; Panels B & C accurate |
| Table 1 | ✅ Accurate (partial) | Values match, but **Difference row missing** from console output |
| Table 2 | Not code-generated | Theoretical |
| Table 3 | ⚠️ Discrepancies | Youden's J: 0.278 vs. 0.382 paper (gap of 0.104) |
| Table 4 | ⚠️ Discrepancies | TD2C row differs; all other methods exact |
| Table 5 | ⚠️ Discrepancies | PCMCI-GPDC runtime not reproduced; all others hardware-dependent |
| Table 6 | ✅ Minor expected | All 15 features present; rank swaps at positions 8–9 and 12–14 |
| Table G.13 | ⚠️ Discrepancies | TD2C per-process values differ across all processes |
| Table H.14 | ⚠️ Discrepancies | TD2C realistic benchmark values differ (all 4 datasets) |

---

## Detailed Findings

### 4.1 — Table 1: Open Information Paths

Values reproduced correctly for all three scenarios (Simple, Complex, Latent). However, the **Difference (Δ) row** (Forward minus Backward) is not printed in the console output. The paper's table includes this row explicitly.

**Issue:** `00.py` computes Forward and Backward counts but does not print the Δ row.

---

### 4.2 — Table 3: Threshold Selection Metrics (LOGO-CV)

3 of 4 metrics within 1–2% tolerance (acceptable floating-point variation):

| Metric | Paper | Reproduced | Delta |
|---|---|---|---|
| Maximize F1-Score | 0.478 | 0.460 | 0.018 ✅ |
| Precision-Recall Break-Even | 0.618 | 0.609 | 0.009 ✅ |
| Maximize Youden's J | 0.382 | 0.278 | **0.104 ⚠️** |
| Minimize D_ROC(0,1) | 0.309 | 0.293 | 0.016 ✅ |

Youden's J gap is notably larger than the others — approximately 6× the typical variation. Verifiers flagged this but considered the overall reproduction successful given the remaining metrics.

---

### 4.3 — Table 4: Overall Synthetic Performance

All competitor methods reproduced exactly. TD2C row shows discrepancies:

| Metric | Paper | Reproduced |
|---|---|---|
| Accuracy | 0.8533 ± 0.0972 | 0.8430 ± 0.0960 |
| Balanced Accuracy | 0.8218 ± 0.1344 | 0.8185 ± 0.1339 |
| F1-Score | 0.6306 ± 0.2126 | 0.6282 ± 0.2098 |
| Precision | 0.5637 ± 0.2368 | 0.5717 ± 0.2426 |
| Recall | 0.7708 ± 0.2241 | 0.7605 ± 0.2275 |

Note: Verifiers used the recommended pre-computed data (`causal_dfs_TEST.pkl`), yet discrepancies remain. This is more significant than in round #518 where exact match was achieved.

---

### 4.4 — Figure 3: F1-Score Boxplot (Realistic Benchmarks)

Reproduced with accuracy. ✅

---

### 4.5 — Figure 4: CD Diagrams (Precision & Recall)

Reproduced with accuracy. Rankings and statistical groupings confirmed. ✅

---

### 4.6 — Table 5: Runtime Benchmark

Runtimes differ from paper (expected — hardware-dependent). Key issue: **PCMCI-GPDC runtimes are not reproduced at all** (missing from output).

| Method | Paper N=25 | Reproduced N=25 |
|---|---|---|
| VAR | 0.141 | 0.304 |
| VARLiNGAM | 342.997 | 121.565 |
| Granger | 2.982 | 5.146 |
| DYNOTEARS | 0.025 | 0.016 |
| MultivarGranger | 6.127 | 22.005 |
| PCMCI (ParCorr) | 3.777 | 18.835 |
| PCMCI (GPDC) | 191.062 | **not reproduced** |
| TD2C (1 job) | 1617.625 | 3498.374 |
| TD2C (50 jobs) | 57.169 | 223.457 |

---

### 4.7 — Table 6: Feature Importance

All 15 features identical as a set. Top 7 match in exact order. Minor rank swaps:
- Ranks 8–9 inverted (HOC_1_3 ↔ te_asymmetry_diff_1_15)
- Ranks 12–14 partially reordered (m_eff_std, cau_eff, mca_mef_eff_std)

Verifiers note this is **expected** given near-identical importance scores. ✅ minor expected

---

### 4.8 — Table G.13: Per-Process Comprehensive Comparison

TD2C rows flagged with discrepancies across all processes (2, 4, 6, 8, 10, 12, 14, 16, 18). Competitor method rows match the paper exactly. Pattern is consistent with Table 4 discrepancy — the TD2C model or its evaluation is producing slightly different outputs.

---

### 4.9 — Table H.14: Realistic Benchmark (per dataset)

TD2C rows show discrepancies across all four realistic datasets:
- DREAM3_10, DREAM3_50, NETSIM_5, NETSIM_10

All competitor results match the paper exactly.

---

### 4.10 — Figure I.6: Additional CD Diagrams

- Panel A (F1-Score): Minor discrepancy — reproduced CD diagram shows slightly different statistical grouping boundary
- Panel B (Accuracy): Reproduced accurately ✅
- Panel C (Balanced Accuracy): Reproduced accurately ✅

---

## Root Cause — CONFIRMED (2026-04-20)

**Issue A root cause identified.** Running metric computation directly from the local `causal_dfs_TEST.pkl` reproduces the paper's TD2C values **exactly**:

```
Accuracy: 0.8533 ± 0.0972 | BA: 0.8218 ± 0.1344 | F1: 0.6306 ± 0.2126 | Prec: 0.5637 | Rec: 0.7708
```

The local pkl (mod date July 27, 2025) is the original file. The verifiers' discrepancy comes from downloading `data.zip` from Google Drive, which was likely re-uploaded during round #518 debugging with a regenerated `causal_dfs_TEST.pkl` containing slightly different D2C model probabilities.

**Fix for Issue A:** Re-create and re-upload `data.zip` from the local (original) data folder.
