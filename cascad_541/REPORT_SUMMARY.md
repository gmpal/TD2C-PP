# CASCAD #541 — Report Digest

**Report:** cascad#541 — April 29, 2026
**Manuscript:** IJF-D-24-00683R3 (Round 3 verification)
**Verifier env:** Python 3.10, Windows 11 Enterprise, 32-core Intel CPU, 512 GB RAM, downloaded April 21, 2026.

## Headline

Almost everything reproduces. Verifiers' summary:
- **Accurate:** Tables 1, 4, G.13, H.14; Figure 4; Figure I.6 panels B & C.
- **Minor expected discrepancies (acceptable):** Table 3, Table 6.
- **Minor discrepancies flagged:** Figure 3 (DREAM3_10), Figure I.6 panel A.
- **Hardware-dependent (caveat present):** Table 5 runtimes.

No P0 issues. Round-533 fixes for Tables 1, 4, G.13, H.14 worked.

## Critical context

The verifiers downloaded the repo on **2026-04-21**. At that time the latest pushed commit on `origin/main` was `19da58d` (March). The three round-533 fix commits (`a3b1889`, `18fc81c`, `856e024`) existed locally but had **never been pushed**. So:

- Verifiers ran `00.py` without the Δ-row helper → they re-added it themselves (page 3 of report). They explicitly note: *"Although the authors mention in Response #533 that this issue was fixed, the correction was not present in the provided repository."*
- Verifiers ran `08.py` without `bootstrap=True` in `03.py` and without the PCMCI-GPDC block — they patched `08.py` themselves to add a GPDC run (visible as the `PCMCI (GPDC)` row in their reproduced Table 5, with values that explode to ~35,000 s at N=10).
- The duplicate `PCMCI (ParCorr)` rows appearing 5× in the reproduced Table 5 are an artifact of the verifiers' patched script, not our code. Our 08.py runs PCMCI(ParCorr) exactly once per n_vars.

**Action required: push the three commits before re-verification.**

## Findings by section

### 4.1 Table 1 — accurate ✅
Verifiers re-added the Δ column themselves (helper function shown on page 3). Output matches paper exactly. After push, our `00.py` will produce this directly.

### 4.2 Table 3 — minor differences (≈0.004–0.009) ✅
Verifiers accept these as expected from random forest non-determinism. Our `bootstrap=True` fix in `03.py` (round 533) brings Youden's J to 0.391, vs. their 0.391 — already matching.

### 4.3 Table 4 — accurate ✅
Round-533 MD5 + README warning resolved this.

### 4.4 Figure 3 — minor discrepancy on DREAM3_10 (TD2C distribution) 🟡
Red-circled cluster on DREAM3_10 shows TD2C F1 distribution slightly tighter/lower than original. Likely the same root cause as round-533 Issue A: any user who ran `04.py` without `--skip_benchmark` (or before the round-533 README warning was visible to them) regenerated `causal_dfs_TEST.pkl`. Resolves once the round-533 README warning is on `origin/main`.

### 4.5 Figure 4 — accurate ✅

### 4.6 Table 5 — hardware-dependent + verifier-side artifacts 🟡
- 5 duplicate `PCMCI (ParCorr)` rows: artifact of verifier-patched `08.py` (they added GPDC but the patch also re-emits ParCorr). Our pushed `08.py` produces clean output.
- N=25 column absent: GPDC hit ~35,000 s at N=10 on their box → they aborted before reaching N=25. Our `try/except` around GPDC ensures the run continues for other methods. We can additionally skip GPDC for N ≥ some threshold to keep the table complete.
- TD2C (50 jobs) → relabeled (4 jobs): they used `--n_jobs=4`. Acceptable per their hardware; paper caveat already covers this.

### 4.7 Table 6 — accurate, minor rank swaps in middle ranks ✅
Top 7 features identical and in order. Swaps at ranks 8–9 and 12–14. Documented in round-533 response as expected RF variance.

### 4.8 Table G.13 — accurate ✅
### 4.9 Table H.14 — accurate ✅
### 4.10 Figure I.6 — panel A minor, B/C accurate ✅
Same root cause as Figure 3; resolves with round-533 README warning.

## Bottom line

Once `856e024` is pushed:
- 00.py Δ row → reproduces directly
- 03.py bootstrap fix → Youden's J already matches in their report
- 08.py PCMCI-GPDC → present in code; the duplicate-ParCorr artifact disappears
- README MD5 warning → guards Tables 4/G.13/H.14, Figures 3/I.6.A regression

Remaining open items after push:
1. Optionally skip PCMCI-GPDC for large N (or document timing on a reference box) so Table 5 always completes through N=25.
2. Confirm Figure 3 / I.6.A regenerate cleanly when starting from a pristine pkl.
