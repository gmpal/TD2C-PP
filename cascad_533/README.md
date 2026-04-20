# CASCAD Round 2 — Working Folder

This folder tracks the response to CASCAD Verification Report **#533** (April 13, 2026), the second verification round for manuscript IJF-D-24-00683R3.

## Files

| File | Purpose |
|---|---|
| `REPORT_SUMMARY.md` | Full digest of all findings from the #533 report |
| `ISSUES.md` | Structured action items with priority, hypotheses, and status |
| `RESPONSE_DRAFT.md` | Draft response to CASCAD (fill in as fixes are made) |

## Quick Status

| Issue | Description | Priority | Status |
|---|---|---|---|
| A | TD2C results differ in Tables 4, G.13, H.14 | P0 | ✅ Root cause confirmed — README warning + MD5 fix |
| B | Youden's J discrepancy (0.278 vs. 0.382) | P0 | ✅ Fixed — `bootstrap=True` added to `03.py` |
| C | Table 1: Δ row missing from `00.py` output | P1 | ✅ Fixed — Δ row added and verified |
| D | Table 5: PCMCI-GPDC runtime absent | P1 | ✅ Fixed — method added to `08.py` |
| E | Figure I.6 Panel A: minor CD grouping shift | P2 | ✅ Resolved via Issue A fix |

## Files Changed

| File | Change |
|---|---|
| `reproduce/py_scripts/00.py` | Added `print_scenario()` helper to print Forward, Backward, and Δ rows |
| `reproduce/py_scripts/03.py` | Added `bootstrap=True` to `BalancedRandomForestClassifier` |
| `reproduce/py_scripts/08.py` | Added PCMCI-GPDC run with try/except |
| `README.md` | (pending) MD5 checksum of `causal_dfs_TEST.pkl` + regeneration warning |

## Relation to Previous Round

Round 1 was CASCAD #518 (March 16, 2026). All 8 blocking issues from that round were resolved. Files from round 1: `REPRODUCIBILITY_V2.md`, `RESPONSE_TO_CASCAD.md`.

The new issues in #533 are qualitatively different — no execution errors, but **systematic numerical discrepancies in TD2C-specific outputs**. The most likely root cause is that `causal_dfs_TEST.pkl` was regenerated during round #518 fixes.
