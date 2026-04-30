# CASCAD Round 3 — Working Folder

This folder tracks the response to CASCAD Verification Report **#541** (April 29, 2026), the third verification round for manuscript IJF-D-24-00683R3.

## Files

| File | Purpose |
|---|---|
| `REPORT_SUMMARY.md` | Full digest of #541 findings |
| `ISSUES.md` | Structured action items with priority and status |
| `Verification Report Cascad 541CP.pdf` | Original report from CASCAD |

## Quick Status

| Issue | Description | Priority | Status |
|---|---|---|---|
| A | Round-533 fix commits never pushed to `origin/main` | P1 | 🔴 Push pending |
| B | Figure 3 DREAM3_10: TD2C distribution minor shift | P2 | 🟡 Investigate post-push |
| C | Table 5: N=25 missing; PCMCI-GPDC blows up at N=10 | P2 | 🟡 Decide on cap |
| D | Table 5: 5× PCMCI(ParCorr) rows | P3 | ✅ Verifier-side artifact, resolves on push |
| E | Table 3 minor diffs (≈0.004–0.009) | P3 | ✅ Verifier accepts |
| F | Table 6 rank swaps in middle ranks | P3 | ✅ Verifier accepts |

## Headline

No P0 issues — round-533 fixes worked for Tables 1, 4, G.13, H.14. The dominant action item is **administrative**: three round-533 fix commits exist locally but were never pushed to `origin/main`, so verifiers ran a stale tree (`19da58d`, March) and re-applied fixes manually. Most #541 anomalies are downstream of that.

## Relation to Previous Rounds

- Round 1: CASCAD #518 (March 16, 2026) → all 8 blocking issues resolved.
- Round 2: CASCAD #533 (April 13, 2026) → all 5 issues resolved in commits `a3b1889`, `18fc81c`, `856e024` (local only, **not pushed**).
- Round 3: CASCAD #541 (April 29, 2026) → this folder.
