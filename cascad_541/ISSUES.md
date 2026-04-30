# CASCAD #541 — Issue Tracker

**Report:** cascad#541 — April 29, 2026
**Manuscript:** IJF-D-24-00683R3
**Status:** 🟡 In progress

---

## Priority Classification

| Priority | Issues | Impact |
|---|---|---|
| **P0 — Blocking** | (none) | All major tables reproduce |
| **P1 — Action needed** | A | Round-533 fix commits not on `origin/main` — verifiers ran stale tree |
| **P2 — Investigate** | B, C | Figure 3 DREAM3_10 minor; Table 5 N=25 missing |
| **P3 — Acceptable** | D, E, F | Verifier-acknowledged minor variance |

---

## Issue A — Round-533 fix commits never pushed (PRIMARY)

**Severity:** High (process), Low (technical) — every other #541 finding traces back to this.

**Finding:** On 2026-04-21 verifiers cloned `origin/main` at commit `19da58d`. The three round-533 fix commits live only in the local working tree:

```
856e024 add: CASCAD Round 2 response and verification tools     [unpushed]
18fc81c fix(03.py, 08.py): add fixes for CASCAD Round 2         [unpushed]
a3b1889 fix(00.py): add print_scenario helper ...               [unpushed]
```

**Evidence in report:**
- Page 3: *"Although the authors mention in Response #533 that this issue was fixed, the correction was not present in the provided repository; the response only explains what should be done."*
- Reproduced Table 5 contains `PCMCI (GPDC)` row but old `08.py` at `19da58d` has no GPDC code → verifiers manually patched it.
- Reproduced Table 5 contains 5× `PCMCI (ParCorr)` rows → verifier-patched script artifact, not our code.

**Fix:**
- [ ] `git push origin main` (3 commits ahead).
- [ ] Notify CASCAD that the fixes are now on `main` and request a re-pull or quick re-run.

**Status:** 🔴 Open — push pending user approval.

---

## Issue B — Figure 3 panel: DREAM3_10 TD2C distribution slightly off

**Section:** 4.4
**Severity:** Low — minor cosmetic shift in box plot.

**Hypothesis:** Same root cause as round-533 Issue A. User may have regenerated `causal_dfs_TEST.pkl` (or its DREAM3 equivalent in `realistic/`) by running `04.py` without `--skip_benchmark`, or by starting from a pkl that pre-dates the MD5 lock.

**Fix:**
- [ ] After push, request verifier re-run with the README MD5 check on the realistic-data pkl as well, or extend the README warning to cover `realistic/dream3/` cached files.
- [ ] Verify locally: regenerate Figure 3 from the canonical pkl and diff against the paper figure.

**Status:** 🟡 To investigate post-push.

---

## Issue C — Table 5: N=25 column missing, PCMCI-GPDC ~35,000 s at N=10

**Section:** 4.6
**Severity:** Low — runtime-only, paper has hardware caveat.

**Cause:** GPDC scales poorly. On the verifier's 32-core box at N=10 it took 9.7 hours; the run was almost certainly aborted before reaching N=25. Our round-533 fix wraps GPDC in `try/except` but does not bound runtime.

**Fix options:**
- [ ] Skip GPDC for N above a threshold (e.g. N > 10) with a clear printed note, so the rest of Table 5 always completes.
- [ ] Or document expected GPDC walltime per N in README.
- [ ] Decide: skip-with-note is the simpler fix and matches the paper's own treatment of GPDC.

**Status:** 🟡 Pending decision.

---

## Issue D — Table 5: duplicate PCMCI (ParCorr) rows, TD2C (50 jobs) → (4 jobs)

**Section:** 4.6
**Severity:** None — verifier-side script modification.

**Finding:** Reproduced Table 5 shows `PCMCI (ParCorr)` 5 times. Our `08.py` runs `PCMCI(..., ci="ParCorr")` exactly once per `n_vars`. The duplicates come from the verifier's manual patch to add GPDC into the old `19da58d` script. Resolves automatically when they re-run our pushed `08.py`.

`TD2C (50 jobs)` relabeled to `(4 jobs)` because they invoked `--n_jobs=4`. Acceptable.

**Status:** ✅ Resolves on push (Issue A).

---

## Issue E — Table 3 minor differences (≈0.004–0.009)

**Section:** 4.2
**Severity:** None — verifiers explicitly accept.

Verifiers' Youden's J = 0.391 matches our round-533 fix output exactly (0.391 ± 0.265). This means `03.py` `bootstrap=True` fix was effectively reproduced (likely by their own re-run after seeing the response, or by RF default behavior on their imblearn version). Other three metrics within 1–2%.

**Status:** ✅ Acceptable.

---

## Issue F — Table 6 rank swaps at positions 8–9 and 12–14

**Section:** 4.7
**Severity:** None — verifiers explicitly accept.

Top 7 features match in order. Middle-rank swaps are RF-variance artifacts. Already documented in round-533 response.

**Status:** ✅ Acceptable.

---

## Change Log

| Date | Action | File | Status |
|---|---|---|---|
| 2026-04-30 | Created issue tracker | `cascad_541/ISSUES.md` | ✅ |
| 2026-04-30 | Confirmed round-533 commits unpushed via `git log origin/main..HEAD` | — | ✅ |
| 2026-04-30 | Verified old `08.py` at `19da58d` has no GPDC → 5× ParCorr rows are verifier patch | — | ✅ |
| TBD | Push round-533 commits to `origin/main` | — | ⏳ |
| TBD | Decide: cap PCMCI-GPDC for large N in `08.py` | `08.py` | ⏳ |
| TBD | Verify Figure 3 / I.6.A regenerate from canonical pkl | — | ⏳ |
