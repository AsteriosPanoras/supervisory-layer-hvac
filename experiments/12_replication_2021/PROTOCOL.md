# H3 pre-registration v1: an independent replication of the 2021 gate test (H2)

Written and frozen 2026-11-12 12:57 EET (time from `date`), before any H3 run. Authors' decision: 12 Nov ~12:40, 40 cells.
The hashes of every file below are in `H3_PREREG_SNAPSHOT.sha256`, the append-only ledger `N10S_FROZEN_SHA256_v1_1.txt`, and an
OpenTimestamps proof (`H3_PREREG_SNAPSHOT.sha256.ots`).

## Why
- H2 (`PREREG_H2_v1`) supported the gate: MG minus TRIG2 = +0.504 points [0.337, 0.704] over its 98 complete main cells (42 stations).
- Its 20 pre-frozen extension cells gave -0.004 points; pooled over the 118 cells, +0.407 [0.210, 0.633].
- Read after the extension's outcome (post hoc, therefore not evidence): one extension cell (b2_47003, kansascity21_w0124, class C)
  holds 22.5 % of the extension's B loss and alone moves the extension's estimate by -0.28 points; per cell the gate gained in 10 of 20
  (mean +0.31 points).
- H3 asks the H2 question again on 40 new cells, with nothing chosen by outcome.

## Question
Does the one-time M-gate (MG) beat the sparse-persistence rule (TRIG2) on 2021 weather windows, and building-weather pairs, that no
earlier run used?

## Design (`make_design_h3.py`; it reads no reward)
- **Cells:** 40, with M 18, C 12, H 10 (the H2 proportions); 28 stations, at most 2 per station.
- **No overlap:** every window starts at least 35 days from every H2 window of its station and from the other H3 window of its station,
  so no H3 episode overlaps an H2 episode.
- **Windows:** the H2 eligibility and class rule, unchanged (Sunday 06:00 starts, exact-window QC, classes by T19); within a class the
  windows are spaced over the T19 range (the H2 rule). The script first checks that it reproduces all 120 H2 windows and their classes.
- **Buildings:** 20 GEN12 (the H2 set) and 20 GEN14 (the H2 variants minus b2_49016 and b2_49017, which the simulator cannot
  initialise), alternating over T19 within each class; no building is used twice.
- **Arms:** B, TRIG2 and MG with the frozen H2 keys (`h3_arms.json` = these three entries of `h2_arms.json`), runner
  `run_episode_k07_v8_mg.py` (sha256 6a3a32bb..., the H2 runner, unchanged). PS5, PD5 and G30 are not run: the primary does not use them.
- **Outputs:** `cells_h3.csv`, `cross_h3.csv`, `h3_arms.json`, `plan_h3_jobs.json` (120 jobs, cell-major).
- **Machines:** the authors' laptop (5 workers, cores 0-11) and Kaggle; both ran this runner in H2 after its QA gate. Every cell is a
  matched bundle on one machine.
- **Scheduler:** `chain_pull_v2.py` (the H2 scheduler with a settable worker count and a start wait; nothing else changed), plan
  `plan_h3_run.json`. The laptop starts when the graph-network rerun leaves it; Kaggle starts when experiment A has finished.

## Primary outcome and decision (fixed now)
- **Outcome:** MG minus TRIG2, in points of B-referenced gain, equal-cell weights (gain(a) = 100 (1 - sum L_a / sum L_B)), with the 95 %
  percentile interval of 3000 station-cluster bootstrap resamples (seed 1), over the complete H3 cells. This is the H2 primary,
  computed by `p296_h3_analysis.py` with the functions of `p128_h2_analysis.py`.
- **REPLICATED:** the lower bound is above 0 and the estimate is positive.
- **NOT REPLICATED:** the upper bound is below +0.25 points (half of H2's main estimate), so an effect of H2's size is excluded.
- **INCONCLUSIVE:** otherwise.
- The outcome is reported whatever it is, in the paper and in the repository.

## Secondary analyses (no multiplicity claim; reported whatever their sign)
1. The per-cell mean difference, with the counts of cells where MG is better, equal and worse.
2. By design class; by generator.
3. Influence: the largest cell's share of the B loss, and the primary without that cell (a sensitivity check, not a decision).
4. H2 and H3 pooled (up to 158 cells), same estimator; the loader first reproduces H2's registered +0.407 (checked before
   registration on the existing H2 data, with no H3 result present).
5. The share of cells where MG raised the gate, and design against observed class.

## Planning (`h3_power.py` -> `H3_POWER_PLANNING.json`)
If the 40 cells behaved like the 118 H2 cells (+0.407), 400 draws stratified like H3 give REPLICATED 67 %, INCONCLUSIVE 33 % and NOT
REPLICATED 0 %; the 5-95 % range of the estimate is +0.18 to +0.69. A smaller true effect lowers the first share. This is planning only:
the H2 outcomes were known before H3 was designed, and nothing in H3 was chosen from them except the size of the question.

## Missingness and stopping
- Complete case over the three arms.
- A run without a result is re-run once (H2 amendment 1). A building the simulator cannot initialise is excluded with all its cells.
  No replacement cells are added.
- No cell, arm, metric or rule changes after the first H3 run.
- The queue runs to completion. If it is not complete by Friday 13 Nov 10:00 EET, the result is reported on the complete cells and
  labelled incomplete. Cells are claimed in the fixed order, so an incomplete run drops cells from the end of that order, not by outcome.

## Limits
- Simulator only.
- The same 42 stations, year and six building families as H2: new windows and building-weather pairs, not new climates or families.
- Nominal intervals.
- 40 cells give about two chances in three of replicating an effect of H2's size.
