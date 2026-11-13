# PREREG_K07X_DEV_v2 — design amendment to PREREG_K07X_DEV_v1 (2026-11-10; v1 is kept intact and is superseded only where stated; no K07-X performance run has been made)
The amendment follows the authors' review of 10 Nov. It changes the cohort balance, the held-out weather, the selection objective and the compute plan. The policies are unchanged.

## A. Cohort (K07X_V2_COHORT.json; outcome-blind rules in k07x_v2_cohort.py)
- **Composite pool:** the first 10 seeds from 52000 upward that pass the official 20 m² acceptance check.
  - Two official batches: 52000–52007 and 52008–52015.
  - Rejected by the generator: 52002, 52013, 52015.
  - Pool: 52000, 52001, 52003, 52004, 52005, 52006, 52007, 52008, 52009, 52010.
- **Composite split:**
  - Held-out: the 4 seeds with the smallest sha256('K07X2|HO|seed|…') — 52001, 52003, 52006, 52009.
  - Development: the other 6 — 52000, 52004, 52005, 52007, 52008, 52010.
- **OOD morphologies:** the 5 of 10 shapes with the smallest sha256('K07X2|OOD|shape|…').
  - Development: trapezoid 52101, t_shape 52103, oval 52105.
  - Held-out: semicircle 52108, h_shape 52102. These shapes never appear in development.
- **Development strips:** the 10 official strips of the 5 training stations that the shipped training set leaves unused, 2 per architecture on different stations.
  - Day-1 class mix (229-step probes, decision record only): C 2, M 5, H 3.
- **Held-out strips: common to all 6 held-out architectures and to the within-family set.**
  - minneapolis_1 (day-1 class C, T19 −1.42 °C) and minneapolis_5 (class M, 16.34 °C): the C strip and the M strip farthest from a class threshold.
  - minneapolis has no H strip. minneapolis_7 (24.19 °C) is excluded as borderline.
  - Minneapolis is used neither by the training set nor by development.
- **Within-family (secondary):** b2_48031 (train_2), b2_40154 (train_7), b2_40155 (train_8), crossed onto the same two strips.
- **Feasibility:** all 15 architectures pass the K07 smoke test (120 steps, 0 illegal actions).
- **Disclosure:** h_shape (52102) received only technical runs before this amendment: smoke, runner-equivalence and parity, all on a development strip. No performance number was read.

## B. Policies (unchanged; small hypothesis-driven set)
- The set: B (operational baseline, not the untouched author default), G30, MG (frozen) and MGG30.
- MGG30 is the M-gate with the E/C/H class profiles replaced by G30.
- **MGG30 is a hypothesis-driven development candidate defined after known negative public-leaderboard evidence (MG +2.1 % vs B on the public LB) and after the CAL1/H2 audit. Any future public-LB score of it is not independent confirmation.**
- No other policy, threshold or tuning.

## C. Selection (p187_k07x_dev_select_v2.py, self-tested incl. an OOD-guard case; replaces p185)
- **Unit:** the architecture. r[a,P] = pooled loss of policy P on architecture a (its 2 strips) relative to B.
- **Target = in-distribution composite.**
  - R_P = mean of r over the 6 development composite architectures (equal weight per architecture).
  - J_P = R_P + 0.5·max(0, worst composite r).
- **OOD guard (eligibility, not objective):** every development OOD architecture must have r ≤ +2 %.
- **Choice:** S = argmin J over eligible policies. B is always eligible with J = 0. Ties within 0.05 pp go to B, G30, MG, MGG30 in that order.
- **Grouped CV:** leave one development composite architecture out (6 folds). Select on the remaining 5 with the same guard, then score the choice on the left-out architecture.
- **GO requires all of:**
  - S ≠ B;
  - CV mean < 0;
  - at least 4 of 6 folds negative;
  - CV worst ≤ +2 %;
  - S chosen in at least 4 of 6 folds;
  - the OOD guard met by S.
- **Otherwise NO-GO:** the result is recorded as "no reliable candidate". No redesign.
- **Reported in either case:** the J table, r per architecture for both groups, folds, stability and the reason for each condition.

## D. Held-out (frozen after selection, before any held-out run; no held-out outcome is used for any choice)
- **Runs:** 6 architectures × 2 common strips × the 4 frozen policies = 48 runs.
- **Primary:** S − B on the **4 held-out composite architectures** (8 cells; in-distribution).
  - Reported: per-architecture changes, sign count, pooled change and a bootstrap over 4 architectures, stated as coarse.
- **Secondary, reported separately and never pooled into the primary:**
  - the 2 OOD architectures;
  - S vs G30 and vs MG;
  - comfort / energy / carbon;
  - worst building;
  - the within-family set (18 runs: 3 parent-family variants × 2 strips × {S, B, G30}, or {B, G30, MG} if S ∈ {B, G30}).
- **Under NO-GO:** the held-out set still evaluates the 4 frozen policies descriptively. Nothing new is designed.

## E. Compute and schedule (measured)
- **Runtime model:** 120-step smoke wall times of all 15 buildings, scaled by a Kaggle factor. The bracket is 0.38 (H2 K07 Kaggle runs, wall = 7.6 + 0.27·zones min) to 0.75 (Penguin M2 Kaggle vs laptop). The K07 parity kernel will calibrate it before development.
  - The new buildings carry larger floor-plan grids than the b2 variants with the same number of zones: up to 2.3 GB per run against 0.7 GB predicted.
- **Development:** 72 runs = 54–107 run-hours.
  - 5 Kaggle kernels × 4 parallel workers (the M2 / M2-R kernels show 4 parallel runs per kernel) give a per-kernel makespan of **2.6–2.9 h (optimistic) to 5.1–5.6 h (conservative)**.
  - The longest single run is 78–154 min.
  - Memory: 4 × 2.3 GB = 9.3 GB per kernel, out of about 30 GB.
  - Session: about 5.6 h at most, below the 11 h start budget and the 12 h limit.
- **Held-out:** 48 runs, 25–50 run-hours, 1.3–2.5 h on 5 kernels; the within-family runs (18 short ones) add about 0.5 h.
  - The laptop (after CTX-A, at most 4 workers) and host-B (after D, plus its own parity check) may take whole cells.
- **Correction:** v1's "2–4 h" covered only the optimistic half of this range.
- **Gate order:**
  1. M2-R complete.
  2. K07 parity on KRE-v2: development architecture t_shape 52103 / phoenix_2, B and MGG30, 1500 steps, bitwise against the local reference (already run).
  3. Authors' go for development.
  4. Development and selection.
  5. Freeze of the held-out analysis.
  6. Held-out and within-family runs.
