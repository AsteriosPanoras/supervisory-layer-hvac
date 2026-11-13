# PREREG_K07SUPFIX_v1: does learning add value inside the full layer?

Written and frozen 2026-11-12 13:45 EET (time from `date`), before any SUPFIX run. Hashes: `K07SUPFIX_PREREG_SNAPSHOT.sha256`, the
append-only ledger `N10S_FROZEN_SHA256_v1_1.txt`, and an OpenTimestamps proof.

## Why
- The authors decided at 13:50 on 12 Nov, after a reviewer's point about the layer's gain on the blind set.
- That gain (+2.82 % over G30) combines a learned start with a feedback rule, the pressure reset.
- The value of learning (+0.95 %, learned over fixed start) was measured without the reset (ROS over ROSFIX).
- The reviewer asked for the comparison inside the full layer.

## Question
With the pressure reset on, does learning the start's time constant still add to the layer's gain on unseen architectures?

## Arms (the 36 blind scenes of the K07 test, 8640 steps)
- **SUP** (existing; K07X3 stage 2, Kaggle): learned start and pressure reset, key `k07sup@<G30>;ROS=1,TR=1`.
- **SUPFIX** (new):
  - the same key with `,ALPHA=0`, so the start's time constant stays at its prior of 1.0 h;
  - everything else is identical: decision rule, identification window, bounds, pressure reset;
  - same frozen runner `run_episode_k07_v9sup.py` (sha256 e9f12b1c...); no new code.
- **Existing arms for the secondary contrasts:** B and ROSFIX (K07X3H), G30 and ROS (K07X3 stage 1).
- **Files:**
  - setup `make_supfix.py`, which writes `exp_lap/`, the job list and `K07SUPFIX_MANIFEST.csv`;
  - stage `run_stage_laptop_supfix.sh`;
  - launcher `launch_supfix.sh`.

## Machine
- **Laptop:** the authors' laptop, 5 workers, cores 0-11.
- **Kit:** the KRE-v2 kernel runner v3 (f01f234c...), the competition-tracker kit and the scenario folder of the graph-network laptop runs on the same scenes.
- **Reproducibility check:** SUP ran on Kaggle, so the first two SUP jobs of its manifest (fixed before any run) are re-run on the laptop first.
  - They are compared bit for bit with their Kaggle originals. This check is descriptive.
  - The pairing is kept either way, and any difference is reported.

## Primary outcome and decision (fixed now)
- **Outcome:** SUP over SUPFIX, gain pooled over the episodes of the 10 composite architectures.
- **Interval:** 95 % architecture-cluster bootstrap, 3,000 replicates, seed 1 (the p289/p277b logic, in `p298_supfix.py`).
- **Decision:**
  - LEARNING ADDS VALUE INSIDE THE LAYER if the interval lies above 0;
  - NOT DETECTED if it contains 0;
  - LEARNING HARMS if it lies below 0.
- The outcome is reported whatever it is.

## Reported with it (no multiplicity claim)
- The exact Wilcoxon signed-rank test and the sign test over the 10 per-architecture gains.
- Better/worse counts; the OOD pair.
- The change of the comfort, energy and carbon terms.
- SUPFIX over G30, and SUPFIX over ROSFIX (the pressure reset on the fixed start).
- The comparison with learning without the reset (ROS over ROSFIX), in % and in points of B-referenced gain.

## Checks done before registration
`p298_supfix.py` reproduces the published contrasts from the frozen runs, with no SUPFIX result present (13:44):
- SUP over G30 = 2.82 %;
- ROS over ROSFIX = 0.95 %;
- SUP over ROS = 1.58 %.

## Missingness and timing
- **Failed runs:** a run that fails is re-run once on the same machine. p298 reports the primary only on the complete 36 scenes, and states any gap.
- **Timing:**
  - starts when the graph-network laptop stage ends (about 15:00);
  - lasts about 2.5 hours;
  - H3 uses the laptop afterwards (H3 execution note 1).

## Limits
- One controller (K07-G30) and the 10 composite architectures of one generator.
- A fixed prior of 1.0 h is one choice of non-learned constant.
