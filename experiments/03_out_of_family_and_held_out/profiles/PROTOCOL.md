# PREREG_K07X_DEV_v1 — K07 cross-architecture generalisation: frozen development protocol (2026-11-10; written before any K07-X run)

**Scope and limits**
- D, M2-R and CTX-A finish untouched. No Penguin extension.
- No leaderboard information is used for any choice.
- No early stopping on outcomes.
- No change to the held-out cohort.

## 1. Read-only root-cause audit (RESULT_K07_ROOTCAUSE_AUDIT.json; p184_k07_rootcause.py)
Data: H2 (118 synthetic cells, all variants of the six parent families) and CAL1 (the 30 public training scenes). Descriptive only. Effects are relative to arm B; negative = lower loss.
- **Profile quality.**
  - In H2 the class profiles beat both B and G30 in classes C (−7.5 % vs B; −3.3 % vs G30) and H (−2.9 %; −1.8 %). In class M they equal G30.
  - In CAL1 the M-gate loses to B on building_train_1 (+3.8 %) and on the fairbanks station (+1.1 %). The single strongly negative scene is building_train_1/fairbanks_1, class C, +7.7 %.
  - **The C/E/H profiles look tuned to the six parents and fragile outside them.**
- **Reassignment timing.**
  - The M-gate (daily re-selection only after an M day 1) gains −2.5 % over the static class profile in M cells, and the gain grows with the number of switches.
  - Daily re-selection everywhere (PD5) or a trigger (TRIG2) costs +0.5 % to +1.0 % in classes C and H.
  - **The re-selection logic is not the weak part.**
- **Startup conditions.**
  - Steps 0–227 are identical for MG and B (0.0 %); day 1 (steps 228–515) is −0.2 %.
  - The whole effect accrues later (−4.0 %). **Startup is not the cause.**
- **Building-family dependence.**
  - MG beats B in all six parent families of H2 (−3.7 % to −7.1 %), with no trend in size, number of AHUs or number of zones.
  - The weakest family is train_1 (worst cell +3.1 %; MG−G30 only −0.8 %).
  - The public leaderboard (different, unseen buildings) reversed the sign: MG +2.1 % vs B.
- **Working hypothesis H-K:** the class-specific profiles for E/C/H do not transfer to new architectures. The robust components are the global profile G30 and the M-class daily re-selection.

## 2. Architectures and weather (K07X_ARCHITECTURE_MANIFEST.md / .json; outcome-blind)
- **Buildings:** 15 buildings from the official generator (DEFAULT_RANGES, official materials stratification, official scenario generator and acceptance check).
  - 5 are new draws of the shipped composite type. 10 are new footprint morphologies.
  - **These are not 15 independent families:** they form two morphology classes, and none is a variant of the six parents.
- **Split (sha256 rule):** 9 development (3 composite + 6 morphologies) and 6 held-out (2 composite + 4 morphologies never seen in development).
- **Technical failure:** building_52002 failed the generator's 20 m² zone acceptance check and was replaced by reserve 52005, in the same slot and with the same strips. This is recorded as a technical failure.
- **K07 support:** all 15 architectures ran a K07 smoke test (120 steps, 0 illegal actions).
- **Weather (official strips):**
  - Development: the 10 strips of the 5 training stations that the shipped training set does not use, 2 per architecture on different stations.
  - Held-out: **minneapolis** (never used in training or development), one cold-half and one warm-half strip per architecture.
- **Within-family secondary:** 3 never-used variants of the six parents (b2_48031 / train_2, b2_40154 / train_7, b2_40155 / train_8), crossed onto the same minneapolis strips.

## 3. Policies (candidate set, frozen)
| Name | Key | Role |
|---|---|---|
| **B** | k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75 | operational baseline. It is the author's AdaptiveColdPolicy with 6 changed constants, so it **differs from the untouched author default**: RAMP 0.0, COLD_SAT −0.7, HEAT_U_SCALE 1.0, KP 0.10, NIGHT_HEAT_DROP_K 2.0, MILD_SHIFT_K 0.5. The author default (LB 0.01582 per its author) is not in the set and is never called B. |
| **G30** | k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0 | strong global profile |
| **MG** | k07unif@mgatehot | frozen M-gate |
| **MGG30** (the one new candidate, from H-K) | k07unif@mgatehot-g30nonM | the M-gate with the E/C/H class profiles replaced by G30; the M profile and the M-gate are unchanged; before step 228 it runs B, like MG |

**Code:**
- Runner run_episode_k07_v9x.py = the frozen v8 runner plus that key suffix only, a table change; the other keys are code-identical.
- K07 file their_k07_policy.py (sha 1ff8ad91…) and constants unchanged.

## 4. Development (72 runs) and selection
- **Design:** 9 architectures × 2 strips × 4 policies (K07X_DEV_MANIFEST.csv).
  - All 4 policies of a cell run on one host and runtime.
  - Kaggle: 5 kernels, cost-balanced allocation 3/3/4/4/4 cells.
  - Kaggle runs use KRE-v2 (PR #1 tracker overlay, certified bitwise) only after the M2-R kernels have finished **and** a K07 parity check on KRE-v2 has passed (1 development architecture × B and MGG30 × 1500 steps, bitwise against the laptop).
  - If the laptop or host-B take part (laptop after CTX-A, host-B after D and a parity check), they run whole cells.
- **Selection (p185_k07x_dev_select.py, self-tested):**
  - J_P = R_P + 0.5·max(0, A_P).
  - R_P = pooled relative loss vs B over the 18 cells.
  - A_P = the worst architecture's pooled relative loss vs B (the adverse-building term).
  - J_B = 0. S = argmin J; ties within 0.05 pp go to the earlier of B, G30, MG, MGG30.
- **Grouped cross-architecture validation:** leave one architecture out, select on the other 8, and score the choice on the left-out architecture. Report CV mean, CV worst, the number of negative folds and selection stability.
- **Stop/go gate (claim "a candidate beats the operational baseline across architectures"):**
  - GO requires all of: S ≠ B; CV mean < 0; at least 6 of 9 folds negative; CV worst ≤ +2 %; S chosen in at least 6 of 9 folds.
  - NO-GO: recorded as "no reliable candidate". No redesign is made or tested.
- **Failures:** a failed run is retried once with identical files. A cell that fails twice is reported and excluded from selection; it is never replaced by an easier one.

## 5. Held-out and within-family (frozen after selection, before any held-out run)
- **Freeze:** the policy S, its settings, code hashes, the held-out cohort and the analysis script are frozen in a separate note before the first held-out run.
- **Held-out:** 6 architectures × 2 minneapolis strips × 4 frozen policies = **48 runs**. The same 4 frozen policies run under GO and under NO-GO; nothing new is designed.
- **Primary:** S − B on the held-out architectures.
  - Reported: pooled relative change; per-architecture changes (sign count over 6 architectures and 12 cells); a percentile bootstrap over architectures (6 clusters, coarse, stated as such); the worst architecture.
  - Under NO-GO, S = B and the primary is reported as "no candidate selected".
- **Secondary:** S vs G30 and vs MG; all policies vs B; comfort / energy / carbon; worst building; composite vs new-morphology class.
- **Within-family (18 runs):** 3 parent-family variants × the same 2 minneapolis strips × 3 policies. The policies are {S, B, G30}, or {B, G30, MG} if S is B or G30.

## 6. Feasibility, compute, deadline
- **Compute:** smoke speeds give about 30–90 min per full run on a Kaggle worker; the large footprints cost up to 1.9 GB per run.
  - Development: about 2–4 h on the 5 Kaggle kernels after M2-R (~03:30).
  - Held-out + within-family (66 runs): Kaggle, laptop and host-B after selection.
- **Deadline:** all K07-X runs and integrity reports by 12 Nov; the rest of the time goes to the paper.
- **Blockers:** an unsupported architecture or a failed execution-equivalence check is recorded as a feasibility blocker. No silent replacement.
