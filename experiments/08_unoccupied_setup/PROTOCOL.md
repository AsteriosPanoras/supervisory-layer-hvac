# PREREG_K07UNS_v1 — K07 unoccupied setup (ASHRAE G36) with and without the learned optimal start

Written 2026-11-11 10:26 (time from `date`). Frozen BEFORE any UNS outcome was read. The fidelity runs (p246) are mechanics checks; no reward of theirs is printed or read.


## Why (official training scenes; diagnosis p244, `pgdiag/RESULT_OFFICIAL_LOSS_MAP.json`)
The diagnosis used the existing 30-day runs on six official training scenes (n10s/legit8640, shipped and corrected simulator):
- K07 (B) spends **12–25 %** of its loss on energy during unoccupied hours (nights and weekends) in the warm scenes; native Penguin spends 7–8 %.
- K07 spends 33–38 % on carbon during unoccupied hours in the cold scenes.
- Comfort is 0.5–6 % of the loss for both.

Cause: official buildings give the controller an unoccupied (eco) band of 289–298 K (16–25 °C). K07 follows it through its sensor band, so it cools the empty building down to 25 °C every warm night and weekend. Penguin uses its own wide band.

## Mechanism (public principles only)
- **Unoccupied setup (U).** ASHRAE Guideline 36 default unoccupied cooling setpoint is 90 °F = 305.4 K. Outside K07's own PI hours, each zone's cooling limit becomes max(its sensor/eco limit, 305.4 K). The heating side is unchanged: K07 already sets heating back by NIGHT_HEAT_DROP_K.
- **Learned optimal start (R = ROS, ours; Seem et al. 1989 principle).** ROS is unchanged from the frozen ROS/SUP runs. Its morning need D is the mean zone distance outside the comfort band, so it measures warm-up and cool-down alike, and it learns how early the morning recovery must begin.

## Runner
`run_episode_k07_v9supe.py` (sha256 `e1b7921f058fe200d1234f4b6480be88f6ceb9f7a16365583004658ff262df17`) is a NEW file: v9supd (`f7c7ab72…`) plus ONE optional `k07sup` parameter, UNS_COOL_K. Its default is 0, which is off and leaves every line as in v9supd.
- The diff is three hunks: parse the parameter, wrap `policy._zone_band` (cooling limit ≥ UNS_COOL_K only when `policy._pi_active` is False), and raise `eco_temp_window[1]` to ≥ UNS_COOL_K.
- Inside PI hours (K07's pre-start and occupancy, including any ROS extension) nothing changes.

## Arms
G = `RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0`

| Arm | Key | Meaning |
|---|---|---|
| G30 | `k07@G` | the base (LB 0.01477) |
| R | `k07ros@G` | G30 + learned start (LB 0.01477) |
| U | `k07sup@G;ROS=0,TR=0,UNS_COOL_K=305.4` | G30 + G36 setup; fixed 2-h morning start (K07's own PRESTART_H) |
| UR | `k07sup@G;ROS=1,TR=0,UNS_COOL_K=305.4` | G30 + G36 setup + learned start (**candidate**) |

Ablation logic:
- U − G30 = the fixed engineering rule.
- UR − U = our mechanism on top of the rule.
- (UR − U) − (R − G30) = what the learned start adds only because the setup exists (interaction).

## Fidelity (p246; development scenes 52010 phoenix_7 and 52103 fairbanks_3, 864/2016 steps)
- **F1:** for the SUP key, v9supe equals v9supd (all arrays and the supervisor log).
- **F2:** U vs G30 is identical up to the first differing action. At that step K07 is outside PI hours and some zone is above 298 K (otherwise the runs are identical).
- **F3:** UR vs U is identical up to the first ROS 'need' extension.

Nothing is launched unless F1–F3 pass.

## Sets and runs
1. **Development K07-X:** 18 cells (`K07X_V2_COHORT` split dev, 6 composite + 3 OOD architectures × 2), 8640 steps. G30 (results_k07x2) and R (results_k07ros) exist. U and UR are new and run on Kaggle (dataset sb26-k07uns-exp, private). UR goes first, and kernel d0 starts with a 288-step UR parity run on 52010 phoenix_7, compared bitwise with the laptop fidelity run's prefix.
2. **Official training scenes:** the six diagnosis scenes (bt1 denver_7, fairbanks_1, phoenix_5, singapore_8; bt2 fairbanks_2; bt4 phoenix_8), 8640 steps. All four arms are new and run on the laptop (cores 0–11, 6 workers), with the competition-tracker kit and the KRE-v2 overlay as on Kaggle. Order: G30+UR, then U, then R.
3. **Fresh K07-X3 (blind):** 36 cells, 8640 steps. G30 and R exist (results_k07x3). U and UR are new and run on Kaggle, **pushed only after GATE_PASS**.

## Disclosures
- The six official scenes informed the design (the diagnosis), so the official part of the gate is a design-set check, not a blind test.
- Development cells 52010 phoenix_7 and 52103 fairbanks_3 carried the fidelity runs; no outcome was read.
- The fresh set was read before for G30, R, SUP, SUP-W, SUP-W2 and B, never for U or UR.
- The setup value (305.4 K) is the G36 default and is not tuned. A second value would be a new pre-registration.

## Gate (p247, frozen now): UR vs G30
- **D (development):** mean over the 6 composite architectures of r_a < 0, where r_a = Σ L_UR / Σ L_G30 − 1 over the architecture's 2 cells, AND every development architecture ≤ +1 %, AND all 18 cells present.
- **O (official):** mean over the 6 scenes of r_s < 0, AND every scene ≤ +1 %, AND all 6 scenes present.
- **GATE_PASS = D and O.** Loss = mean |reward| over 8640 steps.
- Reported, never gating:
  - U vs G30, UR vs U, R vs G30 and the interaction, on both sets;
  - official component decomposition (energy / carbon / comfort × occupied / unoccupied);
  - ROS extensions by morning outdoor temperature (warm ≥ 11 °C vs cold);
  - the share of unoccupied steps with mean zone temperature > 298 K.

## Fresh (p248, frozen now; primary generalisation evidence)
Read only if GATE_PASS and all 36 U + 36 UR runs are complete.
- **Rule "UR generalises":** UR vs G30 composite pooled < 0, AND a majority of the changed composite architectures better, AND no architecture worse than +1 %.
- **Rule "mechanism adds":** the same three conditions for UR vs U.
- Also reported: U vs G30, R vs G30, the interaction, OOD, and by day-1 class (C/M/H).

## Submissions
If GATE_PASS:
1. Build official notebooks for UR and U from the frozen runner logic.
2. Check gateway parity bitwise against the runner.
3. Ask the authors; submit **two** submissions (U, then UR) so that LB(UR) − LB(U) isolates the mechanism. LB(R) = LB(G30) = 0.01477 is already known.

The LB never selects anything, and every score is reported.

## Labels
SB26 K07 UNS. These are simulator results, not leaderboard scores. Every file goes into the ledger `N10S_FROZEN_SHA256_v1_1.txt` (append only).
