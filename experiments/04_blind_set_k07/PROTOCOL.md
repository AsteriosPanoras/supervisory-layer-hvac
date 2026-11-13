# PREREG_K07X3_FRESH_v1 — decisive blind test on the FRESH set
Written 2026-11-10 15:46:27 (from `date`; first version 15:46, whose arm keys a shell heredoc lost; rewritten before any run). Frozen before any run on the fresh scenes other than a 120-step smoke run.

- **Scenes:** K07-X3 (k07x3/K07X3_COHORT.json): 12 never-used official-generator architectures (10 composite + 2 OOD) × 3 strips (one COLD, one MILD and one HOT day-1 class) = 36 scenes.
  - Built outcome-blind, without reading any result.
  - The strips are new 30-day windows (NOAA 2022, 28 stations; stated deviation in k07x3/README.md).
- **Arms (stage 1, now), runner run_episode_k07_v9sup.py, Kaggle KRE-v2:**
  - G30 = `k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0`
  - ROS = `k07ros@<the same overrides>`: frozen, the same decision and learning code as the submitted official notebook. Official parity is bit-identical (dry/PARITY_ROS.txt), and the k07ros branch of v9sup is bit-identical to v9ros (p218 F5).
  - Kernel fk0 starts with a scenes-identity check: B, 120 steps, on building_52208 / raleigh22_w0612 (a HOT strip). It must be bit-identical to the local smoke run. If it is not, the Kaggle fresh runs are not used until the cause is found.
- **Arms (stage 2, only if decided later):**
  - TR and the combined ROS+TR, if TR passes its development gate;
  - Penguin native and the Penguin supervisor, if the Penguin supervisor passes its gate;
  - the cluster "freedom rules" follow-up (FOLLOWUP_CLUSTER_FREEDOM_RULES_2026-11-10.md).
  Each is a separate, recorded step.
- **Analysis (written before any outcome; run only when every stage-1 run is complete):** ROS − G30.
  - Primary: the composite pooled change, and the number of the 10 composite architectures that improve.
  - Also:
    - by weather class (C, M, H), pooled;
    - OOD separately;
    - per scene;
    - no architecture worse than +2 %;
    - ROS behaviour: extended mornings and the weather class of those days, learned tau, fallbacks.
  - No outcome is read before all 72 runs are collected.
- **Labels:** blind test of a frozen, non-optimised mechanism. Any leaderboard score is reported separately and never used for a choice.
