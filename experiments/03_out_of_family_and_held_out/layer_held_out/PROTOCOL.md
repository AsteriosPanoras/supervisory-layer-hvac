# PREREG_K07X3H_v1 — the fresh set: native K07 (B) and the mechanism-only ROS (ALPHA=0) (written 2026-11-11 01:30:10; before any B or ROSFIX fresh run)

## Arms (runner run_episode_k07_v9sup.py, the same as stages 1-2)
- **B** = `k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75`.
- **ROSFIX** = `k07ros@<G30>;ALPHA=0`: tau fixed at 1 h; the k07ros branch of v9sup is bit-identical to v9ros (p218 F5).
- **Scenes:** the 36 fresh scenes, in the same three-kernel splits as stage 2 (hb0-2, hr0-2), on Kaggle.

## Analysis p241
Read only when all 72 runs are complete.
- **G30 vs B (selection transfer):** rule as for the K07-X held-out:
  - composite pooled < 0;
  - at least 6 of 10 composite architectures better;
  - no architecture worse than +2 %.
- **SUP vs B:** the whole chain.
- **ROSFIX vs G30:** the mechanism alone.
- **ROS vs ROSFIX:** online learning.

## Labels
Blind: no B or ROSFIX fresh outcome exists. The G30, ROS and SUP fresh outcomes are already known (stages 1-2).
