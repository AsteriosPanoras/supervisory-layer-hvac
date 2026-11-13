# PREREG_K07SUP_COMBINED_v1 — the combined supervisor (ROS + TR) on the robust base G30 (written 2026-11-10 16:21:04; before any combined run beyond the 864-step fidelity run)
- **Trigger (PREREG_K07TR_v1, analysis 4):** BOTH mechanisms passed their development gates.
  - ROS: 14:57, composite −0.52 %.
  - TR: 16:17, composite −1.33 %.
  The combined arm therefore runs, as preregistered.
- **Key:** `k07sup@<G30 overrides>;ROS=1,TR=1` (runner run_episode_k07_v9sup.py; fidelity F6 PASS: equal to G30 until its first change, both supervisors logged). No parameter changes.
- **Runs:**
  - (a) the 30 K07-X v2 scenes, locally after the Penguin supervisor runs on each host;
  - (b) the 36 fresh K07-X3 scenes on Kaggle (stage 2 of PREREG_K07X3_FRESH_v1), after the stage-1 kernels.
- **Analysis:**
  - (a) the development gate as before (composite mean < 0, no development architecture > +1 %), then the frozen selection p205 with candidates B, G30, MG, MGG30, ROS, TR, SUP, then the held-out SUP − G30 with the same rule;
  - (b) on the fresh set, SUP − G30 and SUP − ROS, read only when all fresh runs of the stage are complete.
- **Attribution (descriptive):** SUP − ROS ≈ the TR contribution; SUP − TR ≈ the ROS contribution; with the ablation, the online-training share.
