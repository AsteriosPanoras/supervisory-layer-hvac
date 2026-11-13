# K07SUPFIX execution notes (companion to PREREG_K07SUPFIX_v1; the registration itself is never edited)

1. **2026-11-12 15:00:38, the first launch failed at start; no episode ran.**
   - At 14:58:59 the stage stopped at once with "scenario building_52200/buffalo22_w1113 is not in the scenarios dataset".
   - **Cause:** the stage script pointed to the K07-X v2 scenario folder (`kaggle_k07x_scen`, the default of the graph-network stage script it was copied from). The blind scenes are in `k07x3/kaggle_k07x3_scen`, which the graph-network blind stage passes explicitly.
   - No summary or reward file exists, and nothing was read.
   - **Fix:** `SCEN=$N/k07x3/kaggle_k07x3_scen`. Nothing else changes. The original script is kept as `run_stage_laptop_supfix_v1_wrong_scen.sh`, and the failed work folder and its SF_DONE are kept with the suffix `_failed_launch_1458`.
   - **Consequence:** the failure wrote SF_DONE = FAILED, so the H3 scheduler took the laptop at 14:59 (H3 note 1).
   - **Next:** SUPFIX runs on the first machine that frees up. Any change of machine is recorded in a further note before the run.

2. **2026-11-12 15:41:31, SUPFIX runs on host-B, before any SUPFIX episode has run.**
   - The laptop and Kaggle are taken by H3 and experiment A until about 19:15. The authors closes the experiments at 21:00.
   - The 38 registered jobs (2 SUP re-runs first, then the 36 SUPFIX jobs), unchanged, run on host-B with the same runner (e9f12b1c), kernel runner (f01f234c), competition-tracker kit and blind scenes as the neural-controller blind stage there. The blind scenes contain the 36 scenes; this was checked before launch.
   - **Load:** half of host-B's 32 logical cores (taskset 0-15), 13 workers (38 jobs then take 3 rounds, as with 16), nice 19, about 16 GB of its 23 GB free memory.
   - **Pairing:** SUP ran on Kaggle, SUPFIX runs on host-B. The two SUP re-runs on host-B are compared with their Kaggle originals, bit for bit, and reported (descriptive). The pairing is kept either way.
   - p298, the bootstrap and the decision rule are unchanged.
   - **Files:** `run_stage_host-B_supfix.sh`, `launch_supfix_host-B.sh`, log `laptop/SF_host-B.log`, outputs filed into `results_k07supfix` and `results_k07supfix_repro`.
