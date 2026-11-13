# K07 UNS: execution notes and disclosures (append only)

Companion to PREREG_K07UNS_v1.md. That file is frozen in the ledger and is not edited. Corrections and execution events are recorded here.


2. **First laptop launch failed (10:31:56).** All 24 official jobs exited in ~5 s with `FileNotFoundError: smart_buildings/training_scenarios/building_train_1/denver_7/floor_plan.npy`.
   - Cause: the official scene configs name their floor plan relative to the kit root, and the competition-tracker kit (`trk_kit`) has no `training_scenarios` folder.
   - Fix: one symlink, `trk_kit/smart_buildings/training_scenarios -> issue1448_g0/sk/smart_buildings/training_scenarios`. These are the same files the scenario folder links to. No code changed, and the KRE-v2 overlay check passes (96 .py files, 1 differs, as always).
   - The failed attempt is kept as `laptop/failed_attempt1_work_off_floorplan_path`. Relaunched 10:33; the 6 first jobs are running.

3. **Sequencer replaced (10:41:57), before any outcome.**
   - v1 (pid 2869326) had pushed d0..d4. It was stopped by PID.
   - v2 (`k07uns_auto_v2.py`) rebuilds its state from Kaggle (it recovered d0..d4 as RUNNING and pushed nothing again). It reads its slot limit from `MAX_ACTIVE` each cycle, so a Kaggle CPU slot can be left free for the official submissions.
   - The analysis, gate and fresh rules are unchanged.

4. **Official notebooks prepared before the gate** (`official_adapter/sup/`):
   - `k07_official_policy_template_uns.py` = the frozen SUPW2 template plus the UNS option and the ROS_ON / TR_ON switches; the diff is shown in the session.
   - `build_k07_official_uns.py` builds `k07_official_UNS` (305b8866…) and `k07_official_UNSROS` (cdafdcd0…).
   - Parity check `uns_parity.sh`: official gateway vs runner v9supe, building_train_1 phoenix_5, 2880 steps. Results go to `dry/PARITY_UNS.txt`; no reward line is kept.
   - Nothing is pushed or submitted without GATE_PASS and the authors' yes.

5. **Sequencer v3 (11:06:48), before any outcome.** v2 (pid 2877962) was stopped by PID; v3 recovered d0..d4 (RUNNING) and pushed nothing again. Change: once the gate passes, the fresh kernels go to the FRONT of the queue, ahead of the remaining development-U kernels (the least important part, kept for the end). MAX_ACTIVE was set to 2, so 3 Kaggle slots stay for the maintenance test (PREREG_ZTSUP_v1).

6. **Kaggle slots (11:1x).** The maintenance test runs on host-B, not Kaggle, so MAX_ACTIVE is now 4 (one slot kept for the two official submissions after the gate).
- 2026-11-11 15:14:59: MAX_ACTIVE 4 -> 3 temporarily, so that the two leaderboard submissions get a Kaggle slot; restored to 4 when both submission scripts end.
- 2026-11-11 15:42:24: both submission scripts ended; MAX_ACTIVE restored to 4.
