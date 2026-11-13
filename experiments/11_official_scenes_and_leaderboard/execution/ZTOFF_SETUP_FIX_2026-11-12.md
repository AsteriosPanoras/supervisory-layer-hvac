# ZT official-scene stage: setup failure and fix (2026-11-12 00:49:45, before any outcome)

- **First launch** (11 Nov 23:30, started 00:32 after the AXSUP END): all 12 runs exited in about 1.5 s, status NO_SUMMARY. The error was FileNotFoundError 'smart_buildings/training_scenarios/building_train_1/denver_7/floor_plan.npy'.
- **Cause:** the stage pointed at $PB/k07x2/trk_kit. That kit has the competition's original tracker 87f69478 and no training_scenarios folder. The official scene configs read their floor plans from smart_buildings/training_scenarios/.
- **Fix:** the kit is now $PB/kit. Its content manifest (smart_control, training_scenarios and weather; .py, .gin, .npy and .csv files) is 501959805db5, identical to the laptop kit $G0/sk, which has the corrected tracker 443aec77. This is the setting that p284's frozen docstring states (corrected tracker 443aec77, as in the paper's K07 official-scene stage).
- **Unchanged:** runner, ZT classes, overlay, scenes, analysis (p284) and the prediction rule.
- **Record:** the failed work folder is kept as work_off_failed_setup_v1.
