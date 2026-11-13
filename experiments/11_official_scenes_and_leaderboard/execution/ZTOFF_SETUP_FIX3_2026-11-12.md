# ZT official-scene stage: third setup failure, fixed by a real copy (2026-11-12 01:02:55, before any outcome)


**Fix 3:**
- $H/kit_trk/smart_buildings is a **real copy** of $PB/k07x2/trk_kit/smart_buildings (base tracker 87f69478).
- It includes a real copy of $PB/kit/smart_buildings/training_scenarios.
- This is the same layout as the laptop's tmp/trk_kit, which ran the paper's official-scene stages.
- Everything else is as in Fix 2: stage script, runner, ZT classes, overlay (run-time tracker 443aec77), scenes, p284 and the prediction rule.
