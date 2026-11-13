# ZT official-scene stage: second setup failure and the correct fix (2026-11-12 01:01:35, before any outcome)


**How KRE-v2 works:** it expects the competition's original tracker (87f69478) as the base and overlays the corrected tracker (PR #1, 443aec77) at run time. The original setting ($PB/k07x2/trk_kit) was therefore right about the tracker. Its only fault was the missing training_scenarios/ folder.

**Fix 2:**
- The kit is $H/kit_trk/smart_buildings/. It links every entry of $PB/k07x2/trk_kit/smart_buildings (base 87f69478) and adds training_scenarios, linked to $PB/kit/smart_buildings/training_scenarios.
- For the six scenes, floor_plan.npy, zone_map.npy and config.gin are byte-identical across all of these: the dataset copy (scen_official/official), $PB/kit, the laptop's tmp/trk_kit and $G0/sk.
- Weather and the building and baseline code are identical in $PB/kit and the trk_kit.
- The run-time tracker is the corrected 443aec77, as p284's frozen docstring states.

**Record:** the failed folder is kept as work_off_failed_kre_v2.
