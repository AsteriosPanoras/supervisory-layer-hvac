# PREREG_K07TR_v1 — request-driven duct static-pressure trim & respond (TR) on the robust base G30 (exploratory; INTERNAL until results)
Written 2026-11-10 (time in the freeze note, from `date`). Frozen before any TR run of more than 864 steps and before any TR outcome was read.
**Design:** k07tr_design/DESIGN_PROPOSAL.md (internal; development data only; never opened a held-out file). It uses a public principle: ASHRAE Guideline 36 trim & respond of the duct static-pressure setpoint from zone requests.

## Why (development data, G30)
- **Rogue zones:** 9 of 18 scenes have 1-3 rogue hot zones (out of band on 52-100 % of occupied steps).
- **Saturated, airflow-limited:** their cooling PI is saturated (mean u 0.88-0.98). The supply air is already at its floor (285 K) on 76-100 % of those steps, while the duct pressure is at the simulator's 200 Pa floor on 93-100 % of PI AHU-steps. A fully open box therefore gets only about 31 % of its design flow.
- **Cost of a static fix:** a static pressure raise would cost 0.4-3.2 % of the loss in fan energy in every scene, including the scenes without rogue zones. Hence an adaptive, request-driven rule.

## Mechanism (key `k07tr@<G30 overrides>`; combined `k07sup@<G30 overrides>;ROS=1,TR=1`; runner run_episode_k07_v9sup.py, a new file)
- **Base:** G30, unchanged. The rule acts after K07 and writes only the static-pressure command of a running AHU in PI hours. It only raises the command, and never above 400 Pa.
- **Requests per zone:** a zone requests when its cooling PI output is ≥ 0.95 and it is above the comfort band; it counts 2 requests when it is more than 1 K above. A request is released below 0.85.
- **Per-AHU pressure update:**
  - with R > 0 requests: +15 Pa per request, at most +32 Pa per step;
  - with no requests: −12 Pa per step;
  - bounded to [200, 400] Pa;
  - reset to 200 Pa (= G30) at the first PI step of each day.
- **Unresponsive zones:** a zone that keeps requesting at 400 Pa for 2 h on 2 days stops counting.
- **Coarse guard:** an AHU overridden on more than 90 % of its PI steps on 3 consecutive days, without fewer requests, is frozen at 200 Pa.
- **Parameters:** the rates are Guideline 36 defaults, NOT tuned. SPMAX 400 Pa and I = 0 come from the development distribution of needed pressure. There are no weather classes and no per-building tuning; the rule adapts online per AHU, from that building's own zones.

## Checks before the runs (p218; no reward read)
- **F1:** existing keys are bit-identical to v9s.
- **F2:** with SPMAX = 200 the run is bit-identical to G30, and requests occur.
- **F3:** the run equals G30 until the first override. At the first override only the overridden AHUs' pressure columns differ, with value sp/1000 − 1.
- **F4:** an offline replay of the rule reproduces the logged requests and pressures.
- **F5:** ROS keys are bit-identical to v9ros.
- **F6:** the combined key equals G30 until its first change, and both supervisors are logged.
- **Kaggle parity:** the first Kaggle kernel starts with TR on building_52010 / singapore_5 for 864 steps, bit-identical to the laptop fidelity run.

## Runs
- TR on all 30 K07-X v2 scenes, 8640 steps.
- **Hosts:**
  - Kaggle takes the cheapest scenes as soon as slots are free.
  - The laptop and host-B take the rest after their ROS part.
  - host-B runs at half speed from 17:00 to 21:00.

## Analysis and gates (same structure as ROS)
1. **Development gate:** TR − G30 per development architecture.
   - The gate passes only if the composite mean is < 0 AND no development architecture is more than 1 % worse.
   - The frozen selection p205 is run with candidates B, G30, MG, MGG30, TR.
   - The held-out results are opened ONLY if the gate passes; the sequencer enforces this.
2. **Held-out (blind):** TR − G30. The rule is the composite pooled change < 0, TR better on ≥ 3 of 4 composite architectures, and no held-out architecture worse than +2 %.
3. **Descriptive:**
   - per scene and per AHU: override share, mean and maximum pressure, requests;
   - the share of rogue-zone steps removed;
   - the comfort / energy / carbon split.
4. **Combined ROS+TR:** run on the held-out scenes only if BOTH mechanisms pass their development gates. It is a separate step, recorded when it is decided.

## Labels
- Exploratory. The 9 development architectures are not blind for TR; the 6 held-out architectures are.
- The design step never saw held-out files. The experimenter had seen the B / G30 / MG / MGG30 held-out results but did not change the design.
- Any leaderboard score is a consistency check, not independent confirmation.
