# PREREG_K07ROS_v1 — recovery-learning optimal start (ROS) on the robust base G30 (exploratory; INTERNAL until results)
Written 2026-11-10 at 12:49 (time from `date`). Frozen before any ROS run of more than 864 steps and before any ROS outcome was read.
**Design:** k07fb_design/DESIGN_PROPOSAL.md (internal). It was built on development data only and blind to every held-out outcome. It uses a public principle: optimal start / recovery-time prediction (Seem, Armstrong & Hancock 1989; ASHRAE Guideline 36).

## Mechanism (key `k07ros@<G30 overrides>`; runner run_episode_k07_v9ros.py, a new file; v9s unchanged)
- **Base:** G30, unchanged. Only the weekday morning start of K07's conditioning (PRESTART_H / `_base_prestart`) can change.
- **Observed at run time:**
  - D(t), the mean zone deviation from the comfort band (two-sided, K);
  - the time of day and day of week from the observation;
  - K07's own conditioning state of the previous step.
- **Learned online in the building:** the recovery time constant τ. Each weekday it takes D at the start of conditioning and one hour later; informative mornings only (D_start ≥ 0.3 K and decreasing). τ_day = 1 h / ln(D_s/D_1h), and ln τ ← 0.5·ln τ + 0.5·ln τ_day, clipped to [0.3, 3] h.
- **Decision:**
  - At every 5-minute step of the window [opening − 4 h, opening − 2 h), conditioning starts as soon as τ·ln(max(D, D*)/D*) ≥ the time left to the opening. Otherwise K07 starts at its own 2 h, exactly as G30.
  - The lead stays at most 3 h until 2 informative mornings exist.
  - The start is reset at noon.
- **Fixed, not optimised:**
  - D* = 0.3 K: the reward's comfort / energy exchange rate (section 1.2 of the design). It is NOT calibrated.
  - τ0 = 1.0 h: the cross-building prior, the median of the development architectures.
  - LMIN = 2 h. The rule never shortens G30's lead, because in the development data shortening the lead to 1 h raised the loss in 7 of 8 buildings.
  - LMAX = 4 h: K07's own extreme-cold prestart.
  - α = 0.5.
- **Fallback:** if on 2 consecutive extended mornings D at the opening exceeds the model prediction by more than 0.5 K, the policy returns to G30's schedule for the rest of the episode.

## Checks before the runs (p214; no reward read)
- **F1:** k07@G30 by v9ros is bit-identical to v9s.
- **F2:** ROS with an empty window is bit-identical to G30.
- **F3:** ROS equals G30 until its first change. Conditioning turns on at the logged start step, and the logged D equals D recomputed from the zone temperatures.
- **F4:** the logged starts are the first crossings of the rule, and the start setting changes only at starts and at noon resets.
- **Kaggle parity:** the first ROS kernel starts with a parity run of 864 steps on building_52103 / fairbanks_3, which must be bit-identical to the laptop fidelity run.

## Runs
- ROS on all 30 K07-X v2 scenes (18 development + 12 held-out), 8640 steps. ROS is paired with the existing B, G30, MG and MGG30 runs of the same scenes.
- **Hosts:** laptop, host-B and Kaggle (KRE-v2). K07 parity is certified on all three hosts, and the new code path is deterministic.
- **Retries:** a failed run is retried once. A scene with a missing run is reported and left out.

## Analysis and gates
1. **Development gate (opened first):** ROS − G30 per development architecture and in the composite mean, plus the frozen selection rule p205 (selrule/) with candidates B, G30, MG, MGG30 and ROS, in that tie order. **The held-out results are opened ONLY if, on development, the ROS − G30 composite mean is < 0 AND no development architecture is worse than G30 by more than 1 %.** Otherwise ROS stops there and is reported as not supported.
2. **Held-out (blind):** ROS − G30 on the 4 composite and the 2 OOD held-out architectures, and the frozen submission check p206.
3. **Descriptive:**
   - per scene; the number of extended mornings and the mean lead;
   - τ per building (learned vs the development values);
   - comfort / energy / carbon decomposition;
   - fallback activations.
4. **The knob screen arms PRE1 and PRE3** (10-day windows) are reported as context: the energy price of a fixed earlier or later start.

## Labels
- Exploratory. Designed after the K07-X development outcomes.
- The development architectures are not blind for ROS; the held-out architectures are. The design step never saw any held-out file. The experimenter had seen the B / G30 / MG / MGG30 held-out results, but did not change the design.
- Any leaderboard score of ROS is a consistency check, not independent confirmation.
