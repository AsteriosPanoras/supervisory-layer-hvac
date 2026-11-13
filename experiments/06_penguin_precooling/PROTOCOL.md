# PREREG_PGPC_v1 — night pre-cooling supervisor on Penguin (PC). Written 2026-11-11 00:47:59; before any PC run beyond the fidelity runs.

## Authors (about 00:30-00:45)
- Penguin is to get "architecture + clustering", with its own parameter freedom.
- Look properly at how Penguin works before running anything.
- "Our mechanism works if you set the right target."

## Why (development data only; pgdiag/NOTE_PENGUIN_INTERNALS_2026-11-11.md)
**What Penguin already has:**
- online per-zone identification of its heat-balance model (bounded RLS);
- a critical-zone pressure reset;
- a one-step model-predictive search per AHU.

**Where its loss is:**
- comfort only about 5 %;
- cooling energy in mild/hot occupied hours 41 %;
- cold heating 26 %.

The earlier ROS-P failed because it buys comfort Penguin does not lack.

**Structural gap:** Penguin is myopic (one step). It never pre-cools; it lets unoccupied zones float up to 307 K and cools hard in its 60-min soon hour.

**Opportunity** (p238 -> RESULT_PRECOOL_OPPORTUNITY.json, 252 development M/H weekday nights):
- Many nights have outside air at least 2 K below the zones.
- Only some buildings are still above the comfort maximum at 05:00: 52105 phoenix_2 +1.3 K, 52005 denver_4 +0.5 K, 52007 atlanta_4 +1.1 K, 52008/52010 phoenix_7 +0.2-0.3 K.
- Others are already below it.

## Mechanism (key `pgx@sup@ROS=0,TR=0,PC=1`; runner run_episode_penguin_v3pc.py, a NEW file)
Public principle: night pre-cooling of the building mass with cool outside air (economizer only).

**Weather decides WHEN:**
- Cluster: the mean outside air the policy observed over the last 24 h is at least 11 C (the same boundary as K07). It is evaluated once per weekday night at 00:00.
- Per step: outside air (previous step) <= mean zone temperature - 2 K.

**The building decides WHETHER and HOW MUCH (learned online):**
- An EWMA (alpha 0.5) of the zone-mean excess over the comfort maximum at the first step of Penguin's own soon hour, on warm-cluster weekday mornings without pre-cooling.
- Pre-cool only if it is > 0.

**Action, weekday nights 00:00 to the soon hour:**
- Penguin's unoccupied upper band standby_maximum := max(hi - 0.5 K, max zone T - 0.25 K, lo + 0.5 K). It is gradual, so outside air can meet it without the chiller.
- Native 307 K at every other step.
- Penguin's own search chooses and records the actions, so its online identification stays consistent.

**Safety:** two consecutive pre-cooled mornings with the zone mean below the comfort minimum at the soon-hour start switch pre-cooling off for the episode.

**Parameters:** a priori (11 C as K07; 2 K economizer margin; 0.5 K below the maximum; 0.25 K per step; 00:00 window). Not tuned on any outcome.

## Fidelity (p239; development scenes 52103 fairbanks_3 and 52105 phoenix_2)
- **F1:** the static key by v3pc equals v3sup.
- **F2:** on a cold scene, every night is 'cold', there is no pre-cooling, and the run equals native Penguin.
- **F3:** on 52105 phoenix_2:
  - identical to native before the first pre-cooling step;
  - at every pre-cooling step, the exact value and economizer rules hold;
  - native value elsewhere.
- **F4:** every gate class replays from oat_c.

## Runs and analysis
**Development:** PC on the 18 development scenes (laptop 4 + host-B 14), after the SUP-W2 local parts. Sequencer pgpc_auto.py.

**p240 gate vs native V0:**
- composite mean r < 0 AND no development architecture > +1 %;
- also: per scene, component split, behaviour.

**Only if the gate passes (separate recorded step):** the blind fresh test on the 36 K07-X3 scenes, V0 vs PC. Rule as for K07:
- composite pooled < 0;
- a majority of the 10 composite architectures better;
- no architecture worse than +2 %.

## Labels
Exploratory. The development data motivated the lever (energy, not comfort) and the building gate; the fresh set is blind.

## Fidelity result (p239 -> RESULT_PGPC_FIDELITY.json, ALL_PASS, 01:21)
- **F1:** the static key by v3pc equals v3sup.
- **F2:** cold scene 52103 fairbanks_3: three 'cold' nights, no pre-cooling step, identical to native Penguin.
- **F3:** 52105 phoenix_2, 7 days:
  - all nights 'warm'; learned need +0.22 K;
  - pre-cooling from step 792 (the 3rd night), 180 steps;
  - identical to native before it;
  - the value and economizer rules hold at every pre-cooling step;
  - native elsewhere; not frozen.
- **F4:** 8 of 8 gate events replay.
