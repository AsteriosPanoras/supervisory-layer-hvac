# Data

`episodes/` holds the per-episode results of the main tests, one row per episode and arm.
- `profile_study_2021_episodes.csv`: 2021 weather, 118 episodes (98 main + 20 extension) × 6 arms (B, select-once PS5, daily re-selection PD5, persistence TRIG2, the one-time gate MG, G30).
- `blind_set_episodes.csv`: blind set, 12 unseen architectures × 3 weather windows, every arm of K07, ZT Best and Penguin.

## Columns

**`loss`** is the episode's mean absolute reward over its 8,640 five-minute steps (30 days); lower is better.
- For the 2021 file it is `-mean(reward)`, as in the pre-specified analysis.
- All rewards are non-positive, so the two definitions coincide.

**Components.** `comfort_w`, `energy_w` and `carbon_w` are the mean absolute weighted components of the reward, with the competition weights 0.6 / 0.2 / 0.2. Their sum equals `loss`.

**Profile-study file:**
- `cohort`: H2-main or H2-ext;
- `station`: the cluster of the bootstrap;
- `weather_file_class`;
- `day1_observed_class`: the class the controller saw at its first decision;
- `daily_switches`.

**Blind-set file:**
- `controller`: K07, ZT Best, Penguin;
- `group`: composite or OOD;
- `day1_class`.

Arms on the blind set:

| Controller | Arms |
|---|---|
| K07 | B, G30, ROSFIX (fixed start), ROS (learned start), SUP (learned start + pressure reset), U (unoccupied setup), UR (setup + learned start) |
| ZT Best | ZB (alone), ZS (+ the unchanged layer) |
| Penguin | V0 (alone), PC (+ learned night precooling) |


## Statistics, as in the paper
- **Gains** are pooled: 100 (1 − Σ L_x / Σ L_ref).
- **Profile source:** intervals come from a 3,000-replicate station-cluster bootstrap (seed 1).
- **Learning source:** intervals come from a 3,000-replicate architecture-cluster bootstrap over the 10 composite architectures (seed 1). Every drawn architecture enters with all its episodes as often as drawn.

## Not included
**Third-party controllers.** The controllers are public Kaggle notebooks by their authors (Apache-2.0); the paper cites them, and our code imports them unchanged:
- K07 and ZT Best: Malyshev Danil;
- Penguin;
- the Residual PPO controller: alexchilton.

**The simulator.** It is the competition's Smart Buildings simulator. Episodes were run with its `evaluate.py`, whose sha256 is given in Appendix K.

The raw step-level logs (rewards, components, zone temperatures, decision logs) are several GB. They are not in this repository because of their size; they are available from the authors on request.

## Licence
Data and scripts of this package: Apache-2.0, as the code they build on.
