# GNN imitation data: training / validation split (fixed before any recording)

- Teacher: the public Residual PPO controller alone (key k07@), 2016 steps (7 days) from each scenario's start.
- Recording runs: run_episode_k07_v9supe_gnn.py in recording mode. The episode is the teacher's own, bit for bit; the smoke test checks this.

| Set | Scenes | Buildings |
|---|---|---|
| Training, held-out set of Section 4 (k07x) | 24: both weathers of 12 buildings | 52000, 52001, 52004, 52005, 52100, 52101, 52102, 52103, 52105, 52106, 52107, 52108 |
| Validation, k07x | 3: one weather each | 52003 atlanta_4, 52104 minneapolis_1, 52109 atlanta_7 |
| Training, official | 10 | train_1 fairbanks_1, phoenix_5; train_2 atlanta_3, fairbanks_2; train_4 denver_2, singapore_3; train_5 fairbanks_5, phoenix_1; train_7 atlanta_8, denver_5 |
| Validation, official | 1 | train_8 fairbanks_8 (building 8 was also never trained on by the teacher's author) |

**Never used:** the blind architectures of k07x3 (building_52200-52212 and 52300-52301).
