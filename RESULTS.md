# Results at a glance

Every test below was fixed before its outcome existed. **Failed** means that a rule or prediction fixed in advance did not hold; it is reported as such in the paper. Gains are pooled (see `GLOSSARY.md`). Intervals are 95 %.

## Profile source (2021 weather, Section 5.1)
| Test | Result | Status |
|---|---|---|
| One-time gate MG vs persistence TRIG2 (primary, 98 episodes) | +0.504 pp [0.337, 0.704]; MG reduced the loss of B by 4.47 % | **held** |
| Pre-specified 20-episode extension | −0.004 pp; pooled with the main sample +0.407 pp [0.210, 0.633], n = 118 | inconclusive (its interval, computed post hoc: [−0.45, +0.71]) |
| Registered replication on 40 new episodes (experiment 12) | +0.255 pp [−0.037, +0.709]; above zero in 95 % of the bootstrap resamples (post hoc) | **inconclusive** by its rule: just short of its criterion, a lower bound above zero |
| All 158 episodes pooled (registered secondary, experiment 12) | +0.366 pp [0.206, 0.535] | — |
| Frozen profiles outside their families | out-of-family development: the gate lost to G30 by 6.8 pp; held-out: G30 gained 4.1 % over B while the gate lost 0.9 % | profiles do not transfer |

## Learning source on the blind set (12 unseen architectures, Section 5.2)
| Contrast | Gain (%) | Interval | Better (of 10) | Status |
|---|---|---|---|---|
| Tuning: G30 vs B | +8.62 | [+6.01, +11.41] | 10 | — |
| **Layer: G30 + layer vs G30** | **+2.82** | [+1.48, +3.94] | 9 | **held** |
| Learned vs fixed start | +0.95 | [+0.11, +1.82], Holm p = 0.023 | 8 | — |
| Pressure reset over the learned start | +1.58 | [+0.58, +2.48] | 9 | — |
| Penguin + learned precooling vs Penguin | +1.88 | [+1.12, +3.07] | 10 | **held** (development gate passed) |
| ZT Best + layer vs ZT Best (maintenance test) | +5.71 | [+3.64, +7.55] | 10 | **held** |
| Residual PPO + layer vs Residual PPO | +3.37 | [+1.57, +5.22] | 8 | **failed** (two architectures lost 1.15 and 1.35 %) |
| Graph network + layer vs graph network | +3.55 | [+1.82, +5.40] | 8 | **failed** (52209 lost 1.17 %) |
| Residual PPO: pressure reset alone vs Residual PPO (experiment 10 A) | +2.99 | [+1.60, +4.53] | 10 | the four registered predictions of 10 A **held** |
| Residual PPO: learned start alone vs Residual PPO (10 A) | +0.49 | [−0.10, +1.03] | 7 | the start brought both losses of the layer |
| Learning inside the layer: learned vs fixed time constant, both with the pressure reset (13) | +0.94 | [+0.10, +1.82] | 7 | **held**: learning adds value inside the layer |
| Time constant known from day 1 (oracle) over the layer (10 B) | +0.02 | [−0.003, +0.057] | 6 (of 9 changed) | both predictions **held**; not deployable |
| Unoccupied setup added to the layer (10 C) | +0.38 | [−0.45, +1.49] | 5 | both predictions **held** |
| Layer on held-out architectures | −3.9 % pooled loss; 2 of 4 composite better | | | **failed** |

## When the layer helps (headroom, Section 5.3)
| Test | Result | Status |
|---|---|---|
| Gain vs comfort share, K07 (post hoc) | Spearman ρ = 0.80 | found |
| Registered on ZT Best | ρ = 0.93 held; mean gain above K07's as predicted (5.97 % > 2.69 %); lowest-share third within ±0.5 % failed (+0.78 %) | **2 of 3 held** |
| Registered on Residual PPO | ρ = 0.75 held; mean gain above K07's held (3.78 %); first week ranks the later gain held (ρ = 0.51); lowest third failed (+0.70 %) | **3 of 4 held** |
| Registered on the graph network | ρ = 0.74 held; lowest-share third +0.48 %, within ±0.5 %, held; gain above K07's held (3.96 % > 2.69 %); first week ranks the later gain held (ρ = 0.49) | **4 of 4 held** |

## Unoccupied setup (Guideline 36, Section 5.4)
| Test | Result | Status |
|---|---|---|
| Development gate (official scenes) | passed | **held** |
| Blind set, fixed rule | −0.62 % alone; +1.27 % with the learned start; one architecture lost 3.3 % | **failed** |

## Leaderboard (consistency check only, never used to choose)
| Submission | Score |
|---|---|
| K07 B / MG / G30 | 0.01517 / 0.01549 / 0.01477 |
| G30 + learned start / + pressure reset | 0.01477 / 0.01477 |
| G30 + unoccupied setup / + learned start | 0.01414 / 0.01415 |
| Penguin (reproduced) | 0.01004 |
| Penguin + learned precooling | 0.01001 |
| Residual PPO alone / + layer | 0.01053 / 0.01064: 1.0 % worse, against a registered prediction of no change (within ±0.5 %) |
| ZT Best alone / + layer | not submitted |
| Submitted but not used in the paper: G30 + layer with a weather-clustering gate (two versions); two Penguin variants from a separate study | 0.01477 / 0.01477; 0.01005 / 0.01005 |

Every submission, its exact notebook and how to reproduce its score: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Status
All experiments have finished (12 November 2026).
