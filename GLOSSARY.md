# Glossary

## Controllers
| Code | Meaning |
|---|---|
| K07 | The main controller: public and rule-based (Danil Malyshev). |
| B | K07's baseline configuration, tuned during development for a first-workday objective. |
| G30 | A stronger fixed K07 configuration, found by a separate search on 12 development episodes for the 30-day horizon. The comparator of the learning source. |
| ZT Best (ZB) | The K07 author's later controller. |
| Penguin (V0) | A public controller (penguin069), reproduced. |
| Residual PPO (AX) | A public trained neural controller (Alex Chilton): a tuned rule plus a residual policy trained with PPO. It has one shared network per entity type and no message passing. |
| GN | Our graph-network controller: a residual graph network on the building's own structure (zones, air handlers, plant), trained by imitation of Residual PPO. |

## Profile source (Section 3, 5.1)
| Code | Meaning |
|---|---|
| profile | A set of overrides of the controller's constants for one weather class. The library holds profiles for six building families. |
| C / M / H | Cold, mild, hot: the weather class of a day; the *initial* class is the one observed at the first decision. |
| PS5 | Select once: keeps the first profile. |
| PD5 | Daily re-selection: may reassign the profile every day. |
| LIST, WEEK | Reassign only on predefined days (sparse). |
| TRIG2 | Persistence: reassign only after a new class appears in two successive valid windows. The pre-specified comparator. |
| MG | The one-time gate: follows PD5 if the first observed class is mild and PS5 otherwise, and is never revisited. |
| pp | Percentage points of baseline-referenced gain, the unit of the profile study. |

## Learning source and rules (Section 3, 5.2–5.4)
| Code | Meaning |
|---|---|
| ROS | Learned optimal start: it learns the building's recovery time constant τ from its own mornings and moves the start within fixed bounds. |
| ROSFIX | The same rule with τ frozen at 1 h (the fixed start). |
| TR | Pressure reset: Guideline 36 trim and respond, a floor on the duct static-pressure command during occupied hours. |
| SUP | ROS + TR: the layer, i.e. the learning source. |
| PC, LNP | Learned night precooling (Penguin): lowers the unoccupied cooling ceiling before dawn on warm weekdays when outdoor air is cool. |
| U, UNS | Guideline 36 unoccupied cooling setup: outside occupied hours the cooling limit is at least 305.4 K (32.2 °C). |
| UR, URT | U + ROS; U + ROS + TR. |
| ZS, AXS, GNS | ZT Best, Residual PPO, the graph network, each with the unchanged layer. |
| AXR, AXT | Residual PPO with the learned start only, or with the pressure reset only. |
| OR | K07-G30 + layer with the oracle prior: τ starts at the value the same episode learned by its end. An upper bound, not a method. |

## Sets
| Term | Meaning |
|---|---|
| families, cohorts | The six building families of the profile library. Cohorts: 2021 (98 main + 20 extension episodes), 2022, 2023. |
| out-of-family, held-out | New architectures used in development, and architectures held out from it. |
| blind set | 12 unseen architectures × 3 weather windows of 2022 (cold, mild, hot first day): 10 with composite floor plans, 2 out-of-distribution (OOD). It was never used for design or selection. |
| official scenes | Six official training scenes, used for diagnosis as a proxy of the hidden leaderboard scenes. |

## Measures
| Term | Meaning |
|---|---|
| loss L | The episode's mean absolute reward over 8,640 five-minute steps (30 days); lower is better. |
| components | Comfort, energy, carbon, weighted 0.6 / 0.2 / 0.2 by the competition. |
| gain | 100 (1 − Σ L_x / Σ L_ref), pooled over the episodes of a set. |
| comfort share, headroom | Comfort component / total loss of the controller beneath. A rule that only removes comfort loss cannot gain more than this share. |
| intervals | Station-cluster bootstrap (profile study); architecture-cluster bootstrap over the 10 composite architectures (blind set); 3,000 replicates; Holm for secondary contrasts. |

## Internal experiment codes (folder and file names)
| Code | Experiment |
|---|---|
| H2 | Profile study, 2021 (01) |
| K07ROS, K07TR, K07SUP, PGXSUP | Development gates (02) |
| K07X, K07X2, K07X3H | Out-of-family and held-out (03) |
| K07X3 | Blind set with K07 (04) |
| ZTSUP | ZT Best maintenance test (05) |
| PGPC | Penguin precooling (06) |
| AXSUP | Neural controller (07) |
| K07UNS | Unoccupied setup (08) |
| GNNSUP | Graph network (09) |
| EXT12 | Extension: ablation, oracle prior, a third rule (10) |
| KRE-v2 | The corrected episode tracker of the starter kit (pull request #1), used for every episode |
| policy key | `k07sup@<constants>;ROS=1,TR=1` = controller and settings of one arm; job lines are `building weather policy_key steps` |
