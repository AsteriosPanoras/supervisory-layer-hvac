## A. Blind set: tests (10 composite architectures)
| Contrast | Gain | 95% CI | p (bootstrap) | p (Wilcoxon) | p (sign) | better/worse of 10 | Holm (boot) |
|---|---|---|---|---|---|---|---|
| SUP_vs_G30 | +2.82 | [+1.53, +3.73] | <0.001 | 0.004 | 0.021 | 9/1 | (primary) |
| G30_vs_B | +8.62 | [+6.56, +11.02] | <0.001 | 0.002 | 0.002 | 10/0 | - |
| SUP_vs_B | +11.19 | [+8.37, +13.71] | <0.001 | 0.002 | 0.002 | 10/0 | - |
| ROS_vs_G30 | +1.25 | [+0.49, +1.79] | <0.001 | 0.020 | 0.109 | 8/2 | <0.001 |
| ROSFIX_vs_G30 | +0.31 | [+0.08, +0.52] | <0.001 | 0.027 | 0.109 | 8/2 | <0.001 |
| ROS_vs_ROSFIX | +0.95 | [+0.12, +1.53] | 0.013 | 0.049 | 0.109 | 8/2 | 0.016 |
| SUP_vs_ROS | +1.58 | [+0.63, +2.28] | <0.001 | 0.004 | 0.004 | 9/0 | <0.001 |
| SUPW2_vs_SUP | +0.21 | [+0.01, +0.36] | 0.008 | 0.125 | 0.125 | 4/0 | 0.016 |
| PC_vs_V0 | +1.88 | [+1.24, +2.92] | <0.001 | 0.002 | 0.002 | 10/0 | <0.001 |

## B. Robustness of the primary (layer vs G30)
- pooled +2.82 %; leave-one-architecture-out [+2.39, +3.15]; mean of architecture gains +2.45 %
- episodes better/worse/equal: [29, 3, 4]; worse than -0.1 %: 1

## D. Learning dynamics
- {'episodes': 36, 'informative_mornings_median': 13.0, 'mornings_to_within_10pct_median': 11.0, 'mornings_to_within_10pct_p75': 15.0, 'share_within_10pct_by_morning_5': 0.2777777777777778}

## E. Worst cases
- blind building_52300|memphis22_w0102 (C): gain -0.62 %, LOS extensions 11, T&R share 6.2 %, components {"energy": {"occupied": -0.24, "unoccupied": 1.68}, "carbon": {"occupied": -1.38, "unoccupied": 3.83}, "comfort": {"occupied": -3.27, "unoccupied": 0.0}}
- held-out building_52001|minneapolis_1 (composite): gain +1.23 %, LOS extensions 11, T&R share 0.07 %, frozen True
- held-out building_52001|minneapolis_5 (composite): gain +0.00 %, LOS extensions 0, T&R share 0.18 %, frozen False
- held-out building_52003|minneapolis_1 (composite): gain +6.46 %, LOS extensions 22, T&R share 5.17 %, frozen False
- held-out building_52003|minneapolis_5 (composite): gain +11.10 %, LOS extensions 0, T&R share 34.95 %, frozen False
- held-out building_52006|minneapolis_1 (composite): gain -0.14 %, LOS extensions 4, T&R share 0.00 %, frozen False
- held-out building_52006|minneapolis_5 (composite): gain +0.00 %, LOS extensions 0, T&R share 0.00 %, frozen False
- held-out building_52009|minneapolis_1 (composite): gain -0.34 %, LOS extensions 7, T&R share 0.00 %, frozen False
- held-out building_52009|minneapolis_5 (composite): gain +0.00 %, LOS extensions 0, T&R share 0.00 %, frozen False
- held-out building_52102|minneapolis_1 (ood): gain +0.71 %, LOS extensions 6, T&R share 0.00 %, frozen True
- held-out building_52102|minneapolis_5 (ood): gain +0.08 %, LOS extensions 0, T&R share 0.45 %, frozen False
- held-out building_52108|minneapolis_1 (ood): gain +0.48 %, LOS extensions 15, T&R share 0.00 %, frozen False
- held-out building_52108|minneapolis_5 (ood): gain +8.65 %, LOS extensions 0, T&R share 15.86 %, frozen False

## F. Occupied vs unoccupied (% of reference loss)
- {"SUP_vs_G30": {"energy_occupied": -0.07, "energy_unoccupied": 0.5, "carbon_occupied": -0.7, "carbon_unoccupied": 1.32, "comfort_occupied": -3.87, "comfort_unoccupied": 0.0}, "PC_vs_V0": {"energy_occupied": -2.28, "energy_unoccupied": 0.82, "carbon_occupied": -0.54, "carbon_unoccupied": 0.18, "comfort_occupied": -0.06, "comfort_unoccupied": 0.0}}

## G. Profile study secondary contrasts (station bootstrap, replicate p-values)
| Contrast | Δ (pp) | 95% CI | p (bootstrap) | Holm |
|---|---|---|---|---|
| MG-PD5 | +0.335 | [+0.200, +0.487] | <0.001 | <0.001 |
| MG-PS5 | +0.781 | [+0.441, +1.236] | <0.001 | <0.001 |
| MG-G30 | +1.230 | [+0.185, +2.325] | 0.025 | 0.044 |
| PD5-PS5 | +0.446 | [+0.070, +0.937] | 0.022 | 0.044 |
