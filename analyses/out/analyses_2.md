## A. Learned recovery constant (h)
| Arm | scenes with >= 2 informative mornings | tau min / median / max | safety freezes |
|---|---|---|---|
| LOS | 36/36 | 0.34 / 1.18 / 2.16 | 1 |
| SUP | 36/36 | 0.34 / 1.17 / 1.90 | 1 |

Per-architecture mean tau (SUP): 52200: 1.45, 52201: 1.01, 52203: 1.14, 52205: 1.71, 52206: 0.82, 52207: 1.11, 52208: 1.40, 52209: 0.69, 52211: 0.60, 52212: 1.47, 52300: 1.39, 52301: 1.28

## B. Activity vs gain (SUP vs G30, per scene)
- scenes where the supervisor never acted: 4 (gain range [0.0, 0.0])
- scenes where it acted: 32 (mean gain 2.29 %; better 29; worse than -0.1 %: 1)
- Spearman: T&R override share vs gain 0.64; LOS extensions vs LOS gain 0.60
- by first-day class: C: extensions 10.9/scene, T&R share 2.4 %; M: extensions 0.3/scene, T&R share 3.4 %; H: extensions 0.7/scene, T&R share 11.9 %

## C. Overhead
| Arm | policy ms/step median (max) | wall s median |
|---|---|---|
| G30 | 2.02 (4.95) | 2917 |
| ROS | 1.97 (4.80) | 2910 |
| SUP | 2.02 (5.27) | 3001 |
