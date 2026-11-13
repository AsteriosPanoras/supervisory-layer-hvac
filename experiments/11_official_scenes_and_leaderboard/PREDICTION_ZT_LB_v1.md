# Leaderboard prediction for ZT Best with and without the unchanged layer (PREDICTION_ZT_LB_v1, 2026-11-12 01:22:40)

**Timing:** written before any ZT Best leaderboard notebook is built and before any submission.

## Rule
PREDICTION_RULE_AX_LB_v1, unchanged (registered 11 Nov ~21:40):
- |g_off| < 0.5 %: the leaderboard is unchanged;
- g_off ≥ +0.5 %: the layer improves the score;
- g_off ≤ −0.5 %: the layer worsens it.

Here it is applied to:
- **ZB:** ZT Best alone;
- **ZS:** ZT Best with the unchanged layer (key k07sup@;ROS=1,TR=1).

## Basis
p284 (hashed before the runs) on the six official training scenes, 8640 steps, all 12 runs OK (RESULT_ZT_OFFICIAL_p284.json).
- **g_off = −0.074 %** (pooled).
- **ZB's shares:** comfort 4.1 %, energy 42.9 %, carbon 53.1 %.

| Scene | Gain | ZB comfort share | Layer activity |
|---|---|---|---|
| train_1 denver_7 | 0.00 % | 0.9 % | none |
| train_1 fairbanks_1 | −0.39 % | 1.8 % | 4 learned starts |
| train_1 phoenix_5 | +0.61 % | 11.2 % | 1,097 pressure-override steps |
| train_1 singapore_8 | 0.00 % | 5.6 % | none |
| train_2 fairbanks_2 | 0.00 % | 3.5 % | none |
| train_4 phoenix_8 | 0.00 % | 0.9 % | none |

## Prediction
**UNCHANGED (|ΔLB| < 0.5 %).**

## Caveats stated in advance
- **Tracker:** as in ERRATUM_AX_LB_v1, the diagnosis ran kernel_runner_v3 with the KRE-v2 overlay, so the corrected tracker 443aec77 was used. The leaderboard runs the competition's original tracker (87f69478).
- **Coverage:** the leaderboard scores one set of hidden scenes.
- **Use:** never used to choose a policy.
- **Submissions** need the authors' yes.
