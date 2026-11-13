# Prediction of the leaderboard check of the neural controller (PREDICTION_AX_LB_v1)

**Registered 2026-11-11 22:58:31 EET** (time from `date`), before either submission (AX, AXS), by the rule PREDICTION_RULE_AX_LB_v1, hashed at 21:39. No leaderboard score of AX or AXS exists.

## Basis: p279, six official training scenes
The plain competition kit was used, with 8640 steps per run.
- **Pooled gain of the layer:** g_off = +0.138 %.
- **AX's pooled weighted shares:** comfort 0.025, energy 0.435, carbon 0.540.

| scene | gain (%) | AX comfort share | AX unoccupied E+C share | learned starts | pressure overrides |
|---|---|---|---|---|---|
| building_train_1/denver_7 | -1.43 | 0.142 | 0.196 | 8 | 0 |
| building_train_1/fairbanks_1 | -0.27 | 0.019 | 0.385 | 22 | 0 |
| building_train_1/phoenix_5 | +0.89 | 0.024 | 0.161 | 2 | 560 |
| building_train_1/singapore_8 | -0.03 | 0.000 | 0.222 | 1 | 0 |
| building_train_2/fairbanks_2 | +1.52 | 0.034 | 0.550 | 15 | 0 |
| building_train_4/phoenix_8 | -1.19 | 0.058 | 0.212 | 4 | 0 |

## Prediction
- **Rule:** |g_off| < 0.5 %, so the category is **unchanged**.
- **Prediction:** |ΔLB| < 0.5 % (the leaderboard score of AXS within 0.5 % of that of AX).
- **Reading by the account of Section 5.3:** the network leaves only 2.5 % of its loss in comfort on these (in-sample) buildings, so the comfort mechanisms of the layer have little to correct.

## Parity of the two notebooks before submission
Official gateway against the research runner, bitwise in continuous and discrete actions (dry_ax/PARITY_AX.txt):
- building_train_1/fairbanks_1 and building_train_4/phoenix_8, 1800 steps, both arms;
- building_train_1/phoenix_5, 2400 steps (AXS), covering the first pressure overrides at steps 2289–2293.

All passed.
