# Result of the leaderboard check of the neural controller (RESULT_AX_LB_v1, 2026-11-12 05:46:52)

**Scores (Kaggle, public leaderboard; LB_SCORES_2026-11-11_AX.json, checked 2026-11-12 05:45:50):**
- AX, Residual PPO unchanged (submission 58250728): **0.01053**;
- AXS, the same with the unchanged layer (submission 58250733): **0.01064**.

**Outcome by the registered rule** (PREDICTION_RULE_AX_LB_v1; ΔLB = 100 (1 − LB_AXS / LB_AX), positive means the layer is better):
- **ΔLB = −1.045 %.** Allowing for the 5-decimal rounding of both scores, ΔLB lies in [−1.140, −0.949] %.
- **Prediction** (PREDICTION_AX_LB_v1): unchanged, |ΔLB| < 0.5 %.
- **Verdict: FAIL.** The layer worsened the leaderboard score of the neural controller by about 1 %, outside the registered band, whatever the rounding.

**Context stated in advance, unchanged:**
- **The basis:** the six-scene official diagnosis (p279) gave g_off = +0.138 % (per scene from −1.43 to +1.52 %). It ran with the corrected tracker, not with the leaderboard's original one (ERRATUM_AX_LB_v1).
- **In-sample scenes:** the network was trained on official buildings 1, 2, 4, 5 and 7, so those scenes are in sample for it.
- **One hidden set:** the leaderboard scores one set of hidden scenes.
- **The leaderboard chooses nothing:** it was not used to choose anything.

**Reading (post hoc, not a registered test):**
- Under the account of Section 5.3, a layer that targets comfort gains little where the controller leaves little loss in comfort (AX: 2.5 % on the official scenes), while its costs in energy and carbon remain.
- A net loss there is consistent with that account, but the account did not predict its size or sign here.
- On the blind set the same layer gained 3.37 % on this controller, and two of ten architectures lost more than 1 %.
