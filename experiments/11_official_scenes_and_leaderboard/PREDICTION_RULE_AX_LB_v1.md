# Rule for predicting the leaderboard check of the neural controller (PREDICTION_RULE_AX_LB_v1)

Written 2026-11-11 ~21:40 EET, before any run of the six-scene official diagnosis had finished and before any submission. The numbers are
filled in later, in PREDICTION_AX_LB_v1.md, by this rule only.

## The two submissions
- **AX:** the public Residual PPO controller exactly.
- **AXS:** the same with the unchanged layer.
- **Leaderboard change of the layer:** ΔLB = 100 (1 − LB_AXS / LB_AX), in %. Positive means the layer is better.

## Basis
p279: AXS against AX on the six official training scenes, with the plain competition kit (the code the leaderboard runs) and 8640 steps.
- g_off = the pooled gain of the layer over these scenes, in %.
- s_off = AX's comfort share.

These scenes are the closest available proxy of the hidden scenes. They were the K07 proxy too: comfort share 0.2–4.6 %, and the leaderboard did not change.

## Rule
| Official-scene gain | Predicted leaderboard change |
|---|---|
| \|g_off\| < 0.5 % | \|ΔLB\| < 0.5 % (unchanged) |
| g_off ≥ +0.5 % | ΔLB > 0 (the layer improves the score) |
| g_off ≤ −0.5 % | ΔLB < 0 (the layer worsens it) |

The prediction is reported as PASS or FAIL, with both scores, whatever the outcome. The leaderboard is never used to choose a policy.

## Caveats stated in advance
- The network was trained on official buildings 1, 2, 4, 5 and 7, so the official scenes are in sample for it. The hidden scenes may or may not be.
- The leaderboard scores one set of hidden scenes.
- On Kaggle, torch may be a different version from the local one, and the network's arithmetic can then differ slightly. Both submissions run in the same Kaggle environment.
