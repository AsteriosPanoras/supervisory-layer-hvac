# Erratum to PREDICTION_AX_LB_v1 and p279 (registered 2026-11-11 23:29:02 EET, before any AX/AXS leaderboard score exists)

**What was wrong.** p279's docstring and PREDICTION_AX_LB_v1.md say that the six-scene official diagnosis ran with the "plain competition kit (the code the leaderboard runs)". That is not correct:
- `ref_run_ax.sh` ran the research runner on `issue1448_g0_2026-10-20/sk`. Its `smart_control/utils/scenario_generator.py` is the CORRECTED tracker (sha256 443aec77bcfe…), the same code as the KRE-v2 overlay of every research stage.
- The competition's original tracker (87f69478712b…, as in the competition-tracker kit) is what the hidden leaderboard runs on Kaggle.
- `evaluate.py` is the same in both (947c968c…).

**What is unchanged.**
- The prediction rule (PREDICTION_RULE_AX_LB_v1), the diagnosis numbers (g_off = +0.138 %) and the predicted category (unchanged, |ΔLB| < 0.5 %).
- The gateway parity: notebook logic against the research runner, both on the same kit.

**Consequence.** The tracker difference changes billing, not the controllers' actions. The diagnosis is therefore a proxy computed with the corrected simulator code, not with the leaderboard's exact code. This is stated whenever the result is reported.
