# PREREG_EXT12_v1: how the layer improves, what a better version could give, and adding a rule (12 November 2026)

**Status:** written and hashed BEFORE any run of experiments A, B or C.

- the selected experiments are A, B, C and E (E is registered separately, if it is attempted);
- deadlines: host-B until 09:00, laptop and Kaggle until 11:00; all of them if they fit, otherwise the important ones.

All three use the blind set (the 36 scenes of the Residual PPO, ZT Best and K07 blind tests, 8640 steps) and the unchanged layer code. Gains are pooled over the 10 composite architectures, with a 95 % architecture-cluster bootstrap (3000 replicates, seed 1; the p277b logic).

## A. Which rule gives the gain on the neural controller (host-B)
- **Arms:**
  - new: AXR = learned start only (k07sup@;ROS=1,TR=0); AXT = pressure reset only (k07sup@;ROS=0,TR=1);
  - existing: AX (k07@) and AXS (ROS=1,TR=1).
- **Setup:** same runner (run_episode_k07_v9supe_ax.py, 557d4aab), machine and kit as the Residual PPO blind test. 72 new runs.
- **Analysis:** p289_axabl.py.
- **Predictions:**
  - **A1:** AXR beats AX (pooled gain > 0).
  - **A2:** AXT beats AX (> 0).
  - **A3:** AXS gains at least as much as the larger of AXR and AXT.
  - **A4:** each single rule lowers the comfort loss (comfort component change < 0).

## B. What a better learned start could give: an oracle prior (Kaggle)
- **Arm OR:**
  - the K07-G30 + layer key of the K07 blind test (SUP), with TAU0 set to the final time constant that the same episode's SUP run learned (summary ros.tau_final_h; all 36 lie in [0.338, 1.898] h, inside the bounds);
  - everything else is identical to SUP;
  - this uses the episode's own future, so it is an **upper bound** on what a prior could give (for example a prior predicted from the building's structure), **not a deployable method**.
- **Comparison:** against the existing SUP runs (Kaggle, runner v9supe environment).
- **Reproducibility check (F-B):** 4 existing SUP jobs are re-run in the same Kaggle setting and must match their originals bit for bit. If they do not, the 36 SUP runs are re-run first and OR is compared with the re-runs.
- **Analysis:** p290_oracle_prior.py.
- **Predictions:**
  - **B1:** OR beats SUP (pooled gain > 0).
  - **B2:** OR's gain over SUP is larger over days 1-14 than over days 15-30. The prior matters while the layer is still learning (time-to-value: learned minus fixed was +0.19 pp at day 7 and +0.80 pp at day 14).

## C. Adding a third rule to the same template (Kaggle)
- **Arm URT:** SUP + the Guideline 36 unoccupied setup of the paper (UNS_COOL_K=305.4; the same value as the UNS blind test) = learned start + pressure reset + unoccupied setup.
- **Compared with:** the existing SUP runs. The same 4-job reproducibility check as B applies.
- **Analysis:** p291_three_rules.py.
- **Predictions:**
  - **C1:** the sign of URT − SUP equals the sign of UR − ROS, measured before on the same scenes (adding the setup on top of the learned start predicts adding it on top of both rules).
  - **C2:** URT lowers the energy + carbon loss relative to SUP (the term the setup targets).

## Reporting
Every result is reported whatever it is, with failed predictions named as failed. Nothing is chosen on the leaderboard.
