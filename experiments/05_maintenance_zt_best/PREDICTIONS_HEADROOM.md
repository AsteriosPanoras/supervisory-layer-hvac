# Prediction registered before the maintenance-test outcome (PREDICTION_ZTSUP_HEADROOM_v1)

Addendum to PREREG_ZTSUP_v1. The PREREG itself is unchanged.


## State at registration
- **No outcome of the 72 maintenance runs has been read.** The runs are still executing on host-B. `results_ztsup_fresh/` does not exist and `ztsup/RESULT_ZTSUP_FRESH.json` does not exist.
- The laptop folders `ztsup/laptop/work_ztfid*` hold only the fidelity runs (RESULT_ZTSUP_FIDELITY.json, ALL PASS), not fresh-set episodes. The laptop and Kaggle ZT stages were not used (NOTES_ZTSUP_EXECUTION).
- The test script `paperpack/p258_headroom_test.py` was written before the outcome and is hashed in the ledger with this file. In `zt` mode it refused to run at registration ("incomplete: 0 episodes").

## Origin of the hypothesis
The source is post hoc: K07 runs that had already been read.

**Account.** The gain of the layer on a host controller follows the error that the host leaves in the dominant term of the objective (comfort, weight 0.6).

**K07-X3 blind set** (36 episodes, K07-G30 -> K07-G30 + layer; `p258 k07` -> `paperpack/out/RESULT_HEADROOM_K07.json`):
- Spearman rho(gain, base comfort share) = 0.798, one-sided permutation p = 0.0001.
- Pooled gain by thirds of the base comfort share (mean share 0.024 / 0.103 / 0.272): +0.03 % / +2.29 % / +4.00 %.
- Mean base comfort share 0.1329. Pooled gain over the 36 episodes 2.69 %. Over the composite episodes only it is 2.82 %, the paper's primary value.

**Also consistent, but not part of this test** (post hoc):
- Official training scenes: the comfort share of K07-G30 is 0.2-4.6 %, and the layer left the leaderboard score unchanged.
- Penguin: comfort share about 0.02; the start mechanism failed its development gate.

## Predictions for ZT Best (ZB) -> ZT Best + the unchanged layer (ZS), all 36 fresh episodes
Definitions are as in p258:
- base comfort share s_e from ZB, with competition weights;
- gain g_e = 100 (1 - L_ZS / L_ZB), where L = mean |reward| over 8640 steps.

The predictions:
- **P1.** Spearman rho(g, s) > 0 with one-sided permutation p < 0.05 (10,000 permutations, seed 0).
- **P2.** The pooled gain of the 12 episodes with the lowest s lies within [-0.5, +0.5] %.
- **P3.** If the mean s of ZB is below 0.1329, the pooled gain over the 36 episodes is below 2.69 %. If it is above, the pooled gain is above 2.69 %.

## Reporting
- Each prediction is reported as PASS or FAIL with its numbers, in the paper and in the notes, whatever the outcome. A FAIL is reported as evidence against the account.
- The primary rule of PREREG_ZTSUP_v1 ("keeps its gain") is unchanged and is evaluated separately by p252.
- Run, after p252 has found all 72 runs complete:
  `python -I paperpack/p258_headroom_test.py zt ztsup/RESULT_HEADROOM_ZT.json`
