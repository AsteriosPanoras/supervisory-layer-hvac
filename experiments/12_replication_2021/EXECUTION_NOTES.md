# H3 execution notes (companion to PREREG_H3_v1; the registration itself is never edited)

1. **2026-11-12 13:45:46, the laptop part of H3 starts after the K07SUPFIX stage (infrastructure only; no H3 job has been claimed or run).**
   - The authors decided at 13:50 to run K07SUPFIX (`k07supfix/PREREG_K07SUPFIX_v1.md`) on the laptop as soon as the graph-network stage leaves it.
   - H3's laptop part therefore waits for `k07supfix/laptop/SF_DONE` instead of the graph-network stage's END. Kaggle still waits for experiment A.
   - The scheduler was stopped while it was still waiting (its log shows no claim) and restarted with `plan_h3_run_n1.json`. That plan differs from `plan_h3_run.json` only in this wait condition.
   - Cells, arms, runner, analysis (p296), decision rule and the reporting deadline are unchanged.

2. **2026-11-12 14:51:04, the authors' hard stop and an earlier Kaggle start (no H3 job has been claimed or run).**
   - **Hard stop.** The authors closes all experiments at 21:00 on 12 Nov. No new H3 chunk starts after 19:45 EET (`no_new_chunk_after_epoch` = 1794505500). The protocol's rule for an incomplete run applies at that time instead of Friday 10:00: the result is reported on the complete cells and labelled incomplete. Cells are claimed in the fixed order, so cells missing at the stop are dropped from the end of that order, not by outcome.
   - **Earlier Kaggle start.** H3's Kaggle part starts as soon as experiment A has pushed its last kernel (a17), not when A has finished. The Kaggle adapter retries a push that the session limit refuses, so H3 takes the slots as A's last kernels end.
   - The laptop part still waits for the K07SUPFIX stage (note 1).
   - The scheduler is restarted with `plan_h3_run_n2.json`, which differs from `plan_h3_run_n1.json` only in these two settings. Cells, arms, runner, analysis (p296) and the decision rule are unchanged.
