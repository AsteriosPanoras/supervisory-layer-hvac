# PREREG_ZTSUP_v1 — maintenance test: the frozen supervisor on the K07 author's later controller (ZT Best)


## Question
We ask whether the method keeps helping after the controller's own author changes the controller, without extra work.

The K07 author (Malyshev Danil, Kaggle, Apache-2.0) published a lineage: K07 → Zone PI K10 → **ZT Best** (`malyshevdanil/smart-buildings-probe-zt-best`). ZT Best is his latest controller, and it includes his own adaptive pre-start (EXTREME_PRESTART_H, MONDAY_EXTRA_PRESTART_H).

Test: the supervisor SUP (learned optimal start + trim & respond), with **unchanged code and parameters**, on top of ZT Best with the author's own constants.

## Runner
`run_episode_k07_v9supe_zt.py` is a NEW file: v9supe (sha e1b7921f…) with ONE change, the same change `run_episode_k07_v9_zt.py` made to v8. The base class becomes APResearchPolicy (ZT Best) instead of AdaptiveColdPolicy.
- The 'k07' folder of the experiment holds ZT's classes (`their_k07_policy.py`, sha a0246bd4…) and ZT's constants (`official_adapter/zt/their_zt`).
- Key `k07@` = ZT Best exactly; `k07sup@;ROS=1,TR=1` = ZT Best + SUP.

## Fidelity (p250; development scenes 52010 phoenix_7 and 52103 fairbanks_3, 864 steps, no reward read)
- **F1:** `k07@` under v9supe_zt equals the earlier ZT runner v9_zt (common arrays, bitwise).
- **F2:** `k07sup@;ROS=0,TR=0` equals `k07@`.
- **F3:** SUP on ZT runs (OK, no illegal action) and is identical to ZT up to its first differing action, where the supervisor has acted (or the runs are identical); activity is reported.

Nothing is launched unless F1–F3 pass.

## Design: zero-shot maintenance
- Nothing is tuned: neither SUP nor ZT, and there is no development step, because nothing is selected.
- **Set:** the fresh blind set K07-X3, 12 architectures (10 composite + 2 OOD) × 3 weather windows = 36 scenes, 8640 steps. ZT has never run on it.
- **Arms:** ZB (`k07@`) and ZS (`k07sup@;ROS=1,TR=1`), 72 runs.
- **Compute:** Kaggle (dataset sb26-ztsup-exp, private) and the laptop after the K07 UNS official stage. All arms of a scene run with the same runner and tracker overlay.

## Analysis (p252, frozen now)
Loss = mean |reward| over 8640 steps. Gain = 100 (1 − Σ L_ZS / Σ L_ZB), pooled over the 10 composite architectures. The 95 % architecture-cluster bootstrap uses 3000 replicates, seed 1.

**Rule "the supervisor keeps its gain after the author's update" (primary), all three needed:**
1. composite pooled gain > 0;
2. a majority of the changed composite architectures better;
3. no architecture worse than 1 %.

**Also reported:**
- by day-1 class and OOD;
- component decomposition (energy / carbon / comfort);
- supervisor activity (LOS 'need' extensions, T&R override share);
- descriptive comparisons on the same 36 scenes: ZB vs K07-G30, ZS vs K07-SUP, and ZS − ZB vs (K07-SUP − K07-G30).

## Disclosures
- The earlier ZT-TRANSFER-1 (8 Nov) asked a different question (weather profiles on ZT) on different scenes.
- Development cells 52010 phoenix_7 and 52103 fairbanks_3 carry the fidelity runs; no outcome is read.

## Labels
SB26 ZTSUP. These are simulator results, not leaderboard scores. Every file goes into the ledger N10S_FROZEN_SHA256_v1_1.txt (append only).
