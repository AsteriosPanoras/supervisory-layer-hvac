# ZT maintenance test: execution notes (append only)

   - The fresh scenes are an rsync copy (dereferenced) of `k07x3/kaggle_k07x3_scen`. The runner `run_episode_k07_v9supe_zt.py` and the ZT classes have the same sha256 as locally (2b8144cb…, a0246bd4…).
   - Results come back into `results_ztsup_fresh/pan/`, which the frozen analysis p252 already reads (`results_ztsup_fresh/**`). The analysis is unchanged.
   - The Kaggle ZT kernels and the laptop ZT stage are not used. Their job files and sequencer stay in the ledger, unused.
   - Like every remote launch, the start is verified on the remote run.log (lesson of 11 Nov 00:17).

2. **Restarted with 8 workers (11:2x), before any outcome.** With 10 workers host-B had only ~4 GB free: each ZT run held ~1.1 GB at start, more than expected, on a shared machine (her own applications use ~15 GB). The stage's process group (pgid 3989785) was stopped about 4 minutes after launch, and no run had finished. The folder is kept as `stopped_attempt1_work_pan_10workers_memory`. The stage was relaunched with 8 workers, all 72 jobs from the start; 8 runs were verified running with ~6 GB free.
- 2026-11-11 13:14:00 EET: PREDICTION_ZTSUP_HEADROOM_v1 registered (P1-P3 on the base comfort share; script p258, reference RESULT_HEADROOM_K07.json; hashes in the ledger). No ZT outcome read; results_ztsup_fresh absent.

## Outcome (2026-11-11 14:56:45), read after all 72 runs were collected (watcher: DONE 14:51)
- **Primary rule (PREREG_ZTSUP_v1, p252):**
  - ZS vs ZB composite pooled gain **+5.71 %** [3.94, 7.14]; 10 of 10 changed composite architectures better; worst architecture +0.01 %.
  - By first-day class: C +2.46, M +6.17, H +7.82; OOD +7.13.
  - **KEEPS_GAIN = True.**
  - Components (% of ZB loss): comfort −6.62, energy +0.62, carbon +0.28.
  - Activity: 95 learned-start extensions; mean pressure-reset share 11.4 %.
- **Descriptive (p252):**
  - ZB vs K07-G30 −41.6 % [−62.8, −24.3]; ZS vs K07 layer −37.4 %.
  - SUP vs G30 on the same cells +2.82 % (reproduces the paper's value).
- **Registered prediction PREDICTION_ZTSUP_HEADROOM_v1** (p258 zt; script, reference and prediction unchanged since 13:14, verified by hash):
  - **P1 PASS:** Spearman rho(gain, ZB comfort share) = 0.93, one-sided p < 0.001.
  - **P2 FAIL:** pooled gain of the third with the lowest share = +0.78 %, outside [−0.5, +0.5]. The mean share of that third is 0.089, whereas for K07 it was 0.024.
  - **P3 PASS:** ZB mean comfort share 0.363 > 0.133 → predicted a gain above 2.69 %; observed 5.97 %.
  - All three verdicts are reported as they are. P2 is reported as a failed prediction.
