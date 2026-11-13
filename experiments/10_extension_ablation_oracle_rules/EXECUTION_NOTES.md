# EXT12 execution notes (companion to PREREG_EXT12_v1; the registration itself is never edited)

1. **2026-11-12 06:46:28, p290 fixed in its JSON output only, before any experiment-B outcome was printed or read.**
   - The sequencer ran p290 at 06:45:44. It stopped in `json.dump`: numpy booleans in the two predictions cannot be written as JSON.
   - p290 prints its results only after that line, so no B outcome was shown.
   - **Fix:** `json.dump(..., default=...)` turns numpy scalars into plain numbers and booleans. The data, contrasts, predictions B1 and B2, and the reference rule are unchanged.
   - The original is kept as `p290_oracle_prior_v1_json_crash.py`.

2. **2026-11-12 06:46:44, the partial output of the crashed run is set aside unread.** Because `json.dump` opens its file before writing, the crash left a partial `RESULT_P290_ORACLE_PRIOR.json`.
   - It was not opened.
   - It is renamed `RESULT_P290_ORACLE_PRIOR_partial_from_crash_UNREAD.json` and kept, so the rerun writes a fresh file.

3. **2026-11-12 12:51:39, experiment A runs on Kaggle, before any AXR or AXT run.**
   - The registration names host-B as A's machine. host-B has been unreachable since 04:18 (it left the network).
   - A therefore runs on Kaggle: the 72 registered jobs (`AXABL_stage1.jobs`), unchanged, as 18 kernels a0..a17 of 4 jobs each, with the same runner (`run_episode_k07_v9supe_ax.py`, 557d4aab) and the same Residual PPO copy, in the private dataset `sb26-axabl-exp`. Launcher `axabl_kaggle.py`, sequencer `axabl_auto.py` (log `AXABL_AUTO.log`).
   - AXR and AXT (Kaggle) are paired with the existing AX and AXS (host-B), as p289 does.
   - **Why the pairing is kept:** the reproducibility pair of the graph-network rerun (`gnnsup/RESULT_GNNSUP_KAGGLE_REPRO.json`: a neural controller alone and with the layer, Kaggle against the laptop, same scene) is not equal bit for bit, but its rewards differ by at most 7.3e-7 per step and its episode losses by 2-4e-10 (about 4e-8 relative). The comparisons of A are of the order of 1 %. Re-running AX and AXS on Kaggle as well (144 runs) would not finish tonight.
   - **Reported as a deviation:** A's new arms ran on a different machine from AX and AXS.
   - p289, the bootstrap and the predictions A1-A4 are unchanged.

4. **2026-11-12 14:46:13, experiment A uses the Kaggle slots that the graph-network rerun leaves free (infrastructure only; p289 has not run, and no AXR or AXT outcome has been read).**
   - The first rule kept A at one slot until all 12 rerun kernels were COMPLETE. With 11 of 12 complete, only 2 of the account's 5 sessions were busy.
   - **New rule:** A uses 5 minus the rerun kernels still active according to its log (`gnn_active()`). Both sequencers re-queue a push that Kaggle refuses (GNN execution note 4), so no kernel can be lost.
   - The sequencer is restarted. It rebuilds its state from Kaggle (a0 and a1 complete, a2 running; nothing is pushed twice).
   - Jobs, runner, kernels, collection and p289 are unchanged. The first version is kept as `axabl_auto_v1_one_slot.py`.
