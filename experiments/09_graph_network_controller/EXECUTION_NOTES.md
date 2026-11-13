# GNNSUP execution notes (companion to PREREG_GNNSUP_v1; the registration itself is never edited)

1. **2026-11-12 01:38:57, blind split between machines (before any blind run).** The laptop needed 950–1,290 s per 2016-step recording episode, about 3–4 times slower per worker than host-B. The blind runs are therefore split as follows:
   - **host-B:** 32 scenes (64 jobs, GNNSUP_BLIND_P), after experiment A, 8 workers, nice 19, until 09:00 at the latest.
   - **Laptop:** the last 4 scenes (8 jobs, GNNSUP_BLIND_L).

   Both arms of a scene always run on the same machine. Jobs, scenes, runner and analysis are unchanged.

   - The failed folder is kept as laptop/failed_gnfid_wrong_scen_dir; it holds no episode output.
   - The stage is relaunched unchanged with kaggle_k07x2_scen.
   - The gate (p288) had already passed: G1 and G2 true, GN/teacher loss 0.9998.

3. **2026-11-12 11:29:24, the 32 host-B scenes are rerun on the laptop and on Kaggle (no host-B blind output was ever read).**
   - **What happened:** at 04:18 the laptop's Wi-Fi left the network through which host-B is reachable. From then on host-B could not be reached, and its result copy failed. host-B is now no longer available.
   - **No result seen:** none of its GNNSUP_BLIND_P output was copied or seen. p285/p286 refused to compute on the 8 laptop runs alone (4 of 36 episodes).
   - **The same 64 jobs** (GNNSUP_BLIND_P, unchanged order, runner, weights e5483721…, scenes) are split:
     - **GNNSUP_BLIND_PL:** the last 10 scenes, 20 jobs, on the laptop (5 workers, nice 19, cores 0-11, k07x3 scenes);
     - **GNNSUP_BLIND_PK:** the first 22 scenes, 44 jobs, on Kaggle CPU notebooks.
   - **Pairing:** both arms of a scene always run on the same machine and kernel, so every paired comparison is within one machine.
   - **Kaggle reproducibility:** the laptop's scene building_52300 pittsburgh22_w0918 (both arms, from GNNSUP_BLIND_L) is also rerun on Kaggle. Whether it is bitwise identical is reported; Kaggle's torch build may differ.
   - **If host-B's output becomes available later,** it is used only as a cross-check of these reruns, never as extra data.

4. **2026-11-12 13:07:18, the Kaggle sequencer re-queues a refused push (infrastructure only; no blind output has been read).**
   - Experiment A now shares the account's five Kaggle CPU sessions (EXT12 note 3), and a session outside these experiments was seen at 12:53-12:57. A push that Kaggle refuses for the session limit was counted as pushed by `gnn_auto.py`, which would then wait forever for a kernel that does not exist.
   - **Fix:** a push without "successfully pushed" goes back to the queue and is retried on the next cycle. Nothing else changes: kernels, jobs, collection, the weights check, p285 and p286.
   - The sequencer is restarted. On start it rebuilds its state from Kaggle (a kernel that exists is never pushed again) and re-collects the finished kernels into the same folders; the reproducibility check is recomputed from the same files.
   - The original is kept as `gnn_auto_v1_no_refusal_check.py`.
