# PREREG_GNNSUP_v1, AMENDMENT 3 (2026-11-12 03:24:04)

**Timing:** written before any fidelity outcome and before any blind run.

**What happened:**
- The fidelity runs (GNN_FID, 6 of 6) ended OK at 03:18.
- p287 then stopped at its own check `assert len(outs) == 1`, before computing or printing anything. The output folder holds the stage folder and also the runner's `manifest_stage1_shard0.json`, which the check counted.
- The launch watcher read the missing result as a failure and, as designed, started no blind run and told host-B not to start.

**Change:**
- p287 now selects only folders in that place. Nothing else changes: F1, F2, F3, ALL_PASS and p275 stay the same.
- The script that crashed is kept as `p287_gnn_fidelity_v1_crash_on_manifest.py`.
- The host-B watcher is restarted unchanged; the laptop blind stage starts after ALL_PASS, as registered.
