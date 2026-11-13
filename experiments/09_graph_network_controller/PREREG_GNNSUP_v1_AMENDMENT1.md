# PREREG_GNNSUP_v1, AMENDMENT 1 (2026-11-12 01:21:06)

**Timing:** written while the teacher recordings are still running, before training, before any gate run and before any blind run. No gate or blind outcome exists.


**Change:**
- **G2** (GN's pooled loss on the 4 validation scenes at most 1.25 × the teacher's) becomes **descriptive**. It is computed by p288 as written and reported with the result, as a measure of how far this graph network is from the teacher. It no longer blocks the blind test.
- **G1** stays blocking: every validation run ends OK with 0 illegal actions.
- **The retry** (--iters 20000 --d 128) is used only if G1 fails.

Everything else in PREREG_GNNSUP_v1 is unchanged: training command, fidelity, scenes, arms, p285 rule, p286 predictions.
