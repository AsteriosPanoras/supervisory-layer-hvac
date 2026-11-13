# PREREG_GNNSUP_v1, ADDENDUM: the weights of the blind test (2026-11-12 03:12:58)

**Timing:** written after the gate and before any blind run. No blind outcome exists.

**Weights:**
- **File:** `gnn_weights_v1.pt`, sha256 `e5483721a48a99d50e1684d28c9dff48da00fe5894908a214b0ceaf70c7e4c51`.
- **Training:** the registered command; the report is `REPORT_TRAIN_v1.json` (34 training episodes, 4 validation episodes; validation R² 0.974–1.000 for every output).
- **Identical copy:** on host-B (`gnnsup/gnn_weights_v1.pt`), with the same sha256.

**Gate (p288, `RESULT_GNNSUP_GATE.json`):**
- G1 is true: 4 of 4 validation runs ended OK with 0 illegal actions.
- G2 (descriptive since amendment 1) is true: GN's pooled loss is 0.9998 of the teacher's (per scene 0.994–1.009).

**Fidelity:** p287 is still running from the right scenario folder (execution note 2). The blind runs start only when `RESULT_GNNSUP_FIDELITY.json` has ALL_PASS, as registered.

**One added check (protective; it does not depend on any outcome):**
- Before p285 runs, every blind summary of GN and GNS must record the weights sha256 above.
- A run that records another sha256, or none, is excluded and reported.
