# AXSUP execution notes (append only)

1. **Fidelity on the laptop (20:52–21:00), ALL PASS.** Stage `axsup/laptop/work_axfid`: 6 runs, 864 steps, development scenes 52010 phoenix_7 and 52103 fairbanks_3.
   - F2: bitwise identical.
   - F3: the first difference falls at the supervisor's first action (steps 407 and 252).
   - Details are in RESULT_AXSUP_FIDELITY.json.
2. **Frozen 21:02:12.** PREREG_AXSUP_v1, p277, p278, the runner, the manifest, the jobs and the stage scripts went into the ledger before any blind run.
   - Setup: `axsup/run_stage_host-B_axsup.sh AXSUP_BLIND 8 pan`, at nice 19. The start was verified on the remote run.log: 72 of 72 jobs, 8 workers, KRE-v2 overlay, 36 scenarios; memory available 12 GB.
   - Files: the runner, kernel_runner_v3, cross_config, overlay and weather table are identical to the ZT Best stage on that machine (checked by sha256 and diff). skynet.pt 8b1b4d0a. torch 2.10.0+cpu was copied from the laptop's project environment.
   - Watcher bar63ixlu checks every 10 min and stops early if any run fails. At the end it copies `work_pan/out` to `results_axsup_blind/pan/`. Then run p277 and p278.
