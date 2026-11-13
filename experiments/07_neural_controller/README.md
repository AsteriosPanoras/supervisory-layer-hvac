# 07 neural_controller

**Question.** The unchanged layer on a public trained neural controller (Residual PPO), blind set.

**Where in the paper.** Section 5.2; Appendix J; Limitations.

**Outcome.** +3.37 % [1.57, 5.22], 8 of 10 better; the pre-specified rule failed (two architectures lost 1.15 and 1.35 %); three of four registered predictions held.

**Files here.** PROTOCOL.md (+ amendment, registered before any outcome), runner generator, fidelity (p275), analyses (p277b reported, p277, p278), results.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
