# 08 unoccupied_setup

**Question.** A Guideline 36 unoccupied cooling setup found by a diagnosis of the official scenes: development gate, then blind.

**Where in the paper.** Section 5.4; Appendix J (Tables 30, 31).

**Outcome.** Passed its gate; on the blind set -0.62 % alone and +1.27 % with the learned start; its blind rule failed (one architecture lost 3.3 %). On the leaderboard 4.3 % below G30.

**Files here.** PROTOCOL.md, analyses (p246 fidelity, p247 gate, p248 blind), results, execution notes.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
