# 04 blind_set_k07

**Question.** The learning source on 12 unseen architectures (blind set) with K07: tuned G30, fixed start, learned start, learned start plus pressure reset.

**Where in the paper.** Section 5.2 (Table 3); Appendix J.

**Outcome.** +2.82 % [1.48, 3.94] over G30, 9 of 10 composite architectures better.

**Files here.** PROTOCOL.md, manifests, analysis (p226), results; corrected intervals in analyses/p281.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
