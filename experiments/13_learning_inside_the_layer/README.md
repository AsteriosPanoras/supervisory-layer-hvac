# 13 learning_inside_the_layer

**Question.** Does learning add value inside the full layer? The layer with the learned start's time constant fixed at its prior of 1 h against the full layer, both with the pressure reset, with K07-G30 on the 36 blind scenes; the protocol was registered before the runs.

**Where in the paper.** Section 5.2; Appendix J (Table 4).

**Outcome.** +0.94 % [0.10, 1.82] over the 10 composite architectures, 7 of 10 better (exact Wilcoxon p = 0.084; -0.18 % on the OOD pair): by the registered rule, learning adds value inside the layer. It removed 1.50 points of comfort loss at 0.56 more in energy and carbon. The two rules add up: learning added 0.84 points of baseline-referenced gain with the reset and 0.86 without it, and the reset added 1.45 on the fixed start and 1.43 on the learned one. Two episodes of the full layer re-run on the second machine matched their originals bit for bit.

**Files here.** PROTOCOL.md (registered before any run; OpenTimestamps proof in `integrity/opentimestamps/`), the manifest, the setup script (`execution/make_supfix.py`: the frozen key of the full layer with the learning rate set to zero, on the frozen runner, so no new code), launch scripts and the job list, the analysis (p298), the result, EXECUTION_NOTES.md (a first launch pointed at a wrong scenario folder and stopped before any run; the runs then went to a second machine).

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: design, job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
