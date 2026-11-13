# 03 out_of_family_and_held_out

**Question.** Frozen profiles and the layer outside the development families (out-of-family development architectures and held-out architectures).

**Where in the paper.** Section 5.2; Appendix J (Table 23); Limitations.

**Outcome.** Out of family the frozen profiles did not transfer; on the held-out set the layer reduced the pooled loss by 3.9 % but improved only two of four composite architectures (pre-specified rule failed).

**Files here.**
- `profiles/`: the frozen profiles on out-of-family development architectures and on held-out architectures (protocol + amendment v2, selection and held-out analyses, results).
- `layer_held_out/`: the learning layer on the held-out architectures (protocol, analysis, result; Table 23).

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
