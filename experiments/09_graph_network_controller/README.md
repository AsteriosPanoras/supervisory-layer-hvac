# 09 graph_network_controller

**Question.** The unchanged layer on a graph-network controller: a residual graph network on the building's own structure (zones, air handlers, plant), trained by imitation of the neural controller on training buildings.

**Where in the paper.** Table 3; Sections 5.2 and 5.3; Appendix J (Tables 4, 29).

**Outcome.** +3.55 % [1.82, 5.40] over the graph network alone, 8 of 10 composite architectures better (+5.73 % on the OOD pair); the pre-specified rule failed (52209 lost 1.17 %). All four registered predictions held (ρ = 0.74; lowest-share third +0.48 %; 3.96 % > 2.69 %; first week ρ = 0.49). Alone, the graph network was 0.80 % worse than the controller it imitates ([-1.94, +0.21]).

**Files here.** Protocol and amendment (registered before training), split, code (network, training, runner generator), gate, fidelity, analyses (p285, p286), results (including the check that all 72 runs carried the registered weights) and the trained weights (`model/`).

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
