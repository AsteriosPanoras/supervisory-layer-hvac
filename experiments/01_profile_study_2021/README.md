# 01 profile_study_2021

**Question.** Profile source: does a one-time gate beat persistence? (pre-specified primary test on 2021 weather, 98 episodes, plus a 20-episode extension).

**Where in the paper.** Section 5.1; Appendices E and G.

**Outcome.** Primary contrast +0.504 pp [0.337, 0.704] (success); the extension showed no difference (-0.004 pp).

**Files here.**
- `PROTOCOL.md`: the pre-specified test.
- `analysis/p128_h2_analysis.py` → `results/RESULT_H2_main.json` and `RESULT_H2_all.json`.
- `H2_BUILDINGS.csv`, `h2_arms.json`: the design.
- `execution/`: the design generator and the cell lists.
- The per-episode results are also in `data/episodes/profile_study_2021_episodes.csv`, from which `reproduce/` recomputes the numbers.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
