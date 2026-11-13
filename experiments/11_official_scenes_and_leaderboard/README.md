# 11 official_scenes_and_leaderboard

**Question.** Official training scenes as a proxy of the hidden leaderboard scenes; the leaderboard submissions and the predictions registered before them.

**Where in the paper.** Section 5.4 (Table 27); Appendix J.

**Outcome.** Learning left the score unchanged (comfort is a small share of the loss there); the unoccupied setup lowered it by 4.3 %.

**Files here.** Submission records and scores; `submitted_notebooks/`, the exact notebook submitted for each score with its Kaggle metadata (hashes as in the records); the notebook sources in `notebooks/`; diagnoses (p279, p284); registered predictions and an erratum.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs. Later amendments are separate files, each registered before any outcome.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
