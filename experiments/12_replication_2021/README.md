# 12 replication_2021

**Question.** Does the primary result of the profile study replicate? The one-time gate (MG) against persistence-based scheduling (TRIG2), the pre-specified comparator, on 40 new 2021 episodes at 28 stations, none overlapping an earlier episode; the protocol was registered before the runs.

**Where in the paper.** Abstract; Section 5.1; Appendix E (Tables 4, 13); Limitations; Conclusion.

**Outcome.** +0.255 pp [-0.037, +0.709] (MG 4.06 % and TRIG2 3.81 % over the baseline B). By its registered rule the result is inconclusive: just short of the criterion for replication (a lower bound above zero) and far from the criterion for failure (an upper bound below +0.25 pp). 95 % of the bootstrap resamples lie above zero (post hoc). Mean episode-wise gain +0.517 pp [0.076, 1.125] (MG better in 16 episodes, worse in 6, tied in 18). All 158 episodes pooled (main sample, extension and replication; registered secondary): +0.366 pp [0.206, 0.535].

**Files here.** PROTOCOL.md (registered before any run; OpenTimestamps proof in `integrity/opentimestamps/`), the arms, the design (`execution/make_design_h3.py`, cells, crossings, run plans), power planning (`analysis/h3_power.py`), the result puller and the job chunks it ran, analyses (p296 registered, p296b post hoc), results, EXECUTION_NOTES.md.

**Layout.**
- `PROTOCOL.md`: what was fixed before the runs.
- `analysis/`: the analysis scripts, hashed before the runs they read.
- `results/`: their outputs.
- `execution/`: design, job lists and launch scripts.
- `EXECUTION_NOTES.md`: events during the runs, recorded as they happened.
