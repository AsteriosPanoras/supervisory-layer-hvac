# Supervisory layer for rule-based HVAC control

Code, protocols, data and analyses of the paper

> **Re-select or Learn? A Supervisory Layer for Rule-Based HVAC Control**
> Asterios Panoras, Apostolos Tsiakalos, School of Informatics, Aristotle University of Thessaloniki.
> NeurIPS 2026 Smart Buildings Challenge, paper track.

**Status:** public, with the paper of the first round. All experiments have finished; `paper/` holds the paper submitted to the first round. The competition's rules ask that code shared publicly while the competition runs (until 15 January 2027) also be shared on Kaggle, so the same files are in the Kaggle dataset [`asteriospanoras/supervisory-layer-hvac`](https://www.kaggle.com/datasets/asteriospanoras/supervisory-layer-hvac), announced in the competition's forum ([`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).

## What the paper does

The paper adds a supervisory layer on top of an unchanged rule-based HVAC controller. The layer gets its values from one of two sources:
- **a profile library:** profiles conditioned on the weather;
- **online learning within fixed bounds:** a learned optimal start and a trim-and-respond reset of the duct static pressure (ASHRAE Guideline 36).

It asks when each source helps. It answers with tests that were fixed in advance, run on buildings and weather the methods were never tuned on, and on several controllers: K07, ZT Best, Penguin, a trained neural controller and a graph network. It also gives an account of the gain: the share of the controller's loss in the term each rule corrects. That account was registered and tested prospectively.

## Map

| Folder | What it holds | Paper |
|---|---|---|
| [`paper/`](paper/) | LaTeX sources, figures, current PDF | — |
| [`layer/`](layer/) | The layer: its rules, their parameters, and the form used for the leaderboard | Sections 3, 4; Appendix K |
| [`experiments/`](experiments/) | One folder per experiment, each with the protocol frozen before the run, the job lists, the analysis and the results | Sections 5.1–5.4; Appendices |
| [`analyses/`](analyses/) | Paper-wide analyses: tables, figures, appendix analyses, and their outputs | All tables and figures |
| [`data/`](data/) | Per-episode results of every arm of the main tests | Tables 2–4 |
| [`reproduce/`](reproduce/) | One script that recomputes the main numbers from `data/` | — |
| [`runners/`](runners/) | The episode runners, one per controller | Section 4 |
| [`simulator/`](simulator/) | How episodes are run: kernel runner, tracker correction | Section 4; Appendix K |
| [`scenarios/`](scenarios/) | How the blind set of unseen architectures was generated | Section 4 |
| [`controllers/`](controllers/) | The public third-party controllers that were used, with their licences | Section 4; Appendix K |
| [`integrity/`](integrity/) | Hash ledger, OpenTimestamps proofs, export map | — |

**[`PROVENANCE.md`](PROVENANCE.md)** maps every table, figure and reported number to the script and the data it comes from. **[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)** gives the entry points, the computing environment, every leaderboard submission with its exact notebook, and what the competition rules ask.

## Check the main numbers (one minute, numpy only)

```
python reproduce/reproduce_tables.py data/episodes
```

It prints each number next to the paper's value and ends with `ALL NUMBERS REPRODUCED`.

## Run an experiment
Episodes run in the competition simulator (the `smart_buildings` package of the SB26 starter kit, not included here) through `simulator/kernel_runner_v3.py` and a runner from `runners/`. Each experiment folder gives:
- the job list: building, weather window, policy key, steps;
- the analysis script;
- the protocol that fixed both before the runs.

## Integrity
- **Frozen material:** every protocol, analysis script and result file was hashed into an append-only ledger before the outcome it concerns. The ledger is in `integrity/ledger_sha256.txt`.
- **Timestamps:** the later registrations also carry OpenTimestamps proofs (`integrity/opentimestamps/`).
- **Edits for publication:** the published copies differ from the archived originals only in comment and text lines: working notes and local paths were removed. `integrity/EXPORT_MAP.csv` lists the hash of every original and of every published file.

## Use of large language models
Large language models assisted with code development, with running and monitoring experiments, with checking numerical results
against their sources, and with drafting, editing and reviewing the manuscript. No LLM was part of any evaluated controller or
generated its control actions. Reported simulation metrics and uncertainty estimates were computed by analysis scripts from
recorded per-episode results; leaderboard scores came from the competition platform. The authors reviewed the AI-assisted
outputs and take full responsibility for the accuracy, originality and integrity of the work. Every protocol was hashed before
its outcome ([Integrity](#integrity)), every result of the main text has a provenance entry ([`PROVENANCE.md`](PROVENANCE.md)),
and `make reproduce` recomputes the main numbers.

## Not included
- **Full per-step traces of every episode:** several GB. They will be archived separately.
- **The competition's simulator and data:** obtain them from the competition (SB26 starter kit; data licence CC BY-NC-SA 4.0).
- **Material that is not part of the paper.**

## Licence
Our code is under the Apache License 2.0 ([`LICENSE`](LICENSE)). Third-party controllers keep their own licences and notices ([`THIRD_PARTY.md`](THIRD_PARTY.md)).
