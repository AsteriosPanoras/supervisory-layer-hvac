# How we work

## Integrity rules
1. **Fixed before the outcome.** A test's protocol, its analysis script and its decision rule are written and hashed into `integrity/ledger_sha256.txt` before any run they read. They are also time-stamped with OpenTimestamps where possible.
2. **Never edited afterwards.** These files are frozen once hashed:
   - `experiments/*/PROTOCOL*.md`;
   - the analysis scripts of registered tests;
   - `experiments/*/results/*`;
   - the ledger, which is append only.

   A correction is a new, dated file (an amendment or an erratum), written before the outcome it affects, or labelled post hoc.
3. **Failures stay.** A rule or prediction that failed is reported as failed, in the results and in the paper.
4. **The blind set is for testing only.** It is never used to design, tune or select anything.
5. **Post hoc analyses are labelled post hoc.** A diagnostic is never presented as a way of choosing a policy.
6. **The leaderboard is a consistency check.** It is never a selection criterion.

## Proposing a new experiment
1. Open an issue: the question, the arms, the scenes, the decision rule or predictions.
2. Write `experiments/NN_name/PROTOCOL.md` and the analysis script, and register both (hash into the ledger) before running.
3. Run. Record unexpected events in `EXECUTION_NOTES.md` as they happen.
4. Report every outcome in `RESULTS.md`, `PROVENANCE.md` and, if it enters the paper, in a new paper version.

## Changing the paper
See [`paper/README.md`](paper/README.md). In short:
- every number in the text must come from a file listed in `PROVENANCE.md`;
- the main text must stay within 8 pages (references and appendices excluded);

## Git
- **`main`** always holds a version that builds and reproduces (the check workflow enforces this).
- **Changes** go on a branch and arrive through a pull request with a short description (template provided). The other author reviews.
- **Commit messages** are short and in English; one change per commit where possible.
- **Never committed:** local absolute paths, credentials, large traces (over 50 MB; they are archived separately).
