# Start here

Three ways in, by the time you have.

## 15 minutes: what the paper claims and how sure we are
1. `paper/paper.pdf`: the abstract, Section 5 (results) and Table 4 at the start of the appendix (what each test showed and its main limit).
2. [`RESULTS.md`](RESULTS.md): every headline number and the status of every test fixed in advance, failures included.
3. [`GLOSSARY.md`](GLOSSARY.md): every abbreviation used in the paper, the code and the file names.

## 1 hour: how it was done
1. [`experiments/README.md`](experiments/README.md): one line per experiment (question, where in the paper, outcome).
2. Follow one experiment end to end, for example [`experiments/07_neural_controller/`](experiments/07_neural_controller/):
   - `PROTOCOL.md`: what was fixed before the runs;
   - `analysis/`: the scripts, hashed before the runs;
   - `results/`: their outputs.
3. [`PROVENANCE.md`](PROVENANCE.md): pick any table or figure and find the script and data behind it.
4. Run `make reproduce` (numpy only, one minute): it recomputes the main numbers and checks them against the paper.

## Deeper: to work on it
1. [`CONTRIBUTING.md`](CONTRIBUTING.md): how we work. The integrity rules come first (what is never edited, how a new test is registered, how the paper is changed).
2. [`paper/README.md`](paper/README.md): building the paper, the 8-page budget, how to change a number or a figure without breaking its provenance.
3. [`ROADMAP.md`](ROADMAP.md): what was done for the first round, what is pending, and where help is most useful.
