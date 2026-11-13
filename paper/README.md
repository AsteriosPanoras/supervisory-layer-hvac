# The paper

- `main.tex`, `appendices.tex`, `refs.bib`: the sources.
- `neurips_2026.sty`: the official template of the competition, unmodified.
- `figures/`: the figures. `paper.pdf` is the current build.

## Build
```
make paper            # or: cd paper && tectonic main.tex
```
We build with [Tectonic](https://tectonic-typesetting.github.io). On Overleaf, upload this folder and use `main.tex` as the main document.

## Rules of the track
- **Pages:** 4–8 pages, excluding references and appendices. `make check` verifies the page budget on the built PDF.
- **Not anonymous:** the author information and abstract must match the OpenReview metadata.
- **Required content:** a related-works section citing the contest paper and the SBsim paper (both cited), and the Kaggle score in the results (Section 5.4; every score in Table 27, Appendix J.5). The template asks: "a results section should follow, including the score from the Kaggle platform".

## Changing a number, a table or a figure
1. Find it in [`PROVENANCE.md`](../PROVENANCE.md): it names the script and the data.
2. Change or re-run the analysis, not the number in the text. If the analysis belongs to a registered test, add a new dated script (never edit the frozen one) and label the result as a correction or as post hoc.
3. Update the text from the new output, then run `make paper` and `make check`.

## Page budget
The main text is at its limit. Anything added has to replace text of the same length, or go to an appendix.
