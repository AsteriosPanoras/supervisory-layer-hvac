# Reproduce the main numbers

```
python reproduce/reproduce_tables.py data/episodes
```

It needs Python 3 and numpy only, and runs in about a minute.
- **What it recomputes:** from the per-episode results, the numbers of Section 5.1 (primary contrast; class gains of Appendix E), the learning-source table of Section 5.2, the correlations of Section 5.3 and the unoccupied setup of Section 5.4.
- **How it checks:** each number is printed next to the paper's value; the run ends with `ALL NUMBERS REPRODUCED`.
- **Further checks:** the full analyses behind every table and figure are listed in `PROVENANCE.md`.
