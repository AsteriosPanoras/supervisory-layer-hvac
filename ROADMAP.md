# Roadmap

## Round one (deadline 13 November 2026, 23:59 UTC)
- [x] **Graph-network controller (experiment 09):** blind result, then a sentence and an appendix paragraph.
- [x] **Extension experiments (10):**
  - which rule gives the gain on the neural controller;
  - the value of knowing the learned start's time constant from day 1 (an oracle);
  - a third rule in the same template.
- [x] **Two registered follow-ups:** a replication of the 2021 gate test on 40 new episodes (experiment 12) and learning inside the full layer (experiment 13).
- [x] **The rule template in the paper:** each rule reads, triggers, acts within bounds and hands back. A table of four rules with what each needs from the controller (`layer/README.md`).
- [x] **A short paragraph on leaderboard and transfer:** a single set of hidden scenes rewards its own mix of losses; evaluation on architectures no method was tuned on complements it.
- [x] **Future work, rewritten:**
  - automatic choice of rules from measured loss shares;
  - priors from the building's structure;
  - more rules in the same template;
  - optimized production controllers and real buildings.
- [x] **Leaderboard checks:** Penguin with precooling, Residual PPO with and without the layer (its registered prediction of no change failed: 1.0 % worse); ZT Best was not submitted.
- [x] **Submission:** the paper on OpenReview, with metadata matching the PDF.
- [ ] **Traces archive and a path layer:** archive the per-step traces (Zenodo or a release) and give the analysis scripts one configurable data root, so that every table and figure re-runs from this repository.

## Where help is most useful
- **Positioning:** related work and novelty, i.e. what is new is the evidence on when such a layer helps, across controllers.
- **The headroom account:** a cleaner formal statement and its assumptions.
- **Statistics:** a check of the cluster bootstrap and the multiplicity corrections.
- **Writing:** clarity within the 8-page budget.
- **New rules worth adding** to the template, and where they would pay (by the headroom account).
