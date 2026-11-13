# Paper-wide analyses

**Running them.** Most scripts read the raw runs: per-step traces and run summaries, several GB, to be archived separately. They also expect the folder layout of the research archive. Until that archive is published they are here to be read and audited, and their stored outputs are in `out/`. `p272b_learning_curve_plot.py` redraws its figure from `out/p272` alone (`make figures`). The main numbers are recomputed from `data/` by `make reproduce`.

Each script reads frozen, already-read results and writes into `out/`. Nothing is run in the simulator. Post hoc analyses say so in their header. `PROVENANCE.md` maps them to the tables and figures. `figures/` holds the scripts of the figures that are not drawn here.

| Script | What it does |
|---|---|
| `p251_paper_package.py` | Paper package 1 (11 Nov 2026): every number from FROZEN, already-read fresh-set results; |
| `p253_paper_analyses.py` | Paper analyses on the FROZEN fresh-set runs (already read; |
| `p254_results_analyses.py` | Results analyses for an earlier draft of the paper on FROZEN, already-read results (nothing new is run; |
| `p255_error_analysis.py` | Error analysis of the learning layer on the blind set (post hoc, descriptive; |
| `p256_case_study_figures.py` | Case study of the worst blind day and the error-analysis figures (post hoc, descriptive; |
| `p256b_case_study_figures_labels.py` | [p256b = p256 with the figure labels in the paper's common language: 'tuned static' -> 'G30'; |
| `p257_uns_analysis.py` | Analysis of the unoccupied-setup experiment (PREREG_K07UNS_v1) on the runs that have finished (post hoc, descriptive; |
| `p258_headroom_test.py` | Prospective test of the headroom account (PREDICTION_ZTSUP_HEADROOM_v1); |
| `p259_when_figure.py` | Figure 'When does the layer help?' (post hoc, descriptive; |
| `p260_mean_zone_check.py` | Mean zone vs. |
| `p261_fig_learned_labels.py` | p261: Figure 'what the layer learned' (Appendix J), identical to figure 3 of an earlier draft of the paper/v48/make_figures_v48.py except the labels in the paper's common language ('tuned static' -> 'G30', 'pressure overrides' -> 'pressure  |
| `p262_when_figure_zt.py` | [p262 = p259 + the 36 episodes of the registered test on ZT Best (RESULT_HEADROOM_ZT.json, p258) in panel (a); |
| `p263_when_figure_col.py` | [p263 = p262 drawn at the 5.5-in column of the official template (letters print at the sizes set here, not at 80 %); |
| `p264_when_figure_col2.py` | [p264 = p263 at 2.45 in instead of 2.7; |
| `p265_when_figure_col3.py` | [p265 = p264 at 2.25 in; |
| `p266_appendix_figures_col.py` | p266: the five appendix figures at column width. |
| `p267_ratio_and_decomposition.py` | p267: post hoc checks of the comfort-share relation and a decomposition of the gains. |
| `p268_targeted_term_headroom.py` | p268: the gain against the share of the term each mechanism targets (TEST_BACKLOG B47 item 2, continuing p267; |
| `p269_targeted_robustness.py` | p269: the remaining robustness checks of the targeted-term relation, for the appendix table of an earlier draft of the paper (TEST_BACKLOG B47 items 1-2; |
| `p270_uns_fresh_decomposition.py` | p270: why the unoccupied setup did not generalise on the blind set (post hoc, descriptive; |
| `p271_regime_spells_2021.py` | p271: the premise of the explanatory model of Appendix G on the 2021 weather. |
| `p272_learning_curve.py` | p272: how fast the learning source pays on an unknown building. |
| `p272b_learning_curve_plot.py` | p272b: redraw of the p272 learning-curve figure from P272_LEARNING_CURVE.json (nothing recomputed). |
| `p273_direction_of_mismatch.py` | p273: why the one-time gate works - the direction of a profile mismatch. |
| `p273c_class_temperatures.py` | p273c: mean observed temperature of the days in each class by the class of the episode's start (2021 main cohort, 98 episodes; |
| `p274_early_diagnostic.py` | p274: can the first week of observations indicate the headroom of the learning layer before most of the episode is played?. |
| `p281_bootstrap_correction.py` | p281: correction of the architecture-cluster bootstrap of the blind-set contrasts (found 11 Nov 2026 ~21:40 while building the supplementary package: supplementary_v1/scripts/reproduce_tables.py reproduced every point es |
| `p283_official_term_hour.py` | p283: where each controller loses on the official training scenes (the closest available proxy of the hidden leaderboard scenes), by term x occupancy x hour band, to choose tomorrow's mechanism by the decomposition rule  |
| `p283b_unoccupied_split.py` | p283b: the unoccupied loss on the official scenes, split finer, and what the unoccupied setup removed from it on K07 (the mechanism that moved the leaderboard by 4.3 %). |
