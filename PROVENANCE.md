# Where every table, figure and number comes from

Numbers follow the current PDF (`paper/paper.pdf`). Each analysis script states in its header what it reads and what it writes. Every result file listed here was hashed into the ledger (`integrity/ledger_sha256.txt`) before the text that reports it was written. Table numbers inside the headers of the frozen analysis scripts refer to the paper version each script was written for (for example `analyses/figures/make_main_figures.py` names v43); the tables below give the current numbers.

**Status:** **pending** marks an item whose provenance is being completed for this repository. The numbers themselves are final.

## Main text

| Item | What | Produced by | From | Output |
|---|---|---|---|---|
| Figure 1 | The layer and its two sources | TikZ source | — | `paper/figures/fig_framework_two_sources_col3.tex` |
| Table 1 | Evaluation sets | Study design (Section 4) | `scenarios/`, `experiments/` | — |
| Table 2 | Main 2021 evaluation (98 episodes) | `experiments/01_profile_study_2021/analysis/p128_h2_analysis.py` (original); `reproduce/reproduce_tables.py` (recomputation) | `data/episodes/profile_study_2021_episodes.csv` | printed |
| Figure 2 | Profile source by initial weather class | `analyses/figures/make_fig_profiles_col3.py` | numbers of Table 2 and Appendix E | `paper/figures/fig_profiles_col3.pdf` |
| Table 3 | Blind set: every contrast, with 95 % architecture-cluster intervals | `analyses/p281_bootstrap_correction.py` (intervals); `reproduce/reproduce_tables.py` (recomputation) | blind-set results of experiments 04–08 | `analyses/out/p281*/` |
| Figure 3 | When the learning source helps (headroom) | `analyses/p265_when_figure_col3.py` | `analyses/out/WHEN_FIGURE_DATA.json`, `analyses/out/RESULT_HEADROOM_K07.json`, `experiments/05_maintenance_zt_best/results/RESULT_HEADROOM_ZT.json` | `paper/figures/fig_when_col3.pdf` |
| Section 5.3 numbers | Headroom account, registered predictions | `analyses/p258_headroom_test.py`, `analyses/p267–p269_*.py`, `analyses/p274_early_diagnostic.py` | blind-set results | `analyses/out/` |
| Section 5.4 numbers | Official scenes and leaderboard | `analyses/p283_official_term_hour.py`, `analyses/p283b_unoccupied_split.py`, `experiments/11_*/analysis/` | official-scene runs | `analyses/out/`, `experiments/11_*/results/` |

## Appendices

| Item | What | Produced by | Output or source |
|---|---|---|---|
| Table 4 | What each test showed | summary of all experiments | `experiments/README.md` |
| Tables 5–11 and 13–15, Figure 4 | Development stages of the profile source (2022 and 2023 cohorts, early selection, transfer, sensitivity) | pending: the development-stage analyses | — |
| Table 12 | Registration record: protocol, freeze time, SHA-256 of the original | the protocols in `experiments/*/PROTOCOL.md` | `integrity/ledger_sha256.txt`, `integrity/EXPORT_MAP.csv`, `integrity/opentimestamps/` |
| Figure 5 | Paired differences on 2021 weather | `analyses/p266_appendix_figures_col.py` | `paper/figures/fig_contrasts_col.pdf` |
| Table 16 | Profile overrides | frozen profile library | `layer/PARAMETERS.md` |
| Table 17 | Direction of the profile mismatch (post hoc) | `analyses/p273_direction_of_mismatch.py`, `analyses/p273c_class_temperatures.py` | `analyses/out/p273/` |
| Table 18 | The rules of the layer in one template, and what each needs from the controller | Section 3 | `layer/PARAMETERS.md` |
| Table 19 | Parameters of the learning source | frozen runner code | `layer/PARAMETERS.md`, `runners/run_episode_k07_v9supe.py` |
| Table 20 | Blind set: 12 architectures × 3 windows | `scenarios/blind_set/` | `scenarios/blind_set/K07X3_MANIFEST.md` |
| Table 21 | Gain per blind architecture | `analyses/p254_results_analyses.py` (part C), corrected intervals `analyses/p281_bootstrap_correction.py` | `analyses/out/` |
| Table 22 | Tests on the blind set | `analyses/p254_results_analyses.py` (part A), `analyses/p281_bootstrap_correction.py` | `analyses/out/` |
| Not in the paper (kept here) | What the layer learned (τ) per episode | `analyses/p266_appendix_figures_col.py` (from `p261_fig_learned_labels.py`) | `paper/figures/fig_learned_col.pdf` |
| Figure 6 | Time to value | `analyses/p272_learning_curve.py`, then `analyses/p272b_learning_curve_plot.py` | `analyses/out/p272/` (the figure `fig_learning_curve_col2.pdf` is `paper/figures/fig_time_to_value_col.pdf`) |
| Figure 7 | Gain per calendar day | `analyses/p266_appendix_figures_col.py` (from `p255_error_analysis.py`) | `paper/figures/fig_extremes_col.pdf` |
| Table 23 | Held-out episodes | `experiments/03_out_of_family_and_held_out/layer_held_out/analysis/p241_fresh_chain.py` | `experiments/03_*/layer_held_out/results/RESULT_K07X3H_CHAIN.json` |
| Appendix J.2 (its table is kept here) | Ten worst blind days: the six holidays and the worst day are described in Appendix J.2 | `analyses/p255_error_analysis.py` | `analyses/out/PAPER_ERROR_ANALYSIS.json` |
| Appendix J.2 (its table is kept here) | Learned-start decisions and prediction error | `analyses/p253_paper_analyses.py` | `analyses/out/PAPER_ANALYSES_*.json` |
| Table 24 | Change of the weighted loss components | `analyses/p254_results_analyses.py` (part F) | `analyses/out/` |
| Table 25 | Decomposition and headroom checks | `analyses/p267_ratio_and_decomposition.py`, `p268_targeted_term_headroom.py`, `p269_targeted_robustness.py` | `analyses/out/` |
| Table 26 | Early indicators from the first week | `analyses/p274_early_diagnostic.py` | `analyses/out/p274/` |
| Table 27 | Kaggle leaderboard (Appendix J.5) | Submission records | `experiments/11_official_scenes_and_leaderboard/submissions/` |
| Table 28 | Which source paid where: a summary of Section 5, with no new numbers | Tables 2, 3, 27 and 29 | — |
| Table 29 | Where each rule paid: every test of the learning source | experiments 04 to 13 and `analyses/out/UNS_ANALYSIS.json` | the result files of those experiments |
| Table 30 | Unoccupied setup on the official scenes | `analyses/p257_uns_analysis.py`, `analyses/p260_mean_zone_check.py` | `analyses/out/UNS_ANALYSIS.json`, `MEAN_ZONE_CHECK.json` |
| Table 31 | Unoccupied setup on the blind set | `experiments/08_unoccupied_setup/analysis/p248_uns_fresh.py`, `analyses/p270_uns_fresh_decomposition.py` | `experiments/08_unoccupied_setup/results/RESULT_K07UNS_FRESH.json` |
| Appendix J, neural controller | Residual PPO with the layer | `experiments/07_neural_controller/analysis/p277b_axsup_blind.py`, `p278_headroom_ax.py`; official scenes `experiments/11_*/analysis/p279_ax_official_diagnosis.py` | `experiments/07_neural_controller/results/` |
| Appendix K | Licences, compute, data | — | `THIRD_PARTY.md` |

## Later additions
Numbers added in the last revisions of the paper come from these files; rows marked post hoc were not registered in advance.

| Item | What | Produced by | Output |
|---|---|---|---|
| Table 27, new rows; Section 5.4 | Leaderboard: Penguin with learned precooling; Residual PPO alone and with the layer | Submission records | `experiments/11_official_scenes_and_leaderboard/submissions/` |
| Abstract, Section 5.1, Appendix E, Tables 4 and 13 | Registered replication of the 2021 gate test on 40 new episodes, and all 158 episodes pooled | `experiments/12_replication_2021/analysis/p296_h3_analysis.py` (design: `execution/make_design_h3.py`; planning: `analysis/h3_power.py`) | `experiments/12_replication_2021/results/RESULT_H3.json` |
| Section 5.1, Appendix E (post hoc) | Share of the bootstrap replicates of the replication above zero | `experiments/12_replication_2021/analysis/p296b_h3_bootstrap_share_posthoc.py` | `experiments/12_replication_2021/results/POSTHOC_H3_BOOTSTRAP_SHARE.json` |
| Section 5.1, Appendix E, Table 13 (post hoc) | 2021 extension: interval, effective number of episodes, mean episode-wise gain | `analyses/p297_review6_analyses.py` (part C) | `analyses/out/p297/P297_REVIEW6.json` |
| Section 5.2, Appendix J, Table 4 | Learning inside the full layer: fixed against learned time constant, both with the pressure reset (registered) | `experiments/13_learning_inside_the_layer/analysis/p298_supfix.py` | `experiments/13_learning_inside_the_layer/results/RESULT_K07SUPFIX.json` |
| Section 5.2, Appendix J, Tables 4 and 29 | Each rule alone on the neural controller (experiment A, registered) | `experiments/10_extension_ablation_oracle_rules/code/p289_axabl.py` | `experiments/10_extension_ablation_oracle_rules/results/RESULT_P289_AXABL.json` |
| Table 3, Sections 5.2 and 5.3, Appendix J, Tables 4 and 29 | The layer on a graph network imitating the neural controller, and its registered predictions | `experiments/09_graph_network_controller/code/p285_gnnsup_blind.py`, `p286_headroom_gnn.py` | `experiments/09_graph_network_controller/results/RESULT_GNNSUP_BLIND.json`, `RESULT_GNNSUP_HEADROOM.json`, `RESULT_GNNSUP_WEIGHTS_CHECK.json` |
| Section 5.3, Equation (7), Table 25, Appendix J (post hoc) | The headroom identity on four controllers, with the neural controller and 95 % architecture-cluster intervals for rho; cross-controller consistency; what to expect on a new architecture | `analyses/p297_review6_analyses.py` (parts A, B, D, E) | `analyses/out/p297/P297_REVIEW6.json` |
| Section 5.2 (0.02%), Appendix J | Oracle start (experiment B, registered) | `experiments/10_extension_ablation_oracle_rules/code/p290_oracle_prior.py` | `experiments/10_extension_ablation_oracle_rules/results/RESULT_P290_ORACLE_PRIOR.json` |
| Section 5.4 (0.38%), Appendix J | Third rule (experiment C, registered) | `experiments/10_extension_ablation_oracle_rules/code/p291_three_rules.py` | `experiments/10_extension_ablation_oracle_rules/results/RESULT_P291_THREE_RULES.json` |
| Section 5.2 (4.70 against 1.34), Appendix J | The neural controller's change by term on the blind set | `experiments/07_neural_controller/analysis/p277b_axsup_blind.py` | `experiments/07_neural_controller/results/` |
| Appendix J (post hoc) | Per architecture: comfort share, change by term, night and early morning | `experiments/07_neural_controller/analysis/p293_ax_blind_terms_posthoc.py` | `experiments/07_neural_controller/results/POSTHOC_AX_BLIND_TERMS.json` |
| Section 6, Appendix J (post hoc) | Official training scenes: when and in which term the layer changed the neural controller's loss | `experiments/11_official_scenes_and_leaderboard/analysis/p292_ax_official_terms_posthoc.py` | `experiments/11_official_scenes_and_leaderboard/results/POSTHOC_AX_OFFICIAL_TERMS.txt` |
| Appendix J, early warning (post hoc, exploratory) | The first week of the neural controller | `experiments/07_neural_controller/analysis/p295_ax_first_week_posthoc.py` | `experiments/07_neural_controller/results/POSTHOC_AX_FIRST_WEEK.json` |
| Figure 3, with Residual PPO | Panel (a) adds the registered test on Residual PPO | `analyses/p294_when_figure_col4.py` | `analyses/out/p294/` |
| Table 22, column Holm (W) | Holm adjustment of the exact Wilcoxon p-values | Holm over the six secondary contrasts of the column p Wilcoxon of the same table | — |
| Abstract, Section 5.4 (Kaggle score 0.01477), Section 6 (Which source, when), Section 7, Table 28 | Rewritten in the last revision: no new numbers; '11 of 12 architectures' counts Table 21 (9 of the 10 composite and both OOD architectures improved) | Tables 2, 3, 21, 27 and 29 | — |
| Appendices A–J | Appendix pass: Appendix A restructured (shared design once, one table of what differs between studies); the 2022 results moved to Appendix C; GL and G30 joined the profile table; the per-episode time constants, the ten worst days and the decisions by weather change left Appendix J.2 for this repository; no new numbers | the files listed above | — |
