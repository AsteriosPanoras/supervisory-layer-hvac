# Reproducibility and the competition rules

What the rules of the NeurIPS 2026 Smart Buildings Challenge ask of code and results (read on 12 and 13 November 2026 from the competition's Kaggle pages), and how to reproduce every result of the paper and every leaderboard score from this repository. The layout follows Kaggle's guidelines for documenting a winning model: summary, hardware, software, data, entry points, trained models, execution time.

## What the rules ask
| Rule | Source | Here |
|---|---|---|
| Code may be shared publicly if this does not violate the intellectual property of third parties and the code is also shared on Kaggle, in the competition's forum or notebooks, for the benefit of all competitors; shared code is deemed licensed under an OSI-approved licence that allows commercial use. | Kaggle rules: Public Code Sharing | This repository was made public with the first-round paper; the same files are in the Kaggle dataset [`asteriospanoras/supervisory-layer-hvac`](https://www.kaggle.com/datasets/asteriospanoras/supervisory-layer-hvac), announced in the competition's forum. Our code is under Apache 2.0, an OSI-approved licence; third-party code keeps its own licence and credits ([`THIRD_PARTY.md`](THIRD_PARTY.md)). |
| No private sharing of competition code outside the team. | Kaggle rules: Private Code Sharing | Until the repository was made public, access was limited to the authors. |
| Open-source code used in a submission must be under an OSI-approved licence that allows commercial use. | Kaggle rules: Use of Open Source | Every third-party controller and the starter kit are under Apache 2.0 ([`THIRD_PARTY.md`](THIRD_PARTY.md)). |
| A prize winner delivers the code that generated the winning submission, with training and inference code and a description of the computing environment, documented after Kaggle's guidelines; the description must allow the approach to be reproduced and link to a code repository with complete instructions. | Kaggle rules: Winner's Obligations | This page, [`README.md`](README.md), the experiment folders and the submitted notebooks below; the paper is the description. |
| A winning submission and its source code are licensed under CC BY 4.0; third-party software, data and models under other licences are exempt. | Kaggle rules: Winner License | Our own code can be licensed so; third-party code keeps its licence. |
| Submissions are Kaggle notebooks: CPU or GPU, at most 9 hours, internet off; external data must be public. | Kaggle: Code Requirements; rules: External Data | All 15 submissions below are CPU notebooks with internet off; the only attached dataset, the Residual PPO weights, is the author's public dataset under Apache 2.0. |
| The competition data are licensed CC BY-NC-SA 4.0. | Kaggle rules: Data Access and Use | They are not redistributed here. |
| Paper track: the organisers' template; related work citing the competition and simulator papers; results with the Kaggle score; submission on OpenReview. | Organisers' template; Kaggle: Prizes | [`paper/`](paper/); Section 5.4 states the Kaggle score and Table 27 (Appendix J.5) holds every score. |

## Summary of the method
The layer sits on top of an unchanged rule-based HVAC controller and sets values the controller already exposes, within bounds fixed in advance. Its values come either from a library of weather-conditioned profiles, re-selected by a schedule, or from bounded rules that learn online from the building: a learned optimal start, a trim-and-respond floor under the duct static pressure, learned night precooling and an unoccupied cooling setup. Nothing is trained offline: every episode starts knowing nothing about the building. The paper evaluates both sources on matched 30-day episodes of the competition simulator, and every test was registered before its outcome ([`experiments/`](experiments/)).

## Hardware and software
- **Hardware:** CPUs only: a laptop (AMD Ryzen 7 5800H, Linux), a shared workstation and Kaggle CPU notebooks.
- **Execution time:** a 30-day episode takes 21–49 minutes on one core and about 1.2 GB of memory; the controllers need 1–4 ms per step.
- **Software:**
  - Python 3 with NumPy (2.0 or 2.1, recorded per run) and Matplotlib ([`requirements.txt`](requirements.txt));
  - the SB26 starter kit environment for the simulator;
  - PyTorch (CPU) for the neural and graph controllers.
- **Simulator:** the competition's evaluation kit, identified by the SHA-256 of its evaluation script (Appendix K of the paper). Every run records that hash and the hashes of its runner, controller code and scenario.

## Entry points
| Goal | Command or file | Time |
|---|---|---|
| Recompute the main numbers from per-episode results | `make reproduce` (ends with `ALL NUMBERS REPRODUCED`) | 1 minute |
| Build the paper and check its page budget | `make paper`, then `make check` | 1 minute |
| Run episodes | Use `simulator/kernel_runner_v3.py` with a job list (one `building weather policy_key steps` per line) and a runner from `runners/`, on the starter kit with the tracker fix of `simulator/tracker_overlay/`. Each experiment's `execution/run_stage_*.sh` gives the exact environment variables. | 21–49 minutes per episode and core |
| Re-run an experiment and its analysis | The experiment's `PROTOCOL.md`, manifest or job list, `execution/` and `analysis/` | as in each protocol |
| Reproduce a leaderboard score | Take the exact notebook submitted for that score, in `experiments/11_official_scenes_and_leaderboard/submitted_notebooks/<notebook>/` with its Kaggle metadata. Copy it to a notebook of the competition and submit it. | within the 9-hour limit |

## Trained models
- **The layer:** it has no trained model; it learns within each episode, from zero.
- **Residual PPO:** the author's weights (`controllers/residual_ppo/skynet.pt`, Apache 2.0), loaded with `torch.load(..., weights_only=True)`.
- **The graph network that imitates Residual PPO:** our weights, `experiments/09_graph_network_controller/model/gnn_weights_v1.pt`, trained by the code in that folder.

## Data
- **Competition data and simulator:** obtained from the competition; not included here.
- **Per-episode results:** in `data/episodes/` and in each experiment's `results/`.
- **Full per-step traces:** several GB; they will be archived separately.

## Every leaderboard submission
| Kaggle notebook | Policy | Public score | In the paper |
|---|---|---|---|
| `sb26-cmp-k07-native-b` | K07, baseline B | 0.01517 | Table 27 |
| `sb26-cmp-k07-mg-adaptation` | K07, one-time gate (MG) | 0.01549 | Table 27 |
| `sb26-cmp2-k07-g30` | K07, tuned fixed G30 | 0.01477 | Table 27 |
| `sb26-cmp2-k07-g30-ros` | G30 + learned start | 0.01477 | Table 27 |
| `sb26-cmp2-k07-g30-sup` | G30 + layer (start, pressure reset) | 0.01477 | Table 27 |
| `sb26-cmp2-k07-g30-uns` | G30 + unoccupied setup | 0.01414 | Table 27 |
| `sb26-cmp2-k07-g30-unsros` | G30 + unoccupied setup + learned start | 0.01415 | Table 27 |
| `sb26-cmp-penguin-deep-native` | Penguin (reproduced) | 0.01004 | Table 27 |
| `sb26-cmp-penguin-pc` | Penguin + learned precooling | 0.01001 | Table 27 |
| `sb26-cmp-resppo` | Residual PPO | 0.01053 | Table 27 |
| `sb26-cmp-resppo-layer` | Residual PPO + layer | 0.01064 | Table 27 |
| `sb26-cmp2-k07-g30-supw` | G30 + layer with a weather-clustering gate, version 1 | 0.01477 | no: a variant not used in the paper |
| `sb26-cmp2-k07-g30-supw2` | The same, version 2 | 0.01477 | no: a variant not used in the paper |
| `sb26-cmp-penguin-a4-static` | A Penguin variant from a separate study | 0.01005 | no: notebook not included |
| `sb26-cmp-penguin-cont-all18` | A Penguin variant from a separate study | 0.01005 | no: notebook not included |

The submission records (`experiments/11_official_scenes_and_leaderboard/submissions/`) give each submission's id, time and notebook hash, and the notebooks in `submitted_notebooks/` are the submitted files byte for byte, except one word of a comment in `sb26-cmp-penguin-pc` changed when this repository was exported. [`integrity/EXPORT_MAP.csv`](integrity/EXPORT_MAP.csv) lists the hash of every original. Scores were never used to choose a configuration.

## Still to do
- **Per-step traces:** archive them with a DOI (Zenodo or a release) and give the analysis scripts one configurable data root, so that every table and figure re-runs from this repository. Until then, `make reproduce` recomputes the main numbers from `data/`.
