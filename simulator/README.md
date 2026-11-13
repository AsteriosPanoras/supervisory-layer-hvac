# How episodes are run

- **`kernel_runner_v3.py`:** runs one stage of an experiment.
  - **Input:** the jobs of `<experiment>/jobs/<stage>.jobs` (one line per episode: building, weather window, policy key, steps).
  - **Execution:** one process per episode, through an episode runner from `runners/`.
  - **Environments:** the same file runs on Kaggle and locally; locally `RUNNER_LOCAL_ROOTS` names the simulator, the scenarios and the experiments folders.
- **`cross_config.py`:** builds a crossed scenario (a building on another weather window) when a job asks for one.
- **`tracker_overlay/` (KRE-v2):** the competition's episode tracker had an accounting error, corrected in the starter kit's pull request #1. Every episode of the paper ran with that correction.
  - The runner requires the competition's original tracker file as the base, puts the corrected file in place, and refuses to run unless exactly that one file differs.
  - `starter_kit_PR1.diff` is the change.
- **The simulator itself:** the `smart_buildings` package of the SB26 starter kit; it is not included.
