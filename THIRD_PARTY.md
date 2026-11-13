# Third-party code and data

Every third-party file in this repository is public and is used under its own licence. We credit the authors here and in the paper (Appendix K).

| What | Author | Source | Licence | Here |
|---|---|---|---|---|
| K07 controller | Danil Malyshev (malyshevdanil) | Kaggle notebook, SB26 | Apache 2.0 | `controllers/k07/` |
| ZT Best controller | Danil Malyshev | Kaggle notebook "Smart Buildings Probe ZT Best" | Apache 2.0 | Classes used through the runners |
| Penguin controller | penguin069; its code embeds a public controller by Daniel Wu (danielwuu), as its header states | Kaggle notebooks "Smart Buildings: Deep Fine Search Lean Margin" (penguin069) and "CPU_B_CONDITIONAL_HOTWATER_20261022" (danielwuu) | Apache 2.0 | Used through `runners/run_episode_penguin_v3pc.py` |
| Residual PPO controller (code and weights) | Alex Chilton (alexchilton) | Kaggle notebook "Yes, RL Can Run a Building: Residual PPO" and its dataset | Apache 2.0 | `controllers/residual_ppo/` (see `LICENSE-NOTICE.txt`) |
| Tracker correction (KRE-v2): `scenario_generator.py` as changed by the starter kit's pull request #1, "Fix tracker occupancy queries creating fictitious populations" | The Smart Buildings Competition Authors; pull request by thinhngo-x | github.com/smart-buildings-2026/starter-kit | Apache 2.0, with the starter kit's NOTICE (below) | `simulator/tracker_overlay/` |
| Simulator, scenario generator, training buildings, weather | SB26 organisers | Competition starter kit and data | Data: CC BY-NC-SA 4.0 | Not included; scenario definitions of our own blind set are in `scenarios/` |

**How we use them:**
- Third-party code was read before it was run.
- The Residual PPO weights are loaded only with `torch.load(..., weights_only=True)`.
- Where we wrap a controller, the wrapper is ours; the controller's code is unchanged unless a runner says otherwise, in which case it says exactly what changed.

**NOTICE of the SB26 starter kit** (reproduced as its Apache 2.0 licence requires):

```
Smart Buildings Foundation Model Competition
Copyright 2024-2026 The Smart Buildings Competition Authors.

This project is licensed under the Apache License, Version 2.0 (see LICENSE).

This work builds on the Smart Buildings Simulator ("smart_control" / sbsim),
originally developed and open-sourced by Google, also under the Apache License,
Version 2.0. The simulator components under
`smart_buildings/smart_control/` are derived from that project. We gratefully
acknowledge the original authors.
```
