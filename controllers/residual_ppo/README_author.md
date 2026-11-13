**If these weights or this code save you a training run, an upvote here or on [the notebook](https://www.kaggle.com/code/alexchilton/yes-rl-can-run-a-building-residual-ppo) keeps them updated: every better checkpoint we measure lands here with its numbers.**

Quick start: add this dataset to a notebook and run the notebook's last cell; it is a complete submission with these weights (its public-board score was pending when this version was built; the best so far is 0.01107 (v23 L1W12 @ 3.0M)).

# Smart Buildings Residual RL: weights and code

A trained residual PPO policy for the NeurIPS 2026 Smart Buildings challenge, the code that runs it, and every measured curve the notebook
[Yes, RL Can Run a Building: Residual PPO](https://www.kaggle.com/code/alexchilton/yes-rl-can-run-a-building-residual-ppo) plots.
The notebook trains nothing; it loads these weights, shows how they were made and what failed, and submits them.

## What the policy does

A hand rule runs the building (the public per-zone PI controller "zt" of malyshevdanil, plus an off-hours rule that shuts everything down outside its heating hours).
Inside those hours a small network adds a correction to every command: per zone the damper, the reheat valve and the zone's target temperature; per air handler the supply temperature, pressure and recirculation; for the plant the hot-water temperature and pressure.
One shared net per entity type (zone, air handler, plant), so the same weights run any building, whatever its number of zones.
The last layer starts at zero, so an untrained policy is exactly the rule.

## Files

| file | what it is |
|---|---|
| `skynet.pt` | the weights (fa2_all_from_s3_8m@7004160, 7,004,160 steps), the rule's constants and the training arguments |
| `policy.py` | `SkynetPolicy`: the class the competition gateway calls (`get_action(obs)`) |
| `features.py` | per-zone, per-air-handler and plant features, built from the building's metadata alone |
| `nets.py` | the networks (128 wide, 2 layers per entity type) |
| `base.py`, `zt_controller.py` | the rule under the residual (zt is Apache-2.0, malyshevdanil; base.py builds on the organisers' Apache-2.0 PIDWaterResetPolicy) |
| `curves.json` | every evaluation the notebook plots: checkpoint scores per climate, both held-out folds, the board history |
| `TAG.txt` | which checkpoint this version holds |
| `env.py`, `skynet_train.py`, `skynet_eval.py`, `sbrl.py`, `cma_search.py` | the trainer and evaluator, to train your own weights (below) |
| `hist_*.jsonl` | the tuned rule constants each `--base` reads |
| `selection_set.json`, `selection_extra.json` | the 20 + 40 held-out generated scenarios that judge checkpoints (section 4d); the trainer needs them in `skynet/` |
| `README.md` | this card |
| `MANIFEST.json` | every file with its size and SHA-256, the checkpoint tag and its training arguments |
| `LICENSE-NOTICE.txt` | the licences of the code this builds on |

## How to use it

```python
import os, sys, shutil
from pathlib import Path
W = next(Path(p) for p, d, f in os.walk('/kaggle/input') if 'skynet.pt' in f)   # the dataset's folder, wherever it mounts
os.makedirs('/tmp/pkg/skynet', exist_ok=True)                                 # rebuild the package the dataset stores flat
for f in ('__init__.py', 'base.py', 'features.py', 'nets.py', 'policy.py', 'zt_controller.py'):
    shutil.copy(W / f, f'/tmp/pkg/skynet/{f}')
sys.path.insert(0, '/tmp/pkg')
from skynet.policy import SkynetPolicy, load
net, params, scale, latch = load(W / 'skynet.pt')
# policy = SkynetPolicy(building_meta_data, action_spec, net, params, scale, PIDWaterResetPolicy, validate_action, latch)
```

The notebook's last cell is a complete submission with these weights.

## How it was trained

PPO, 6 simulator workers, full 30-day scenarios back to back, buildings 1, 2, 4, 5 and 7 in all five climates (Fairbanks, Atlanta, Denver, Phoenix, Singapore); building 8 held out and never trained on.
Rollout 2048 steps per worker, 4 epochs, minibatch 256, learning rate 8e-05, gamma 0.995, GAE lambda 0.95, clip 0.2, residual scale 1.0.
Trained as a chain of runs on CPUs, each starting from the previous run's weights: 5.0M steps on the off-hours rule, 4.0M with the zone-target head, 250k on the pre-start ramp rule, then a wider net (d128), 2,700 generated buildings and 12 workers per update; the notebook's sections 3 and 4f give the chain and every checkpoint's score.
It trained on a bill that charges gas at the hot-water tank's actual temperature; we do not use the known simulator billing bugs.

## Train it yourself

The notebook's section 3b rebuilds the training tree from these files and runs a short training on Kaggle CPU.
The full recipe on your own machine (competition data unpacked as the starter kit, `SB_KIT=<its path>`):

```
python -u skynet_train.py --run my_run --base zt_off_cmagen_ramp --scale 1.0 --steps 1000000 --n-steps 2048 --init skynet.pt
python -u skynet_eval.py --ckpt data/skynet/my_run/ckpt_<steps>.pt --building building_train_8
```

Checkpoints land in `data/skynet/<run>/` every 10 updates; `--resume` continues a run; held-out building 8 is scored every 250k steps.
About 140-200 steps per second with 6 asynchronous workers on a laptop CPU. The learning rate falls linearly to zero at `--steps`.

## Scores

| controller | 60 held-out generated buildings (honest bill) | public board |
|---|---:|---:|
| version 11 of the notebook (m3r @ 750k) | 0.008607 | 0.01128 |
| these weights (fa2_all_from_s3_8m@7004160) | 0.007405 | pending |

Lower is better. The full history per checkpoint and climate is in `curves.json` and in the notebook's first table.
The notebook also lists every change that did not work, with its number, so you can skip those runs.

## Licence

Apache-2.0, as the code it builds on.

If you use these weights or the code, an upvote here or on [the notebook](https://www.kaggle.com/code/alexchilton/yes-rl-can-run-a-building-residual-ppo) is much appreciated :-)
