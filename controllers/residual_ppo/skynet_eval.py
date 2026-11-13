"""Score a skynet checkpoint (mean actions, no sampling) on the held-out building: 5 climates x 30 days, one process each.

    .venv/bin/python -u skynet_eval.py --ckpt data/skynet/<run>/ckpt_<steps>.pt
    .venv/bin/python -u skynet_eval.py --floor [--base cma|shipped] [--nudge 1e-4]   # the base controller alone

Appends one line to logs/skynet/<run>/eval_honest.jsonl (or logs/skynet/floor/eval_honest.jsonl; gas billed by sbrl.honest_bill): per-scenario |mean reward| and the mean.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from skynet.base import cma_best
from skynet.env import Workers
from skynet.features import ZONE_OUT, AHU_OUT, PLANT_OUT
from skynet.nets import FlatNets
from skynet_train import HELD_OUT, pad


class AvgNets:
    """Several checkpoints as one policy: each output head is the mean of the nets' mean actions (actions, not weights)."""

    def __init__(self, nets):
        self.nets = nets

    def __call__(self, *x):
        outs = [n(*x) for n in self.nets]
        mu = tuple(torch.stack(heads).mean(0) for heads in zip(*[m for m, _ in outs]))
        return mu, torch.stack([v for _, v in outs]).mean(0)


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="a checkpoint, or several separated by commas (their mean actions are averaged at every step)"); g.add_argument("--floor", action="store_true")
    ap.add_argument("--bill", choices=["honest", "shipped"], default="honest", help="shipped: the bill the board uses (tank billed at its setpoint); rows go to eval.jsonl")
    ap.add_argument("--steps", type=int, default=None, help="default 8640 (30 days) on --building, 2016 (7 days) on --set selection")
    ap.add_argument("--building", default=HELD_OUT)
    ap.add_argument("--set", choices=["selection", "selection60"], help="score on skynet/selection_set.json (20 held-out generated scenarios, 4 per board climate) "
                    "instead of --building; selection60 adds skynet/selection_extra.json (40 more, 8 per climate); rows go to eval_selection.jsonl / eval_selection60.jsonl")
    ap.add_argument("--base", choices=["cma", "shipped", "zt", "zt_off"], default="cma", help="--floor only: the controller")
    ap.add_argument("--nudge", type=float, default=0.0,
                    help="--floor only: add this to every AHU static-pressure command (a null-group member; 0 = the floor)")
    a = ap.parse_args()
    import sbrl
    if a.set:
        files = ["selection_set.json"] + (["selection_extra.json"] if a.set == "selection60" else [])
        scen = [(str(ROOT / "data/gen/scenarios" / n), w) for f in files for n, w in json.loads((ROOT / "skynet" / f).read_text())["scenarios"]]
        a.building = a.set
    else:
        scen = [(a.building, w.name) for w in sorted((sbrl.SCENARIOS / a.building).iterdir()) if (w / "config.gin").exists()]
    a.steps = a.steps or (2016 if a.set else 8640)
    if a.floor:
        from cma_search import DEFAULT
        net, params, scale = None, {"cma": cma_best, "shipped": lambda: dict(DEFAULT), "zt": lambda: {"controller": "zt"}, "zt_off": lambda: {"controller": "zt_off"}}[a.base](), 1.0
        tag, out = f"floor_{a.base}_nudge{a.nudge:g}_{a.building}", ROOT / "logs/skynet/floor"
    else:
        cks = [torch.load(c, weights_only=False) for c in a.ckpt.split(",")]   # several: their mean actions are averaged
        ck = cks[0]
        if any(c["params"] != ck["params"] or c["args"]["scale"] != ck["args"]["scale"] or c["args"].get("hold", 1) != ck["args"].get("hold", 1) for c in cks):
            sys.exit("--ckpt list: every checkpoint must share base params, scale and hold")
        nets = [FlatNets.from_checkpoint(c) for c in cks]
        net = nets[0] if len(nets) == 1 else AvgNets(nets)
        params, scale = ck["params"], ck["args"]["scale"]
        tag = "+".join(f"{c['args']['run']}@{c['steps']}" for c in cks) if len(cks) > 1 else f"{ck['args']['run']}@{ck['steps']}"
        out = ROOT / "logs/skynet" / ck["args"]["run"]
    params = dict(params, bill=a.bill)                # honest: sbrl.honest_bill, rows to eval_honest.jsonl; shipped: the board's bill, rows to eval.jsonl
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); t0 = time.time()
    rewards = []
    for c0 in range(0, len(scen), 20):                  # at most 20 simulator processes at once (memory)
        chunk = scen[c0: c0 + 20]
        W = Workers([[s] for s in chunk], params, a.steps, 1 if net is None else ck["args"].get("latch", 1))
        feats = W.reset([0] * len(chunk)); chunk_rewards = [[] for _ in chunk]; live = [True] * len(chunk)
        hold = 1 if net is None else ck["args"].get("hold", 1); tick, held = [0] * len(chunk), [None] * len(chunk)   # as skynet_train.py
        while any(live):
            if net is None:
                acts = []
                for f in feats:
                    ahu = np.zeros((len(f[1]), AHU_OUT)); ahu[:, 1] = a.nudge          # scale is 1 here, so the nudge is in action units
                    acts.append((np.zeros((len(f[0]), ZONE_OUT)), ahu, np.zeros(PLANT_OUT)))
            else:
                with torch.no_grad():
                    mu, _ = net(*pad(feats))
                acts = [(mu[0][i, : len(f[0])].numpy(), mu[1][i, : len(f[1])].numpy(), mu[2][i].numpy()) for i, f in enumerate(feats)]
                for i in range(len(chunk)):
                    if tick[i] % hold == 0 or held[i] is None: held[i] = acts[i]
                    else: acts[i] = held[i]
                    tick[i] += 1
            res = W.step(acts, scale)
            for i, (nf, r, done, info) in enumerate(res):
                if not info.get("free", 1.0): held[i] = None
                if live[i]:
                    chunk_rewards[i].append(r); live[i] = not done
            feats = [nf for nf, _, _, _ in res]
        W.close()
        rewards += chunk_rewards
    per = {f"{Path(b).name}/{w}": abs(float(np.mean(r))) for (b, w), r in zip(scen, rewards)}
    rec = dict(tag=tag, ckpt=a.ckpt, steps=a.steps, scores=per, mean=float(np.mean(list(per.values()))), seconds=round(time.time() - t0))
    rec["bill"] = a.bill
    rec["set"] = a.building
    if a.set:
        rec["climates"] = {c: float(np.mean([v for k, v in per.items() if k.split("/")[1].startswith(c)])) for c in ("atlanta", "denver", "fairbanks", "phoenix", "singapore")}
    name = f"eval_{a.set}.jsonl" if a.set else ("eval_honest.jsonl" if a.bill == "honest" else "eval.jsonl")
    with open(out / name, "a") as f: f.write(json.dumps(rec) + "\n")
    rd = out / "rewards"; rd.mkdir(exist_ok=True)                 # per-step rewards, for splits by weekday or hour; a set's
    # 60 scenarios share 28 weathers, so its traces are keyed <building>__<weather> (keyed by weather, 32 were overwritten)
    np.savez(rd / f"{tag.replace('/', '_')}_{a.building}_{a.steps}{'' if a.bill == 'honest' else '_shipped'}.npz", **{(f"{Path(b).name}__{w}" if a.set else w): np.array(r) for (b, w), r in zip(scen, rewards)})
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
