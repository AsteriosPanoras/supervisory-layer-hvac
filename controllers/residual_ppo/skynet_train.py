"""PPO for the skynet stack (M1: flat shared fast nets), residual on pid_water with the CMA constants.

Train: building_train_1, _2, _4, _5, _7, every climate (25 scenarios), full 30-day scenarios back to back.
Held out: --holdout (default building_train_8), never trained on; skynet_eval.py scores checkpoints on it.
Train: the other buildings of ALL_BUILDINGS, every climate; --holdout none trains on all six (the final model).

    .venv/bin/python -u skynet_train.py --run m1_s03_d64 [--resume | --init PATH] [--steps 3000000] [--workers 6]
                                         [--scale 0.3] [--d 64] [--depth 2] [--seed 0] [--base cma|shipped]

Writes data/skynet/<run>/latest.pt every --ckpt-every updates, a snapshot every --eval-every steps (and starts
skynet_eval.py on it in the background), and one line per update to logs/skynet/<run>/history.jsonl.
--resume restores weights, optimiser, both RNGs, counters and each worker's scenario cursor; the rollout in
flight and the position inside the current scenario are not restored (each worker restarts its scenario at day 0).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from skynet.base import cma_best
from skynet.env import Workers
from skynet.nets import FlatNets, pad, pad_acts, pad_vmask   # pad helpers live in nets.py (the rollout workers use them too)

ALL_BUILDINGS = ["building_train_1", "building_train_2", "building_train_4", "building_train_5", "building_train_7", "building_train_8"]
HELD_OUT = "building_train_8"                     # the default --holdout
TRAIN_BUILDINGS = [b for b in ALL_BUILDINGS if b != HELD_OUT]


def zt_off_cma(log="logs/cma_zt/history.jsonl"):
    """zt_off with the best constants a tools/cma_zt.py run has logged so far (its history.jsonl, last line).
    zt_off_cmagen: the run tuned on shipped + generated buildings (logs/cma_zt_gen; b8 honest bill, rule alone 0.01109).
    zt_off_cmagen_ramp: the same plus the pre-start ramp RAMP_K 1.0 (logs/rl_diag/honest_grid_ramp.log: b8 0.01077, b7 0.00734).
    zt_off_warm_ramp: the warm-up CMA's constants (logs/cma_zt_warm, 13 levers incl. pre-start hours) plus the ramp
    (logs/rl_diag/eval_warm_cma.log: b8 0.01063, b7 0.00723)."""
    rec = json.loads((ROOT / log).read_text().splitlines()[-1])
    return {"controller": "zt_off", "attrs": rec["best_attrs"], "cma_gen": rec["gen"], "cma_fitness": rec["best"], "cma_log": log}


def scenario_lists(n_workers, seed, buildings=TRAIN_BUILDINGS, extra=()):
    import sbrl
    scen = [(b, w.name) for b in buildings for w in sorted((sbrl.SCENARIOS / b).iterdir()) if (w / "config.gin").exists()] + list(extra)
    rng = np.random.default_rng(seed); rng.shuffle(scen)
    return [scen[i::n_workers] for i in range(n_workers)]


def gen_scenarios(n_cold, n_other, seed):
    """Generated scenarios (tools/gen_buildings.py, data/gen) that built without an error: n_cold each of fairbanks and
    minneapolis, n_other each of the other four climates, drawn once with seed."""
    man = ROOT / "data/gen/manifest.jsonl"
    rows = [json.loads(l) for l in open(man)] if man.exists() else []
    held = {tuple(x) for x in json.loads((ROOT / "skynet/selection_set.json").read_text())["scenarios"]}   # never train on the selection set
    rows = [r for r in rows if "error" not in r and (r["name"], r["weather"]) not in held
            and (ROOT / "data/gen/scenarios" / r["name"] / r["weather"] / "config.gin").exists()]
    rng = np.random.default_rng(seed); pick = []
    for clim, n in [("fairbanks", n_cold), ("minneapolis", n_cold), ("atlanta", n_other), ("denver", n_other), ("phoenix", n_other), ("singapore", n_other)]:
        pool = [r for r in rows if r["weather"].startswith(clim + "_")]
        if n and len(pool) < n:
            sys.exit(f"gen_scenarios: {len(pool)} {clim} scenarios available, {n} asked")
        pick += [pool[i] for i in rng.choice(len(pool), n, replace=False)] if n else []
    extra = ROOT / "skynet/selection_extra.json"         # held out after the draw, so existing pools draw exactly as before
    seen = {(r["name"], r["weather"]) for r in pick} & {tuple(x) for x in json.loads(extra.read_text())["scenarios"]} if extra.exists() else set()
    if seen:
        sys.exit(f"gen_scenarios: this pool draws {len(seen)} of skynet/selection_extra.json's held-out scenarios, e.g. {sorted(seen)[0]}")
    return [(str(ROOT / "data/gen/scenarios" / r["name"]), r["weather"]) for r in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--resume", action="store_true"); ap.add_argument("--init")
    ap.add_argument("--steps", type=int, default=3_000_000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=512); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995); ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=10.0); ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=8640)
    ap.add_argument("--ckpt-every", type=int, default=10); ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--no-eval", action="store_true", help="smoke runs only: do not start held-out evaluations")
    ap.add_argument("--base", choices=["cma", "shipped", "zt", "zt_off", "zt_off_all", "zt_off_cma", "zt_off_cmagen", "zt_off_cmagen_ramp", "zt_off_cmagen_ramp_all", "zt_off_cmagen_ramp_all_free", "zt_off_cmagen_ramp_free", "zt_off_warm_ramp", "zt_off_cma_all"], default="cma",
                    help="the controller under the residual; zt_off_cma = zt_off with the best zt constants of tools/cma_zt.py, frozen into the checkpoint; "
                         "zt_off_all / zt_off_cma_all / zt_off_cmagen_ramp_all = zt_off / zt_off_cma / zt_off_cmagen_ramp with the residual acting at every hour; zt_off_cmagen_ramp_all_free = the last with the RL zone target not clipped to the comfort band (50 K margin); zt_off_cmagen_ramp_free = zt_off_cmagen_ramp (RL in PI hours only) with that target freedom (the off-hours rule is only its starting action)")
    ap.add_argument("--latch", type=int, default=1, help="steps a switch vote holds once cast (1 = the plain dead-zone vote)")
    ap.add_argument("--holdout", default=HELD_OUT, choices=ALL_BUILDINGS + ["none"],
                    help="the building left out of training and scored by the background evals; none = train on all six")
    ap.add_argument("--reward", choices=["raw", "paired"], default="raw", help="paired: minus a zt shadow on the same scenario (2x simulation)")
    ap.add_argument("--hold", type=int, default=1, help="steps a sampled residual is held (12 = one hour); only the sampling step carries a log-prob")
    ap.add_argument("--gen-cold", type=int, default=0, help="add this many generated fairbanks and minneapolis scenarios each to the training pool")
    ap.add_argument("--gen-other", type=int, default=0, help="and this many of each other climate"); ap.add_argument("--gen-seed", type=int, default=1)
    ap.add_argument("--log-std", type=float, help="set every exploration log-std to this after a fresh start or --init (not on --resume)")
    a = ap.parse_args()

    out, log = ROOT / "data/skynet" / a.run, ROOT / "logs/skynet" / a.run
    out.mkdir(parents=True, exist_ok=True); log.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(2)
    n_in = None                                   # a resumed run keeps the input widths it was trained with
    if a.resume and (ROOT / "data/skynet" / a.run / "latest.pt").exists():
        _m = torch.load(ROOT / "data/skynet" / a.run / "latest.pt", weights_only=False)["model"]
        n_in = tuple(_m[f"{e}.0.weight"].shape[1] for e in ("zone", "ahu", "plant"))
    net = FlatNets(a.d, a.depth, **({"n_in": n_in} if n_in else {})); opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    from cma_search import DEFAULT
    params = {"cma": cma_best, "shipped": lambda: dict(DEFAULT), "zt": lambda: {"controller": "zt"}, "zt_off": lambda: {"controller": "zt_off"}, "zt_off_all": lambda: {"controller": "zt_off_all"}, "zt_off_cma": zt_off_cma, "zt_off_cmagen": lambda: zt_off_cma("logs/cma_zt_gen/history.jsonl"), "zt_off_cmagen_ramp": lambda: (lambda b: dict(b, attrs=dict(b["attrs"], RAMP_K=1.0)))(zt_off_cma("logs/cma_zt_gen/history.jsonl")), "zt_off_warm_ramp": lambda: (lambda b: dict(b, attrs=dict(b["attrs"], RAMP_K=1.0)))(zt_off_cma("logs/cma_zt_warm/history.jsonl")), "zt_off_cma_all": lambda: dict(zt_off_cma(), controller="zt_off_all"), "zt_off_cmagen_ramp_all": lambda: (lambda b: dict(b, controller="zt_off_all", attrs=dict(b["attrs"], RAMP_K=1.0)))(zt_off_cma("logs/cma_zt_gen/history.jsonl")), "zt_off_cmagen_ramp_all_free": lambda: (lambda b: dict(b, controller="zt_off_all", attrs=dict(b["attrs"], RAMP_K=1.0, TARGET_OFF_MARGIN_K=-50.0)))(zt_off_cma("logs/cma_zt_gen/history.jsonl")), "zt_off_cmagen_ramp_free": lambda: (lambda b: dict(b, attrs=dict(b["attrs"], RAMP_K=1.0, TARGET_OFF_MARGIN_K=-2.0)))(zt_off_cma("logs/cma_zt_gen/history.jsonl"))}[a.base](); update, steps, next_eval = 0, 0, a.eval_every
    params = dict(params, bill="honest")              # sbrl.honest_bill: gas billed at the tank's actual temperature
    cursors = [0] * a.workers

    if a.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", weights_only=False)
        if {k: v for k, v in ck["cfg"].items() if k != "n_in"} != {k: v for k, v in net.cfg.items() if k != "n_in"} or ck["args"]["scale"] != a.scale or ck["args"]["workers"] != a.workers or ck["args"].get("base", "cma") != a.base or ck["args"].get("latch", 1) != a.latch or ck["args"].get("reward", "raw") != a.reward or ck["args"].get("holdout", HELD_OUT) != a.holdout or ck["args"].get("hold", 1) != a.hold:
            sys.exit(f"refusing to resume: checkpoint cfg {ck['cfg']} scale {ck['args']['scale']} workers {ck['args']['workers']}")
        if ck["params"].get("bill") != "honest" or any(ck["args"].get(k) != getattr(a, k) for k in ("gen_cold", "gen_other", "gen_seed")):
            sys.exit(f"refusing to resume: checkpoint bill {ck['params'].get('bill', 'shipped')}, generated pool "
                     f"{[ck['args'].get(k) for k in ('gen_cold', 'gen_other', 'gen_seed')]} vs {[a.gen_cold, a.gen_other, a.gen_seed]}")
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"])
        update, steps, next_eval, cursors, params = ck["update"], ck["steps"], ck["next_eval"], ck["cursors"], ck["params"]
        print(f"RESUMED {out / 'latest.pt'}: update {update}, {steps:,} steps, cursors {cursors}, next eval at {next_eval:,}; "
              f"the rollout in flight and each worker's position inside its scenario are NOT restored (restart at day 0)", flush=True)
    elif a.init:
        src, dst = torch.load(a.init, weights_only=False)["model"], net.state_dict(); grown = []
        for k, v in src.items():                 # a smaller checkpoint (fewer outputs, or a narrower --d): its block is
            if dst[k].shape != v.shape:          # copied; new rows keep the fresh net's init (zero for the heads, random
                assert dst[k].dim() == v.dim() and all(n >= o for n, o in zip(dst[k].shape, v.shape)), (k, v.shape, dst[k].shape)
                w = dst[k].clone()               # for hidden units); new input columns start at zero, so the wider net
                if w.dim() == 2: w[:, v.shape[1]:] = 0.0   # computes exactly what the checkpoint did until it trains
                if k == "c_head.0.weight":       # its input is three d-wide embeddings side by side: copy each block
                    o, n = v.shape[1] // 3, w.shape[1] // 3; w[:, :] = 0.0
                    for j in range(3): w[: v.shape[0], j * n: j * n + o] = v[:, j * o: (j + 1) * o]
                else:
                    w[tuple(slice(0, n) for n in v.shape)] = v
                v = w; grown.append(k)
            dst[k] = v
        net.load_state_dict(dst); print(f"INIT params only from {a.init}" + (f"; grown with zero rows: {grown}" if grown else ""), flush=True)

    gen = gen_scenarios(a.gen_cold, a.gen_other, a.gen_seed)
    lists = scenario_lists(a.workers, a.seed, [b for b in ALL_BUILDINGS if b != a.holdout], gen)
    if gen: print(f"generated scenarios: {len(gen)} (" + ", ".join(f"{Path(b).name}/{w}" for b, w in gen) + ")", flush=True)
    print(f"run {a.run} | {sum(p.numel() for p in net.parameters()):,} params | scale {a.scale} | workers {a.workers}: "
          + " ; ".join(f"w{i} {len(l)} scen" for i, l in enumerate(lists)) + f"\nbase params {params}", flush=True)
    if a.log_std is not None and not (a.resume and (out / "latest.pt").exists()):
        with torch.no_grad():
            for ls in net.stds(): ls.fill_(a.log_std)
        print(f"exploration log-std set to {a.log_std}", flush=True)
    W = Workers(lists, params, a.max_steps, a.latch, a.reward == "paired")
    W.reset(cursors)
    ep_r = [[] for _ in range(a.workers)]
    hist = open(log / "history.jsonl", "a")

    def save(path):
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(),
                        update=update, steps=steps, next_eval=next_eval, cursors=cursors, params=params, cfg=net.cfg, args=vars(a)), path)

    try:
        while steps < a.steps:
            t0 = time.time(); buf = []; overrides = 0; done_eps = []; t_wait = 0.0
            # every worker samples with its own copy of the net and steps until n_steps x workers steps in total
            # (skynet/env.py rollout): a fast building contributes more steps, nobody waits for the slowest each step
            seeds = np.random.randint(0, 2 ** 31 - 1, size=a.workers)
            tw = time.time()
            res = W.rollout({k: v.detach().clone() for k, v in net.state_dict().items()}, net.cfg, a.n_steps * a.workers, a.scale, a.hold, seeds)
            t_wait = time.time() - tw; v_last = [vl for _, vl in res]
            for i, (out_i, _) in enumerate(res):
                for (f, act, lp_, v_, r, done, info, fr) in out_i:
                    buf.append((i, f, act, lp_, v_, r * a.reward_scale, done, info["vmask"], fr))
                    overrides += info["overrides"]; ep_r[i].append(info.get("raw", r))
                    if done:
                        done_eps.append(dict(scenario=info["scenario"], score=abs(float(np.mean(ep_r[i]))), steps=len(ep_r[i])))
                        ep_r[i] = []; cursors[i] = info["cursor"] + 1
            steps += sum(len(o) for o, _ in res); t_roll = time.time() - t0
            for g in opt.param_groups: g["lr"] = a.lr * max(0.0, 1.0 - steps / a.steps)     # linear decay to 0 at --steps
            # GAE per worker, bootstrapped at the rollout cut
            samples = []
            for i in range(a.workers):
                tr = [b for b in buf if b[0] == i]; adv = 0.0; nv = float(v_last[i])
                for (_, f, act, lp_, v_, r, done, vm, fr) in reversed(tr):
                    nonterm = 0.0 if done else 1.0
                    delta = r + a.gamma * nv * nonterm - v_
                    adv = delta + a.gamma * a.lam * nonterm * adv
                    samples.append((f, act, lp_, v_, adv, adv + v_, vm, fr, i)); nv = v_
            advs = np.array([s[4] for s in samples]); rets = np.array([s[5] for s in samples]); vals = np.array([s[3] for s in samples])
            ev = 1 - np.var(rets - vals) / max(np.var(rets), 1e-12)
            # per worker (its climates), over the steps where the residual acted, so a cold climate's large rewards
            # do not set the scale for every other climate
            frs = np.array([s[7] for s in samples]); wk = np.array([s[8] for s in samples]); advs_n = np.zeros_like(advs)
            for i in range(a.workers):
                m = (wk == i) & (frs > 0)
                if m.sum() > 1:
                    advs_n[m] = (advs[m] - advs[m].mean()) / (advs[m].std() + 1e-8)
            idx = np.arange(len(samples)); stats = []
            for _ in range(a.epochs):
                np.random.shuffle(idx)
                for s in range(0, len(idx), a.minibatch):
                    mb = idx[s: s + a.minibatch]
                    Z, zm, A, am, P = pad([samples[j][0] for j in mb])
                    acts_t = pad_acts([samples[j][1] for j in mb], Z.shape[1], A.shape[1])
                    mu, v = net(Z, zm, A, am, P)
                    lp, ent = net.log_prob_entropy(mu, acts_t, zm, am, pad_vmask([samples[j][6] for j in mb], A.shape[1]))
                    fr_t = torch.tensor([samples[j][7] for j in mb], dtype=torch.float32); lp = lp * fr_t
                    old = torch.tensor([samples[j][2] for j in mb]); adv_t = torch.tensor(advs_n[mb], dtype=torch.float32)
                    ratio = (lp - old).exp()
                    pg = -(torch.min(ratio * adv_t, ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t) * fr_t).sum() / fr_t.sum().clamp(min=1)
                    vf = ((v - torch.tensor(rets[mb], dtype=torch.float32)) ** 2).mean()
                    loss = pg + a.vf_coef * vf
                    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    stats.append((pg.item(), vf.item(), ent.mean().item(), ((ratio - 1).abs() > a.clip).float().mean().item()))
            update += 1
            st = np.mean(stats, 0)
            rec = dict(update=update, steps=steps, sps=round(a.n_steps * a.workers / (time.time() - t0), 1),
                       roll_s=round(t_roll, 1), wait_s=round(t_wait, 1), update_s=round(time.time() - t0 - t_roll, 1),
                       reward_mean=float(np.mean([b[5] for b in buf]) / a.reward_scale), explained_var=float(ev),
                       pg_loss=st[0], vf_loss=st[1], entropy=st[2], clip_frac=st[3],
                       log_std={k: [round(x, 3) for x in p.tolist()] for k, p in zip(("zone", "ahu", "plant"), net.stds())},
                       overrides_per_1k=round(1000 * overrides / (a.n_steps * a.workers), 2), episodes=done_eps)
            hist.write(json.dumps(rec) + "\n"); hist.flush()
            print(f"u {update} | {steps:,} steps | lr {opt.param_groups[0]['lr']:.1e} | {rec['sps']} sps | r {rec['reward_mean']:.5f} | ev {ev:.3f} | ent {st[2]:.2f} | "
                  f"clip {st[3]:.3f} | overrides/1k {rec['overrides_per_1k']} | log_std z {rec['log_std']['zone']}"
                  + "".join(f"\n   EPISODE {e['scenario']} score {e['score']:.5f} ({e['steps']} steps)" for e in done_eps), flush=True)
            if update % a.ckpt_every == 0:
                save(out / "latest.tmp"); (out / "latest.tmp").replace(out / "latest.pt")
            if steps >= next_eval:
                snap = out / f"ckpt_{steps}.pt"; save(snap); save(out / "latest.tmp"); (out / "latest.tmp").replace(out / "latest.pt")
                if not a.no_eval and a.holdout != "none":
                    elog = open(log / f"eval_{steps}.log", "w")
                    subprocess.Popen(["nice", "-n", "10", sys.executable, "-u", str(ROOT / "skynet_eval.py"), "--ckpt", str(snap), "--building", a.holdout],
                                     stdout=elog, stderr=subprocess.STDOUT, cwd=ROOT)
                next_eval += a.eval_every
        save(out / "latest.tmp"); (out / "latest.tmp").replace(out / "latest.pt"); save(out / "final.pt")
        print("DONE", flush=True)
    finally:
        W.close()


if __name__ == "__main__":
    main()
