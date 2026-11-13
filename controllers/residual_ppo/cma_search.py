"""CMA-ES over the constants of the organisers' best PID (pid_water) plus a minimum on-time for every on/off switch.

Score (the competition's): mean over scenarios of |mean reward| over the episode; lower is better.
Train set: 5 buildings x 2 weathers, the first WINDOW steps of each scenario.
Held out: building_train_8, all its weathers, never used by the search.
Every generation is appended to logs/cma/history.jsonl; the optimiser state is pickled to logs/cma/es.pkl.

    .venv/bin/python -u cma_search.py [--resume] [--gens 30] [--workers 6] [--window 864]
"""
import argparse, json, math, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "logs" / "cma"
HELD_OUT = "building_train_8"

# name, default (the shipped value), lower, upper, log-scale
PARAMS = [
    ("kp", 0.5, 0.05, 3.0, True),
    ("ki", 0.02, 0.0005, 0.5, True),
    ("kd", 0.1, 0.001, 1.0, True),
    ("integral_limit", 10.0, 1.0, 50.0, True),
    ("zone_damper_full_k", 2.0, 0.3, 6.0, True),
    ("REHEAT_FULL_K", 2.0, 0.3, 6.0, True),
    ("RESET_FULL_K", 4.0, 0.5, 10.0, True),
    ("RESET_EFFORT_QUANTILE", 0.25, 0.0, 1.0, False),
    ("RESET_DEADBAND_K", 0.25, 0.0, 2.0, False),
    ("min_on_steps", 1.0, 1.0, 576.0, True),           # 1 = the shipped behaviour (no hold)
]


def decode(z):
    """CMA works in [0, 1]^d (sigma 0.2); map to the parameter ranges, log-scaled where flagged."""
    p = {}
    for (name, _, lo, hi, lg), u in zip(PARAMS, np.clip(z, 0, 1)):
        p[name] = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if lg else lo + u * (hi - lo)
    return p


def encode(p):
    return np.array([(math.log(p[n]) - math.log(lo)) / (math.log(hi) - math.log(lo)) if lg else (p[n] - lo) / (hi - lo)
                     for n, _, lo, hi, lg in PARAMS])


DEFAULT = {n: d for n, d, *_ in PARAMS}


def episode(args):
    """|mean reward| of one scenario window under the parameterised PID."""
    building, weather, params, steps = args
    sys.path.insert(0, str(ROOT))
    import sbrl
    E, env, meta, spec, _ = sbrl.make_base(building, weather)
    kw = {k: params[k] for k in ("kp", "ki", "kd", "integral_limit", "zone_damper_full_k")}
    pol = E.POLICY_MAP["pid_water"](meta, spec, **kw)
    for k in ("REHEAT_FULL_K", "RESET_FULL_K", "RESET_EFFORT_QUANTILE", "RESET_DEADBAND_K"):
        setattr(pol, k, params[k])
    hold = int(round(params["min_on_steps"]))
    obs, _ = env.reset(); rs = []; on_for = None
    for _ in range(steps):
        a = E.validate_action(E.get_policy_action(pol, obs), spec)
        d = a["discrete_action"].copy()
        if on_for is None: on_for = np.zeros(len(d), int)
        if hold > 1:                                  # keep a switch on for at least `hold` steps once it is on
            d = np.where((on_for > 0) & (on_for < hold), 1, d).astype(np.int32)
        on_for = np.where(d == 1, on_for + 1, 0)
        obs, r, te, tr, _ = env.step({"continuous_action": a["continuous_action"], "discrete_action": d})
        rs.append(float(r))
        if te or tr: break
    return abs(float(np.mean(rs)))


def scenarios(split):
    sys.path.insert(0, str(ROOT)); import sbrl
    out, i = [], 0
    for b in sorted(p.name for p in sbrl.SCENARIOS.iterdir() if p.is_dir()):
        ws = sorted(p.name for p in (sbrl.SCENARIOS / b).iterdir() if (p / "config.gin").exists())   # one per climate
        if split == "train" and b != HELD_OUT:                  # rotate, so the 10 train scenarios cover every climate twice
            out += [(b, ws[(2 * i) % len(ws)]), (b, ws[(2 * i + 1) % len(ws)])]; i += 1
        if split == "held_out" and b == HELD_OUT: out += [(b, w) for w in ws]
    return out


def score(pool, params, scen, steps):
    vals = list(pool.map(episode, [(b, w, params, steps) for b, w in scen]))
    return float(np.mean(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--window", type=int, default=864)
    ap.add_argument("--popsize", type=int, default=10)
    a = ap.parse_args()
    import cma
    OUT.mkdir(parents=True, exist_ok=True)
    train, held = scenarios("train"), scenarios("held_out")
    print(f"train {len(train)} scenarios {train}\nheld out {len(held)} scenarios {held}\nwindow {a.window} steps", flush=True)
    hist = OUT / "history.jsonl"
    with ProcessPoolExecutor(a.workers) as pool:
        if a.resume and (OUT / "es.pkl").exists():
            es = pickle.loads((OUT / "es.pkl").read_bytes()); gen0 = sum(1 for _ in open(hist))
            print(f"RESUMED: generation {gen0}, {es.countevals} evaluations, best {es.result.fbest:.5f}", flush=True)
        else:
            t = time.time(); base, base_v = score(pool, DEFAULT, train, a.window)
            hb, hb_v = score(pool, DEFAULT, held, a.window)
            (OUT / "baseline.json").write_text(json.dumps(dict(train=base, train_each=base_v, held_out=hb, held_out_each=hb_v)))
            print(f"BASELINE pid_water as shipped: train {base:.5f}  held-out {hb:.5f}  ({time.time() - t:.0f}s)", flush=True)
            es = cma.CMAEvolutionStrategy(encode(DEFAULT), 0.2, {"bounds": [0, 1], "popsize": a.popsize, "seed": 0, "verbose": -9})
            hist.write_text(""); gen0 = 0
        for g in range(gen0, a.gens):
            t = time.time(); Z = es.ask()
            F = [score(pool, decode(z), train, a.window)[0] for z in Z]
            es.tell(Z, F); best = decode(es.result.xbest)
            rec = dict(gen=g + 1, evals=es.countevals, gen_best=min(F), gen_median=float(np.median(F)), best=es.result.fbest,
                       sigma=es.sigma, best_params=best, seconds=round(time.time() - t))
            if (g + 1) % 5 == 0 or g + 1 == a.gens:
                rec["held_out_of_best"] = score(pool, best, held, a.window)[0]
            with open(hist, "a") as f: f.write(json.dumps(rec) + "\n")
            (OUT / "es.pkl").write_bytes(pickle.dumps(es))
            print(f"gen {g + 1}: best {rec['best']:.5f} gen-best {rec['gen_best']:.5f} median {rec['gen_median']:.5f} sigma {es.sigma:.3f}"
                  + (f" | held-out {rec['held_out_of_best']:.5f}" if "held_out_of_best" in rec else "") + f" ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
