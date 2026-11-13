"""p274: can the first week of observations indicate the headroom of the learning layer before most of the episode is played?.  Exploratory, post hoc, frozen blind runs (K07); nothing is run.
All choices below were fixed before the first run of this script and are reported whatever comes out.
Observation window: days 1-7 (steps 0-2015) of the comparator G30, the configuration that runs before the layer is switched on.
Predictors (per episode, from the G30 run, week 1 only):
  S1 opening shortfall (K; sensors only, the primary predictor): at each opening (first occupied step after at least 1 h without
     occupancy) the mean over zones of |T_zone at the opening - median T_zone over the occupied period that follows|, averaged over
     the openings of week 1;
  S2 recovery delay (min; sensors only): from the opening until the mean zone temperature is within 0.5 K of its median over the
     occupied period (the end of the period if never), averaged over the openings;
  R1 week-1 comfort share (reward terms; the comfort term is a function of zone temperatures and occupancy): sum over week 1 of
     |0.6 x normalized_productivity_regret| / sum over week 1 of |reward|;
  reference: the full-episode comfort share of p267 (uses the whole episode, so it is not an early diagnostic).
Outcomes (per episode, days 8-30 = steps 2016-8639, no overlap with the window): gain of SUP (learned start + pressure reset) and of
ROS (learned start alone) versus G30, 100 (1 - sum L_arm / sum L_G30), with L = |reward| per step.
Statistics: Spearman rho over the 30 composite episodes (primary) and all 36 (secondary); 95% interval by resampling the architectures
3,000 times (seed 1); the range of rho when one architecture is left out; rho over the 10 composite architectures (mean predictor vs
pooled gain); a median split of the 30 composite episodes on S1 and on R1 (pooled days 8-30 gain of each half).
Gate (as p267): pooled full-episode SUP gain over the composite episodes 2.82 %, rho(full-episode gain, full-episode comfort share) 0.80
over all 36 episodes (the first run stopped at this gate, which used the 30 composite episodes by mistake; no result had been computed).
This is not a selector: 30 episodes from 10 architectures, the same runs that measured the layer, and no threshold is validated.
Run 3 fixes a bug of run 2 in the secondary set only: the resampling drew 10 indices for the 12 architectures of the 36 episodes, so the
two out-of-distribution architectures never entered the intervals of that set; the composite results do not change.
usage: python -I p274_early_diagnostic.py <out_dir>"""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
MAN = [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"]
man = [r for f in MAN for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
grp = {r["architecture"]: r["group"] for r in man}
ARMS = ("G30", "ROS", "SUP"); WK = 2016
REW, G30X = {}, {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if not r or r["policy"] not in ARMS or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
    k = (r["architecture"], r["weather"], r["policy"])
    if k in REW: raise SystemExit(f"duplicate {k}")
    z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); REW[k] = np.abs(z["rewards"].astype(float))
    if r["policy"] == "G30":
        ck = [str(x) for x in z["comp_keys"]]
        G30X[k[:2]] = dict(occ=z["occupancy"].astype(float), zt=z["zone_t_k"].astype(float), comf=np.abs(0.6 * z["comps"][:, ck.index("normalized_productivity_regret")].astype(float)))
cells = sorted({(a, w) for a, w, p in REW if all((a, w, q) in REW for q in ARMS)}); assert len(cells) == 36, len(cells)
comp = [c for c in cells if grp[c[0]] == "composite"]; assert len(comp) == 30
def openings(occ, end):
    on = occ > 0; out = []
    for t in range(12, end):
        if on[t] and not on[t - 12:t].any():
            u = t
            while u + 1 < len(on) and (on[u + 1] or on[u + 1:u + 13].any()): u += 1
            out.append((t, u + 1))
    return out
X = {}
for c in cells:
    d = G30X[c]; zt = d["zt"]; fin = np.isfinite(zt).all(axis=0); zt = zt[:, fin]; ops = openings(d["occ"], WK)
    assert len(ops) >= 4, (c, len(ops))
    s1, s2 = [], []
    for t0, t1 in ops:
        med = np.median(zt[t0:t1], axis=0); s1.append(float(np.mean(np.abs(zt[t0] - med))))
        zm = zt[t0:t1].mean(axis=1); mm = np.median(zm); hit = np.where(np.abs(zm - mm) <= 0.5)[0]
        s2.append(5.0 * float(hit[0] if len(hit) else t1 - t0))
    rg = REW[c + ("G30",)]
    X[c] = {"S1": float(np.mean(s1)), "S2": float(np.mean(s2)), "R1": float(d["comf"][:WK].sum() / rg[:WK].sum()), "openings": len(ops),
            "full_share": float(d["comf"].mean() / rg.mean())}
def gain(arm, sel, a=0, b=8640): return 100 * (1 - sum(REW[c + (arm,)][a:b].sum() for c in sel) / sum(REW[c + ("G30",)][a:b].sum() for c in sel))
from scipy.stats import spearmanr
rho = lambda x, y: float(spearmanr(x, y)[0])
assert f"{gain('SUP', comp):.2f}" == "2.82", gain("SUP", comp)                                                # gate
assert f"{rho([gain('SUP', [c]) for c in cells], [X[c]['full_share'] for c in cells]):.2f}" == "0.80"
archs = sorted({a for a, w in comp})
def analyse(sel, arm):
    y = [gain(arm, [c], WK) for c in sel]; out = {}
    for p in ("S1", "S2", "R1", "full_share"):
        x = [X[c][p] for c in sel]; r0 = rho(x, y); ar = sorted({a for a, w in sel}); bs = []; rng = np.random.default_rng(1)
        for _ in range(3000):
            ss = [c for j in rng.integers(0, len(ar), len(ar)) for c in sel if c[0] == ar[j]]
            if len({X[c][p] for c in ss}) > 2: bs.append(rho([X[c][p] for c in ss], [gain(arm, [c], WK) for c in ss]))
        loo = [rho([X[c][p] for c in sel if c[0] != a], [gain(arm, [c], WK) for c in sel if c[0] != a]) for a in ar]
        am = [np.mean([X[c][p] for c in sel if c[0] == a]) for a in ar]; ag = [gain(arm, [c for c in sel if c[0] == a], WK) for a in ar]
        out[p] = {"rho_episodes": r0, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "loao_range": [min(loo), max(loo)],
                  "rho_architectures": rho(am, ag)}
    return out
R = {"note": "exploratory, post hoc; predictors from week 1 of G30; outcomes days 8-30; not a selector", "window_steps": WK,
     "predictors": {f"{a}|{w}": X[(a, w)] for a, w in cells},
     "gain_days8_30_composite": {"SUP": gain("SUP", comp, WK), "ROS": gain("ROS", comp, WK)}, "results": {}}
for name, sel in (("composite_30", comp), ("all_36", cells)):
    R["results"][name] = {arm: analyse(sel, arm) for arm in ("SUP", "ROS")}
for p in ("S1", "R1"):
    med = np.median([X[c][p] for c in comp]); hi = [c for c in comp if X[c][p] > med]; lo = [c for c in comp if X[c][p] <= med]
    R[f"median_split_{p}"] = {"threshold": float(med), "n_high": len(hi), "n_low": len(lo),
                              "SUP_gain_days8_30_high": gain("SUP", hi, WK), "SUP_gain_days8_30_low": gain("SUP", lo, WK),
                              "ROS_gain_days8_30_high": gain("ROS", hi, WK), "ROS_gain_days8_30_low": gain("ROS", lo, WK)}
json.dump(R, open(f"{OUT}/P274_EARLY_DIAGNOSTIC.json", "w"), indent=1)
print(f"gate passed; days 8-30 pooled gain (composite): SUP {R['gain_days8_30_composite']['SUP']:.2f} %, ROS {R['gain_days8_30_composite']['ROS']:.2f} %")
for name in R["results"]:
    for arm in ("SUP", "ROS"):
        for p, v in R["results"][name][arm].items():
            print(f"{name:12s} {arm} vs {p:10s} rho {v['rho_episodes']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] LOAO [{v['loao_range'][0]:+.2f},{v['loao_range'][1]:+.2f}] arch rho {v['rho_architectures']:+.2f}")
for p in ("S1", "R1"):
    m = R[f"median_split_{p}"]; print(f"median split on {p} (>{m['threshold']:.3f}): SUP {m['SUP_gain_days8_30_high']:.2f} vs {m['SUP_gain_days8_30_low']:.2f} %, ROS {m['ROS_gain_days8_30_high']:.2f} vs {m['ROS_gain_days8_30_low']:.2f} %")
