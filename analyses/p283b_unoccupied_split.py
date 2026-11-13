"""p283b: the unoccupied loss on the official scenes, split finer, and what the unoccupied setup removed from it on K07 (the mechanism that
moved the leaderboard by 4.3 %).  Read-only on stored runs; nothing is run.  Cells: weekday 00-02, weekday 02-06 (the morning pre-start
window), weekday evening 17-24, weekend all day; each term (comfort, energy, carbon) as a share of the controller's weighted loss.
K07-G30 and K07 + setup (key k07sup@<G30>;ROS=0,TR=0,UNS_COOL_K=305.4) from the k07uns official stage (6 scenes, 8640 steps); the
Residual PPO controller alone from official_adapter/dry_ax/diag (the same 6 scenes).  Output: the share of each cell for G30 and ResPPO,
and the change the setup made on K07 in points of G30's loss (negative = removed).  usage: python -I p283b_unoccupied_split.py <out.json>"""
import sys, os, glob, json, numpy as np, datetime as dt
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
G30 = "k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
U = "k07sup@" + G30[4:] + ";ROS=0,TR=0,UNS_COOL_K=305.4"
def find(pattern, key):
    out = {}
    for f in glob.glob(pattern, recursive=True):
        s = json.load(open(f))
        if s["policy_key"] == key and s["status"] == "OK" and s["steps_done"] == 8640: out[(s["building"], s["weather"])] = f
    return out
RUNS = {"G30": find(f"{N}/k07uns/laptop/work_off/out/**/*.summary.json", G30), "U": find(f"{N}/k07uns/laptop/work_off/out/**/*.summary.json", U),
        "ResPPO": find(f"{N}/official_adapter/dry_ax/diag/*.summary.json", "k07@")}
sc = sorted(set(RUNS["G30"]) & set(RUNS["U"]) & set(RUNS["ResPPO"])); assert len(sc) == 6, (len(sc), {k: len(v) for k, v in RUNS.items()})
def cells(f):
    z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]
    ts = [dt.datetime.fromisoformat(str(t)) for t in z["ts"]]; hour = np.array([t.hour for t in ts]); wkend = np.array([t.weekday() >= 5 for t in ts]); occ = z["occupancy"].astype(float) > 0
    w = {"comfort": np.abs(0.6 * z["comps"][:, ck.index("normalized_productivity_regret")]), "energy": np.abs(0.2 * z["comps"][:, ck.index("normalized_energy_cost")]),
         "carbon": np.abs(0.2 * z["comps"][:, ck.index("normalized_carbon_emission")])}
    M = {"weekday 00-02": (~wkend) & (hour < 2) & ~occ, "weekday 02-06 (pre-start window)": (~wkend) & (hour >= 2) & (hour < 6) & ~occ,
         "weekday 17-24 unoccupied": (~wkend) & (hour >= 17) & ~occ, "weekday other unoccupied": (~wkend) & ~occ & ~((hour < 6) | (hour >= 17)),
         "weekend (all unoccupied steps)": wkend & ~occ, "occupied": occ}
    return {(c, t): float(v[m].sum()) for c, m in M.items() for t, v in w.items()}, float(sum(v.sum() for v in w.values()))
agg = {k: {} for k in RUNS}; tot = {k: 0.0 for k in RUNS}
for b, w in sc:
    for k in RUNS:
        c, T = cells(RUNS[k][(b, w)]); tot[k] += T
        for kk, v in c.items(): agg[k][kk] = agg[k].get(kk, 0.0) + v
CELLS = sorted({c for c, t in agg["G30"]}); TERMS = ("comfort", "energy", "carbon")
res = {"scenes": [f"{b}/{w}" for b, w in sc], "setup_total_change_pct_of_G30": 100 * (tot["U"] - tot["G30"]) / tot["G30"], "cells": {}}
for c in CELLS:
    res["cells"][c] = {t: {"G30_share": agg["G30"][(c, t)] / tot["G30"], "ResPPO_share": agg["ResPPO"][(c, t)] / tot["ResPPO"],
                           "setup_change_pts_of_G30": 100 * (agg["U"][(c, t)] - agg["G30"][(c, t)]) / tot["G30"]} for t in TERMS}
json.dump(res, open(OUT, "w"), indent=1)
print(f"K07 + setup vs G30 on these 6 official scenes: {res['setup_total_change_pct_of_G30']:+.2f} % of G30's loss")
print(f"{'cell':34s} {'term':8s} {'G30 share':>9s} {'setup chg':>9s} {'ResPPO share':>12s}")
for c in CELLS:
    for t in TERMS:
        v = res["cells"][c][t]
        if max(v["G30_share"], v["ResPPO_share"]) >= 0.01 or abs(v["setup_change_pts_of_G30"]) >= 0.1:
            print(f"{c:34s} {t:8s} {100*v['G30_share']:8.1f}% {v['setup_change_pts_of_G30']:+8.2f} {100*v['ResPPO_share']:11.1f}%")
