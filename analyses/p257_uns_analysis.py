"""Analysis of the unoccupied-setup experiment (PREREG_K07UNS_v1) on the runs that have finished (post hoc, descriptive; the gate verdict p247 is unchanged).  usage: python -I p257_uns_analysis.py <out_dir>
Official training scenes (6 x {G30, U = G30 + G36 unoccupied setup, UR = U + learned start, R = G30 + learned start}): per-scene change, where the change happens (occupied vs
unoccupied steps; weekday nights vs weekends; morning hours 06-09), the zone temperature reached at night, and the morning comfort cost of the setup with and without the
learned start.  Development (K07-X 18 cells): UR vs G30 per architecture and by first-day class; U vs G30 and UR vs U only where U has finished."""
import sys, os, csv, glob, json, collections, datetime as dt, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
KEYS = {f"k07@{G}": "G30", f"k07ros@{G}": "R", f"k07sup@{G};ROS=0,TR=0,UNS_COOL_K=305.4": "U", f"k07sup@{G};ROS=1,TR=0,UNS_COOL_K=305.4": "UR"}
WN = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
def comp(z, nm): ck = [str(x) for x in z["comp_keys"]]; k, wt = WN[nm]; return np.abs(wt * z["comps"][:, ck.index(k)].astype(float))
def load(pattern, ok):
    R = {}
    for f in glob.glob(pattern, recursive=True):
        s = json.load(open(f)); p = KEYS.get(s["policy_key"])
        if p is None or s["status"] != "OK" or s["steps_done"] != 8640 or not ok(s): continue
        R[(s["building"], s["weather"], p)] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return R
res = {"note": "post hoc descriptive analysis of finished runs; the pre-registered gate verdict is unchanged"}
OFF = load(f"{H}/../k07uns/laptop/work_off/out/**/*.summary.json", lambda s: True)
scenes = sorted({(b, w) for b, w, p in OFF})
off = {}
for b, w in scenes:
    if not all((b, w, p) in OFF for p in ("G30", "U", "UR", "R")): continue
    zg, zu, zr = OFF[(b, w, "G30")][1], OFF[(b, w, "U")][1], OFF[(b, w, "UR")][1]
    ts = [str(x) for x in zg["ts"]]; tt = [dt.datetime.fromisoformat(x) for x in ts]; dow = np.array([t.weekday() for t in tt]); hr = np.array([t.hour for t in tt])
    occ = zg["comps"][:, [str(x) for x in zg["comp_keys"]].index("total_occupancy")] > 0
    lg, lu, lur = np.abs(zg["rewards"]), np.abs(zu["rewards"]), np.abs(zr["rewards"]); tot = lg.sum()
    night_wd = (~occ) & (dow <= 4); wkend = (~occ) & (dow >= 5); morning = (hr >= 6) & (hr < 9) & (dow <= 4)
    zt = lambda z: np.nanmean(z["zone_t_k"].astype(float), axis=1) - 273.15
    dU = {nm: (comp(zu, nm) - comp(zg, nm)) for nm in WN}; dUR = {nm: (comp(zr, nm) - comp(zu, nm)) for nm in WN}
    off[f"{b}|{w}"] = {"U_vs_G30_pct": 100 * (lu.sum() / tot - 1), "UR_vs_U_pct": 100 * (lur.sum() / lu.sum() - 1), "R_vs_G30_pct": 100 * (np.abs(OFF[(b, w, "R")][1]["rewards"]).sum() / tot - 1),
                       "U_change_pct_of_G30_loss": {"energy+carbon, weekday nights": 100 * sum(dU[nm][night_wd].sum() for nm in ("energy", "carbon")) / tot,
                                                    "energy+carbon, weekends": 100 * sum(dU[nm][wkend].sum() for nm in ("energy", "carbon")) / tot,
                                                    "energy+carbon, occupied": 100 * sum(dU[nm][occ].sum() for nm in ("energy", "carbon")) / tot,
                                                    "comfort, occupied": 100 * dU["comfort"][occ].sum() / tot, "comfort, weekday mornings 06-09": 100 * dU["comfort"][morning].sum() / tot},
                       "UR_vs_U_comfort_weekday_mornings_pct_of_G30_loss": 100 * dUR["comfort"][morning].sum() / tot,
                       "max_mean_zone_temp_unoccupied_C": {"G30": float(np.nanmax(zt(zg)[~occ])), "U": float(np.nanmax(zt(zu)[~occ]))},
                       "mean_zone_temp_at_06h_weekdays_C": {"G30": float(np.nanmean(zt(zg)[(hr == 6) & (dow <= 4)])), "U": float(np.nanmean(zt(zu)[(hr == 6) & (dow <= 4)]))},
                       "oat_mean_C": float(zg["oat_c"].mean()), "oat_night_mean_C": float(zg["oat_c"][~occ].mean()),
                       "learned_start_extensions_UR": sum(1 for e in (OFF[(b, w, "UR")][0].get("ros") or {}).get("log", []) if e["event"] == "start" and e.get("why") == "need")}
res["official"] = off
# development
coh = json.load(open(f"{N}/k07x/K07X_V2_COHORT.json"))["cells"]; DEV = {(c["architecture"], c["weather"]) for c in coh if c["split"] == "dev"}; grp = {c["architecture"]: c["group"] for c in coh}
dcls = {(c["architecture"], c["weather"]): c["day1_class"] for c in coh}
D = {}
for pat in (f"{N}/results_k07x2/**/*.summary.json", f"{N}/results_k07uns_dev/**/*.summary.json"):
    D.update(load(pat, lambda s: (s["building"], s["weather"]) in DEV))
def gain(x, y, cells):
    cs = [c for c in cells if (*c, x) in D and (*c, y) in D]
    if not cs: return None, 0
    return 100 * (1 - sum(np.abs(D[(*c, x)][1]["rewards"]).sum() for c in cs) / sum(np.abs(D[(*c, y)][1]["rewards"]).sum() for c in cs)), len(cs)
dev = {"UR_vs_G30_by_architecture": {a: gain("UR", "G30", [c for c in DEV if c[0] == a])[0] for a in sorted({c[0] for c in DEV})},
       "UR_vs_G30_by_first_day_class": {k: gain("UR", "G30", [c for c in DEV if dcls[c] == k]) for k in ("C", "M", "H")},
       "U_vs_G30_finished_cells": gain("U", "G30", sorted(DEV)), "UR_vs_U_finished_cells": gain("UR", "U", sorted(DEV))}
res["development"] = dev
json.dump(res, open(f"{OUT}/UNS_ANALYSIS.json", "w"), indent=1, default=float)
f2 = lambda v: "-" if v is None else f"{v:+.2f}"
md = ["## Official training scenes (change in loss, %; negative = better)", "| scene | U vs G30 | UR vs U | R vs G30 | night mean OAT | max zone T unocc. G30 / U | zone T at 06h G30 / U | learned-start extensions (UR) |", "|---|---|---|---|---|---|---|---|"]
for k, v in off.items():
    md.append(f"| {k} | {f2(v['U_vs_G30_pct'])} | {f2(v['UR_vs_U_pct'])} | {f2(v['R_vs_G30_pct'])} | {v['oat_night_mean_C']:.1f} | {v['max_mean_zone_temp_unoccupied_C']['G30']:.1f} / {v['max_mean_zone_temp_unoccupied_C']['U']:.1f} | {v['mean_zone_temp_at_06h_weekdays_C']['G30']:.1f} / {v['mean_zone_temp_at_06h_weekdays_C']['U']:.1f} | {v['learned_start_extensions_UR']} |")
md += ["", "Where the setup changes the loss (% of G30 loss):"] + [f"- {k}: " + "; ".join(f"{kk} {vv:+.2f}" for kk, vv in v["U_change_pct_of_G30_loss"].items()) + f"; learned start on weekday-morning comfort (UR vs U) {v['UR_vs_U_comfort_weekday_mornings_pct_of_G30_loss']:+.3f}" for k, v in off.items() if abs(v["U_vs_G30_pct"]) > 0]
md += ["", "## Development (gain, %; positive = better)", f"- UR vs G30 by architecture: { {a[-5:]: f2(v) for a, v in dev['UR_vs_G30_by_architecture'].items()} }", f"- UR vs G30 by first-day class: { {k: (f2(v[0]), v[1]) for k, v in dev['UR_vs_G30_by_first_day_class'].items()} }",
       f"- U vs G30 (finished cells): {f2(dev['U_vs_G30_finished_cells'][0])} (n={dev['U_vs_G30_finished_cells'][1]}); UR vs U: {f2(dev['UR_vs_U_finished_cells'][0])} (n={dev['UR_vs_U_finished_cells'][1]})"]
open(f"{OUT}/uns_analysis.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
