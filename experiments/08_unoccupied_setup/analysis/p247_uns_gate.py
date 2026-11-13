"""K07 UNS gate and ablation on development K07-X and the six official training scenes (PREREG_K07UNS_v1); written 2026-11-11 before any UNS outcome.
usage: python -I p247_uns_gate.py <out.json>
Loss per run = mean |reward| over 8640 steps (status OK, no illegal action).  Arms: G30, R (k07ros), U (G36 setup), UR (setup + ROS).
GATE (UR vs G30) = D and O.  D: mean over composite development architectures of r_a < 0, every development architecture <= +1 %, all 18 cells.
O: mean over the 6 official scenes of r_s < 0, every scene <= +1 %, all 6 scenes.  The gate needs only G30 and UR; ablation parts with missing runs are
reported as incomplete and the script can be re-run later (same frozen file) once U and R are in."""
import sys, json, glob, os, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
KEYS = {f"k07@{G}": "G30", f"k07ros@{G}": "R", f"k07sup@{G};ROS=0,TR=0,UNS_COOL_K=305.4": "U", f"k07sup@{G};ROS=1,TR=0,UNS_COOL_K=305.4": "UR"}
coh = json.load(open(f"{N}/k07x/K07X_V2_COHORT.json"))["cells"]; DEV = {(c["architecture"], c["weather"]) for c in coh if c["split"] == "dev"}; grp = {c["architecture"]: c["group"] for c in coh}
OFF = [("building_train_1", "denver_7"), ("building_train_1", "fairbanks_1"), ("building_train_1", "phoenix_5"), ("building_train_1", "singapore_8"), ("building_train_2", "fairbanks_2"), ("building_train_4", "phoenix_8")]
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
def load(patterns, cells):
    L, F = {}, {}
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            s = json.load(open(f)); p = KEYS.get(s["policy_key"]); c = (s["building"], s["weather"])
            if p is None or c not in cells or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
            if (*c, p) in L: raise SystemExit(f"duplicate run {(*c, p)}: {F[(*c, p)]} and {f}")
            L[(*c, p)] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); F[(*c, p)] = f
    return L, F
def comp_dev(L, x, ref):
    cells = sorted(c for c in DEV if (*c, x) in L and (*c, ref) in L); archs = sorted({a for a, w in cells}); co = [a for a in archs if grp[a] == "composite"]
    if not cells: return {"cells": 0, "status": "INCOMPLETE"}
    ra = {a: sum(L[(a, w, x)] for aa, w in cells if aa == a) / sum(L[(a, w, ref)] for aa, w in cells if aa == a) - 1 for a in archs}
    pc = sum(L[(a, w, x)] for a, w in cells if a in co) / sum(L[(a, w, ref)] for a, w in cells if a in co) - 1 if co else None
    return {"cells": len(cells), "status": "COMPLETE" if len(cells) == len(DEV) else "INCOMPLETE", "per_architecture_pct": {a: 100 * v for a, v in ra.items()},
            "composite_mean_pct": 100 * float(np.mean([ra[a] for a in co])) if co else None, "composite_pooled_pct": None if pc is None else 100 * pc,
            "per_scene_pct": {f"{a}|{w}": 100 * (L[(a, w, x)] / L[(a, w, ref)] - 1) for a, w in cells}, "changed_architectures": sum(1 for v in ra.values() if v != 0.0),
            "better_architectures": sum(1 for v in ra.values() if v < 0.0), "worst_architecture_pct": 100 * max(ra.values())}
def comp_off(L, x, ref):
    cells = [c for c in OFF if (*c, x) in L and (*c, ref) in L]
    if not cells: return {"cells": 0, "status": "INCOMPLETE"}
    rs = {f"{b}|{w}": L[(b, w, x)] / L[(b, w, ref)] - 1 for b, w in cells}
    return {"cells": len(cells), "status": "COMPLETE" if len(cells) == len(OFF) else "INCOMPLETE", "per_scene_pct": {k: 100 * v for k, v in rs.items()},
            "mean_scene_pct": 100 * float(np.mean(list(rs.values()))), "pooled_pct": 100 * (sum(L[(*c, x)] for c in cells) / sum(L[(*c, ref)] for c in cells) - 1),
            "changed_scenes": sum(1 for v in rs.values() if v != 0.0), "better_scenes": sum(1 for v in rs.values() if v < 0.0), "worst_scene_pct": 100 * max(rs.values())}
def interaction(L, cells):
    cs = [c for c in cells if all((*c, p) in L for p in ("G30", "R", "U", "UR"))]
    if len(cs) != len(cells): return {"cells": len(cs), "status": "INCOMPLETE"}
    g = sum(L[(*c, "G30")] for c in cs); f = lambda p: sum(L[(*c, p)] for c in cs)
    return {"cells": len(cs), "status": "COMPLETE", "U_minus_G30_pct": 100 * (f("U") - g) / g, "R_minus_G30_pct": 100 * (f("R") - g) / g, "UR_minus_U_pct": 100 * (f("UR") - f("U")) / g,
            "interaction_pct": 100 * ((f("UR") - f("U")) - (f("R") - g)) / g, "note": "pooled over the cells, in % of the G30 pooled loss"}
def components(Fo, x, ref):
    acc = collections.defaultdict(float); tot = 0.0; n = 0
    for c in OFF:
        if (*c, x) not in Fo or (*c, ref) not in Fo: continue
        n += 1
        for p, sgn in ((x, 1.0), (ref, -1.0)):
            z = np.load(Fo[(*c, p)].replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(k) for k in z["comp_keys"]]; cc = {k: z["comps"][:, ck.index(k)].astype(float) for k in ck}; occ = cc["total_occupancy"] > 0
            for nm, (k, wt) in W.items():
                v = np.abs(wt * cc[k]); acc[f"{nm}_occupied"] += sgn * float(v[occ].sum()); acc[f"{nm}_unoccupied"] += sgn * float(v[~occ].sum())
            if p == ref: tot += float(np.abs(z["rewards"]).sum())
    return {"scenes": n, "change_pct_of_ref_total": {k: 100 * v / tot for k, v in acc.items()} if tot else None}
def behaviour(Fs, cells):
    ext = collections.Counter(); leads = collections.defaultdict(list); hot = {}
    for c in cells:
        if (*c, "UR") in Fs:
            s = json.load(open(Fs[(*c, "UR")])); z = np.load(Fs[(*c, "UR")].replace(".summary.json", ".npz"), allow_pickle=True); oat = z["oat_c"]
            for e in (s.get("ros") or {}).get("log", []):
                if e["event"] == "start" and e.get("why") == "need":
                    k = "warm" if float(oat[e["step"]]) >= 11.0 else "cold"; ext[k] += 1; leads[k].append(float(e["tod_h"]))
        for p in ("G30", "U"):
            if (*c, p) not in Fs: continue
            z = np.load(Fs[(*c, p)].replace(".summary.json", ".npz"), allow_pickle=True)
            if "pi_active_c" not in z.files or "zone_t_k" not in z.files: continue
            pi = z["pi_active_c"].astype(bool); zt = z["zone_t_k"].astype(float)
            m = np.nanmean(zt, axis=1); un = ~pi; hot[f"{c[0]}|{c[1]}|{p}"] = float(np.mean(m[un] > 298.0)) if un.any() else None
    return {"UR_need_extensions_by_morning_oat": dict(ext), "UR_need_start_hour_mean_by_morning_oat": {k: float(np.mean(v)) for k, v in leads.items()}, "share_unoccupied_steps_mean_zone_above_298K": hot}
res = {}
Ld, Fd = load([f"{N}/results_k07x2/**/*.summary.json", f"{N}/results_k07ros/**/*.summary.json", f"{N}/results_k07uns_dev/**/*.summary.json"], DEV)
Lo, Fo = load([f"{H}/laptop/work_off*/out/**/*.summary.json"], set(OFF))
gd, go = comp_dev(Ld, "UR", "G30"), comp_off(Lo, "UR", "G30")
res["gate_D_dev_UR_vs_G30"] = {**gd, "rule": {"composite_mean_lt_0": gd.get("composite_mean_pct") is not None and gd["composite_mean_pct"] < 0,
                                              "every_dev_architecture_le_1pct": bool(gd.get("per_architecture_pct")) and all(v <= 1.0 for v in gd["per_architecture_pct"].values()),
                                              "all_18_cells": gd["cells"] == len(DEV)}}
res["gate_O_official_UR_vs_G30"] = {**go, "rule": {"mean_scene_lt_0": go.get("mean_scene_pct") is not None and go["mean_scene_pct"] < 0,
                                                   "every_scene_le_1pct": bool(go.get("per_scene_pct")) and all(v <= 1.0 for v in go["per_scene_pct"].values()),
                                                   "all_6_scenes": go["cells"] == len(OFF)}}
res["GATE_COMPLETE"] = gd["cells"] == len(DEV) and go["cells"] == len(OFF)
res["GATE_PASS"] = res["GATE_COMPLETE"] and all(res["gate_D_dev_UR_vs_G30"]["rule"].values()) and all(res["gate_O_official_UR_vs_G30"]["rule"].values())
for x, ref in (("U", "G30"), ("UR", "U"), ("R", "G30")):
    res[f"dev_{x}_vs_{ref}"] = comp_dev(Ld, x, ref); res[f"official_{x}_vs_{ref}"] = comp_off(Lo, x, ref)
res["dev_interaction_composite"] = interaction(Ld, sorted(c for c in DEV if grp[c[0]] == "composite")); res["official_interaction"] = interaction(Lo, OFF)
res["official_components"] = {f"{x}_vs_{ref}": components(Fo, x, ref) for x, ref in (("U", "G30"), ("UR", "U"), ("UR", "G30"))}
res["behaviour_dev"] = behaviour(Fd, sorted(DEV)); res["behaviour_official"] = behaviour(Fo, OFF)
res["runs_found"] = {"dev": collections.Counter(k[2] for k in Ld), "official": collections.Counter(k[2] for k in Lo)}
json.dump(res, open(OUT, "w"), indent=1)
print("GATE_COMPLETE", res["GATE_COMPLETE"], "GATE_PASS", res["GATE_PASS"], "| runs", dict(res["runs_found"]["dev"]), dict(res["runs_found"]["official"]))
print("D dev UR vs G30: composite mean %", gd.get("composite_mean_pct"), "| worst arch %", gd.get("worst_architecture_pct"), "| rule", res["gate_D_dev_UR_vs_G30"]["rule"])
print("O official UR vs G30: mean scene %", go.get("mean_scene_pct"), "| worst scene %", go.get("worst_scene_pct"), "| rule", res["gate_O_official_UR_vs_G30"]["rule"])
for k in ("dev_U_vs_G30", "dev_UR_vs_U", "dev_R_vs_G30"): print(k, res[k].get("status"), "composite mean %", res[k].get("composite_mean_pct"), "| pooled %", res[k].get("composite_pooled_pct"), "| better/changed", res[k].get("better_architectures"), res[k].get("changed_architectures"))
for k in ("official_U_vs_G30", "official_UR_vs_U", "official_R_vs_G30"): print(k, res[k].get("status"), "mean scene %", res[k].get("mean_scene_pct"), "| per scene", {a: round(v, 2) for a, v in (res[k].get("per_scene_pct") or {}).items()})
print("interaction dev", res["dev_interaction_composite"], "| official", res["official_interaction"])
