"""Fresh blind test analysis (PREREG_K07X3_FRESH_v1 + PREREG_K07SUP_COMBINED_v1 (b)); written and frozen before any fresh outcome.  usage: python -I p226_fresh_analysis.py <out.json>
Loss per run = mean |reward| over 8640 steps.  For each arm X in (ROS, SUP) vs G30, and SUP vs ROS: per architecture r = sum over its 3 strips L_X / sum L_ref - 1; composite (10)
pooled change and count better; OOD (2) separately; by day-1 class (C, M, H) pooled over all architectures; worst architecture; per scene; behaviour (ROS extended mornings and the
class of those days, TR override share).  Rule (as the held-out rule): composite pooled < 0, >= majority (6 of 10) better, no architecture worse than +2 %."""
import sys, json, glob, csv, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
man = list(csv.DictReader(open(f"{H}/K07X3_RUN_MANIFEST.csv"))) + (list(csv.DictReader(open(f"{H}/K07X3_RUN_MANIFEST_S2.csv"))) if os.path.exists(f"{H}/K07X3_RUN_MANIFEST_S2.csv") else [])
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
L, BEH = {}, {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if not r or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
    k = (r["architecture"], r["weather"], r["policy"]); L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); b = {}
    if s.get("ros"): b["ros_extended"] = sum(1 for e in s["ros"]["log"] if e["event"] == "start" and e.get("why") == "need"); b["ros_frozen"] = s["ros"]["frozen"]; b["tau_h"] = s["ros"]["tau_final_h"]
    if s.get("tr"): b["tr_override_share"] = sum(s["tr"]["override_steps"]) / max(1, sum(s["tr"]["pi_steps"]))
    BEH[f"{k[0]}|{k[1]}|{k[2]}"] = b
def comp(x, ref):
    cells = sorted({(a, w) for a, w, p in L if p == x and (a, w, ref) in L}); archs = sorted({a for a, w in cells})
    ra = {a: sum(L[(a, w, x)] for aa, w in cells if aa == a) / sum(L[(a, w, ref)] for aa, w in cells if aa == a) - 1 for a in archs}
    co = [a for a in archs if grp[a] == "composite"]; od = [a for a in archs if grp[a] == "ood"]
    pooled = lambda S: (sum(L[(a, w, x)] for a, w in cells if a in S) / sum(L[(a, w, ref)] for a, w in cells if a in S) - 1) if S else None
    bycls = {c: (sum(L[(a, w, x)] for a, w in cells if cls[(a, w)] == c) / sum(L[(a, w, ref)] for a, w in cells if cls[(a, w)] == c) - 1) if any(cls[(a, w)] == c for a, w in cells) else None for c in ("C", "M", "H")}
    pc = pooled(co); rule = {"composite_pooled_lt_0": pc is not None and pc < 0, "composite_better_majority": sum(1 for a in co if ra[a] < 0) >= 6, "no_architecture_worse_than_2pct": all(v <= 0.02 for v in ra.values())}
    return {"cells": len(cells), "per_architecture_pct": {a: 100 * v for a, v in ra.items()}, "composite_pooled_pct": None if pc is None else 100 * pc, "composite_better": sum(1 for a in co if ra[a] < 0),
            "n_composite": len(co), "ood_pooled_pct": None if pooled(od) is None else 100 * pooled(od), "by_day1_class_pct": {c: (None if v is None else 100 * v) for c, v in bycls.items()},
            "worst_architecture_pct": 100 * max(ra.values()) if ra else None, "per_scene_pct": {f"{a}|{w}": 100 * (L[(a, w, x)] / L[(a, w, ref)] - 1) for a, w in cells}, "rule": rule, "PASS": all(rule.values())}
res = {"runs": len(L), "ROS_vs_G30": comp("ROS", "G30"), "SUP_vs_G30": comp("SUP", "G30"), "SUP_vs_ROS": comp("SUP", "ROS"), "behaviour": BEH}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("ROS_vs_G30", "SUP_vs_G30", "SUP_vs_ROS"):
    v = res[k]; print(k, "cells", v["cells"], "| composite pooled %", v["composite_pooled_pct"], "| better", v["composite_better"], "/", v["n_composite"], "| by class", v["by_day1_class_pct"], "| worst", v["worst_architecture_pct"], "| PASS", v["PASS"])
