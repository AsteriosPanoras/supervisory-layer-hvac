"""Fresh blind test of the K07 unoccupied setup with and without the learned start (PREREG_K07UNS_v1); written 2026-11-11 before any UNS outcome.
usage: python -I p248_uns_fresh.py <out.json>
Runs only if RESULT_K07UNS_GATE.json says GATE_PASS, PARITY_D0.txt says PASS, and all 36 U and 36 UR fresh runs are complete; otherwise it reads no fresh U/UR file.
Loss per run = mean |reward| over 8640 steps; per architecture r = sum over its 3 strips L_X / sum L_ref - 1.  Composite (10) pooled, OOD (2) separately, by day-1 class.
Rule 'UR generalises' (UR vs G30) and rule 'mechanism adds' (UR vs U): composite pooled < 0 AND a majority of the changed composite architectures better AND no architecture
worse than +1 % vs the reference.  Also U vs G30, R vs G30 (known), and the interaction (UR - U) - (R - G30) pooled over the composite cells in % of G30."""
import sys, json, glob, csv, os, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
g = json.load(open(f"{H}/RESULT_K07UNS_GATE.json")) if os.path.exists(f"{H}/RESULT_K07UNS_GATE.json") else {}
if not g.get("GATE_PASS"): json.dump({"status": "NOT RUN: gate not passed"}, open(OUT, "w")); sys.exit(print("gate not passed: fresh U/UR not read"))
par = open(f"{H}/PARITY_D0.txt").read().strip() if os.path.exists(f"{H}/PARITY_D0.txt") else "MISSING"
if not par.startswith("PASS"): json.dump({"status": f"NOT RUN: Kaggle parity {par}"}, open(OUT, "w")); sys.exit(print("Kaggle parity not passed: fresh U/UR not read", par))
man = list(csv.DictReader(open(f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv"))) + list(csv.DictReader(open(f"{H}/K07UNS_FRESH_MANIFEST.csv")))
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
pol = {"G30": "G30", "ROS": "R", "U": "U", "UR": "UR"}
L, S = {}, {}
for d in ("results_k07x3", "results_k07uns_fresh"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in pol or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], pol[r["policy"]])
        if k in L: raise SystemExit(f"duplicate run {k}")
        L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); S[k] = s
n = {p: sum(1 for k in L if k[2] == p) for p in ("G30", "R", "U", "UR")}
if n["U"] < 36 or n["UR"] < 36 or n["G30"] < 36: json.dump({"status": f"INCOMPLETE {n}"}, open(OUT, "w")); sys.exit(print("incomplete", n))
def comp(x, ref):
    cells = sorted({(a, w) for a, w, p in L if p == x and (a, w, ref) in L}); archs = sorted({a for a, w in cells}); co = [a for a in archs if grp[a] == "composite"]; od = [a for a in archs if grp[a] == "ood"]
    ra = {a: sum(L[(a, w, x)] for aa, w in cells if aa == a) / sum(L[(a, w, ref)] for aa, w in cells if aa == a) - 1 for a in archs}
    pooled = lambda A: (sum(L[(a, w, x)] for a, w in cells if a in A) / sum(L[(a, w, ref)] for a, w in cells if a in A) - 1) if A else None
    bycls = {c: (sum(L[(a, w, x)] for a, w in cells if cls[(a, w)] == c) / sum(L[(a, w, ref)] for a, w in cells if cls[(a, w)] == c) - 1) for c in ("C", "M", "H") if any(cls[(a, w)] == c for a, w in cells)}
    ch = [a for a in co if ra[a] != 0.0]
    return {"cells": len(cells), "per_architecture_pct": {a: 100 * v for a, v in ra.items()}, "composite_pooled_pct": 100 * pooled(co), "ood_pooled_pct": None if not od else 100 * pooled(od),
            "by_day1_class_pct": {c: 100 * v for c, v in bycls.items()}, "composite_changed": len(ch), "composite_changed_better": sum(1 for a in ch if ra[a] < 0),
            "worst_architecture_pct": 100 * max(ra.values()), "per_scene_pct": {f"{a}|{w}": 100 * (L[(a, w, x)] / L[(a, w, ref)] - 1) for a, w in cells}}
def rule(v): return {"composite_pooled_lt_0": v["composite_pooled_pct"] < 0, "majority_of_changed_better": v["composite_changed"] > 0 and 2 * v["composite_changed_better"] > v["composite_changed"],
                     "no_architecture_worse_than_1pct": v["worst_architecture_pct"] <= 1.0}
res = {"runs": n, "UR_vs_G30": comp("UR", "G30"), "UR_vs_U": comp("UR", "U"), "U_vs_G30": comp("U", "G30"), **({"R_vs_G30": comp("R", "G30")} if n["R"] >= 36 else {})}
res["rule_UR_generalises"] = rule(res["UR_vs_G30"]); res["UR_GENERALISES"] = all(res["rule_UR_generalises"].values())
res["rule_mechanism_adds"] = rule(res["UR_vs_U"]); res["MECHANISM_ADDS"] = all(res["rule_mechanism_adds"].values())
co = sorted({(a, w) for a, w, p in L if grp[a] == "composite" and all((a, w, q) in L for q in ("G30", "R", "U", "UR"))}); gs = sum(L[(*c, "G30")] for c in co); f = lambda p: sum(L[(*c, p)] for c in co)
res["interaction_composite"] = {"cells": len(co), "U_minus_G30_pct": 100 * (f("U") - gs) / gs, "R_minus_G30_pct": 100 * (f("R") - gs) / gs, "UR_minus_U_pct": 100 * (f("UR") - f("U")) / gs,
                                "interaction_pct": 100 * ((f("UR") - f("U")) - (f("R") - gs)) / gs} if co else None
ext = collections.Counter()
for (a, w, p), s in S.items():
    if p != "UR": continue
    for e in (s.get("ros") or {}).get("log", []):
        if e["event"] == "start" and e.get("why") == "need": ext["need"] += 1
res["behaviour"] = {"UR_need_extensions": ext["need"]}
json.dump(res, open(OUT, "w"), indent=1)
for k in [x for x in ("UR_vs_G30", "UR_vs_U", "U_vs_G30", "R_vs_G30") if x in res]:
    v = res[k]; print(k, "| composite pooled %", round(v["composite_pooled_pct"], 3), "| OOD %", v["ood_pooled_pct"], "| by class", {c: round(x, 3) for c, x in v["by_day1_class_pct"].items()}, "| changed/better", v["composite_changed"], v["composite_changed_better"], "| worst", round(v["worst_architecture_pct"], 3))
print("UR_GENERALISES", res["UR_GENERALISES"], res["rule_UR_generalises"]); print("MECHANISM_ADDS", res["MECHANISM_ADDS"], res["rule_mechanism_adds"]); print("interaction", res["interaction_composite"], "| behaviour", res["behaviour"])
