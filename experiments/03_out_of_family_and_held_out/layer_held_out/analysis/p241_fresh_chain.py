"""Fresh-set chain analysis (PREREG_K07X3H_v1); written before any B or ROSFIX fresh outcome.  usage: python -I p241_fresh_chain.py <out.json>
Reads only when all 36 B and 36 ROSFIX fresh runs are complete (plus G30, ROS, SUP from stages 1-2).  Loss per run = mean |reward|; per architecture r = sum over its 3 strips.
Comparisons: G30 vs B (selection transfer; rule as the K07-X held-out: composite pooled < 0, >= 6 of 10 composite better, no architecture worse than +2 %), SUP vs B (whole chain),
ROSFIX vs G30 (mechanism alone), ROS vs ROSFIX (online learning).  Composite pooled, OOD, by day-1 class, per architecture."""
import sys, json, glob, csv, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
man = list(csv.DictReader(open(f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv"))) + list(csv.DictReader(open(f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"))) + list(csv.DictReader(open(f"{H}/K07X3H_RUN_MANIFEST.csv")))
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
L = {}
for d in ("results_k07x3", "results_k07x3h"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        L[(r["architecture"], r["weather"], r["policy"])] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
n = {p: sum(1 for k in L if k[2] == p) for p in ("B", "G30", "ROSFIX", "ROS", "SUP")}
if min(n.values()) < 36: json.dump({"status": f"INCOMPLETE {n}"}, open(OUT, "w")); sys.exit(print("incomplete", n))
def comp(x, ref):
    cells = sorted({(a, w) for a, w, p in L if p == x and (a, w, ref) in L}); archs = sorted({a for a, w in cells}); co = [a for a in archs if grp[a] == "composite"]; od = [a for a in archs if grp[a] == "ood"]
    ra = {a: sum(L[(a, w, x)] for aa, w in cells if aa == a) / sum(L[(a, w, ref)] for aa, w in cells if aa == a) - 1 for a in archs}
    pooled = lambda S: (sum(L[(a, w, x)] for a, w in cells if a in S) / sum(L[(a, w, ref)] for a, w in cells if a in S) - 1) if S else None
    byc = {c: 100 * (sum(L[(a, w, x)] for a, w in cells if cls[(a, w)] == c) / sum(L[(a, w, ref)] for a, w in cells if cls[(a, w)] == c) - 1) for c in ("C", "M", "H")}
    pc = pooled(co); rule = {"composite_pooled_lt_0": pc < 0, "composite_better_ge_6": sum(1 for a in co if ra[a] < 0) >= 6, "no_architecture_worse_than_2pct": max(ra.values()) <= 0.02}
    return {"composite_pooled_pct": 100 * pc, "ood_pooled_pct": 100 * pooled(od) if od else None, "composite_better": sum(1 for a in co if ra[a] < 0), "by_day1_class_pct": byc,
            "worst_architecture_pct": 100 * max(ra.values()), "per_architecture_pct": {a: 100 * v for a, v in ra.items()}, "rule": rule, "PASS": all(rule.values())}
res = {"runs": n, "G30_vs_B": comp("G30", "B"), "SUP_vs_B": comp("SUP", "B"), "ROSFIX_vs_G30": comp("ROSFIX", "G30"), "ROS_vs_ROSFIX": comp("ROS", "ROSFIX")}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("G30_vs_B", "SUP_vs_B", "ROSFIX_vs_G30", "ROS_vs_ROSFIX"):
    v = res[k]; print(k, "| composite pooled %", round(v["composite_pooled_pct"], 3), "| better", v["composite_better"], "/ 10 | OOD %", v["ood_pooled_pct"], "| by class", {c: round(x, 2) for c, x in v["by_day1_class_pct"].items()}, "| worst", round(v["worst_architecture_pct"], 2), "| PASS", v["PASS"])
