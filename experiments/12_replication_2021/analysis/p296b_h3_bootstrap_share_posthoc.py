"""p296b (POST HOC, descriptive; after the H3 decision was read): the share of the 3,000 station-cluster bootstrap replicates of the H3
primary (the same replicates as p296: seed 1, the functions of p128) that lie above zero, and the corresponding one-sided p.  It
changes nothing in the registered decision (INCONCLUSIVE).  usage: python -I p296b_h3_bootstrap_share_posthoc.py <out.json>"""
import sys, os, json, csv, numpy as np
N = os.path.dirname(os.path.abspath(__file__)); src = open(f"{N}/p296_h3_analysis.py").read()
ns = {"__file__": f"{N}/p296_h3_analysis.py", "__name__": "p296b"}
exec(src[:src.index("C3 = list(")], ns)                 # p296's loader, arms and functions, unchanged
C3 = list(csv.DictReader(open(f"{N}/cells_h3.csv"))); L3, I3, ok3 = ns["load"](f"{N}/results_h3", "h3*", C3, ns["A3"])
M, st, groups, prim = ns["block"](ok3, L3)
rng = np.random.default_rng(1); us = sorted(groups); b = []                 # the replicates of p296's ci(), seed 1, 3000
for _ in range(3000):
    p = rng.choice(len(us), len(us)); b.append(prim(np.concatenate([groups[us[i]] for i in p])))
b = np.array(b); est = float(prim(np.arange(len(ok3))))
res = {"note": "post hoc, descriptive", "estimate": round(est, 3), "ci95": [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)],
       "share_of_replicates_above_0": round(float((b > 0).mean()), 3), "one_sided_p_bootstrap": round(float((b <= 0).mean()), 3)}
assert res["estimate"] == 0.255 and res["ci95"] == [-0.037, 0.709], res
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res))
