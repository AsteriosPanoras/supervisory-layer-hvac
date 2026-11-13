"""Penguin supervisor analysis (PREREG_PGXSUP_v1).  usage: python -I p224_pgxsup_gate.py <dev|heldout> <out.json> <dir> [<dir> ...]
Files are filtered by the requested split's scene names BEFORE they are opened (the held-out split is run only after the development gate passed).
Per architecture (strips pooled): r = L_ROSP / L_V0 - 1.  dev: GATE = composite mean < 0 and every architecture <= +1 %.  heldout: composite pooled < 0, >= 3 of 4 better, none > +2 %."""
import sys, json, glob, csv, os, numpy as np
SPLIT, OUT = sys.argv[1], sys.argv[2]; DIRS = sys.argv[3:]; H = os.path.dirname(os.path.abspath(__file__))
man = [r for r in csv.DictReader(open(f"{H}/PGXSUP_EVAL_MANIFEST.csv")) if r["split"] == SPLIT]; sc = {(r["architecture"], r["weather"]) for r in man}; key = {r["policy"]: r["policy_key"] for r in man}; grp = {r["architecture"]: r["group"] for r in man}
L, BEH = {}, {}
for d in DIRS:
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        b = os.path.basename(f)
        if not any(b.startswith(f"{a}_{w}_") for a, w in sc): continue
        s = json.load(open(f))
        if (s["building"], s["weather"]) not in sc or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        p = next((q for q, k in key.items() if k == s["policy_key"]), None)
        if p is None: continue
        L[(s["building"], s["weather"], p)] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
        if p == "ROSP": st = [e for e in s["ros"]["log"] if e["event"] == "start"]; BEH[f"{s['building']}|{s['weather']}"] = {"starts": len(st), "mean_lead_h": float(np.mean([s["ros"]["opening_hour"] - e["tod_h"] for e in st])) if st else None, "tau_final_h": s["ros"]["tau_final_h"], "frozen": s["ros"]["frozen"]}
ok = [c for c in sorted(sc) if (*c, "ROSP") in L and (*c, "V0") in L]; archs = sorted({c[0] for c in ok})
ra = {a: sum(L[(*c, "ROSP")] for c in ok if c[0] == a) / sum(L[(*c, "V0")] for c in ok if c[0] == a) - 1 for a in archs}; comp = [a for a in archs if grp[a] == "composite"]
res = {"split": SPLIT, "cells": len(ok), "missing": [c for c in sorted(sc) if c not in ok], "r_ROSP_vs_V0_pct": {a: 100 * v for a, v in ra.items()},
       "per_cell_pct": {f"{c[0]}|{c[1]}": 100 * (L[(*c, "ROSP")] / L[(*c, "V0")] - 1) for c in ok}, "behaviour": BEH}
if SPLIT == "dev":
    cm = float(np.mean([ra[a] for a in comp])) if comp else None
    res.update({"composite_mean_pct": None if cm is None else 100 * cm, "gate": {"composite_mean_lt_0": cm is not None and cm < 0, "every_dev_architecture_le_1pct": all(v <= 0.01 for v in ra.values()), "all_18_cells": len(ok) == 18}})
    res["GATE_PASS"] = all(res["gate"].values())
else:
    pooled = sum(L[(*c, "ROSP")] for c in ok if c[0] in comp) / sum(L[(*c, "V0")] for c in ok if c[0] in comp) - 1 if comp else None
    rule = {"composite_pooled_lt_0": pooled is not None and pooled < 0, "composite_better_ge_3_of_4": sum(1 for a in comp if ra[a] < 0) >= 3, "no_architecture_worse_than_2pct": all(v <= 0.02 for v in ra.values())}
    res.update({"composite_pooled_pct": None if pooled is None else 100 * pooled, "rule": rule, "ROSP_PASSES_HELDOUT": all(rule.values())})
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k not in ("per_cell_pct", "behaviour")})[:900])
