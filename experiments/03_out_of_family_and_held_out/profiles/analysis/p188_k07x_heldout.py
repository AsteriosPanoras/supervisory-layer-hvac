"""K07-X v2 held-out + within-family analysis (written before the selection is known; frozen with S before any held-out run).
usage: python -I p188_k07x_heldout.py <results_dir> <heldout_manifest.csv> <wf_manifest.csv> <selection.json> out.json
manifest columns: architecture, group (composite|ood|within_family), weather, policy, policy_key.  Loss per cell = mean |reward| over 8640 steps.
PRIMARY: S vs B on the held-out COMPOSITE architectures: r_a = sum over a's 2 strips L_S / sum L_B - 1; mean over architectures (equal weight), percentile bootstrap over architectures
(10000, seed 1; 4 clusters = coarse), sign count over architectures and cells, worst architecture.  If S == B the primary is 'no candidate selected' and every policy is reported vs B.
SECONDARY (separate, never pooled into the primary): OOD held-out; S vs G30 and vs MG; all policies vs B per group and per strip class (C = minneapolis_1, M = minneapolis_5);
comfort / energy / carbon split; adverse cells; within-family set."""
import json, glob, csv, sys, numpy as np
R_DIR, HOM, WFM, SEL, OUT = sys.argv[1:6]; sel = json.load(open(SEL)); S = sel["S"]; rng = np.random.default_rng(1)
W = {"energy": ("normalized_energy_cost", -0.2), "carbon": ("normalized_carbon_emission", -0.2), "comfort": ("normalized_productivity_regret", 0.6)}
man = list(csv.DictReader(open(HOM))) + list(csv.DictReader(open(WFM))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
L, CP, INT = {}, {}, {"runs": 0, "ok": 0, "illegal": 0, "not_ok": []}
for f in glob.glob(f"{R_DIR}/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if not r: continue
    INT["runs"] += 1
    if s["status"] == "OK" and s["steps_done"] == 8640 and s.get("illegal_actions", 0) == 0:
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; k = (r["architecture"], r["weather"], r["policy"])
        L[k] = float(np.abs(z["rewards"]).mean()); CP[k] = {nm: float((-w * z["comps"][:, ck.index(kk)]).mean()) for nm, (kk, w) in W.items()}; INT["ok"] += 1
    else: INT["not_ok"].append((s["building"], s["weather"], s["policy_key"][:30], s["status"], s["steps_done"])); INT["illegal"] += int(s.get("illegal_actions", 0) or 0)
def group(g): return sorted({r["architecture"] for r in man if r["group"] == g})
def pols(g): return sorted({r["policy"] for r in man if r["group"] == g}, key=lambda p: ["B", "G30", "MG", "MGG30"].index(p))
def strips(a): return sorted({r["weather"] for r in man if r["architecture"] == a})
def r_arch(a, p, q="B", ws=None):
    ws = ws or strips(a)
    if not all((a, w, x) in L for w in ws for x in (p, q)): return None
    return sum(L[(a, w, p)] for w in ws) / sum(L[(a, w, q)] for w in ws) - 1
def boot(v):
    v = np.array([x for x in v if x is not None]); 
    if len(v) < 2: return None
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)]); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def contrast(g, p, q="B"):
    A = group(g); ra = {a: r_arch(a, p, q) for a in A}; v = [x for x in ra.values() if x is not None]
    cells = [(a, w) for a in A for w in strips(a) if (a, w, p) in L and (a, w, q) in L]; rc = [L[(a, w, p)] / L[(a, w, q)] - 1 for a, w in cells]
    split = {nm: float(np.mean([CP[(a, w, p)][nm] - CP[(a, w, q)][nm] for a, w in cells])) for nm in W} if cells else {}
    by_strip = {w: float(np.mean([L[(a, w, p)] / L[(a, w, q)] - 1 for a in A if (a, w, p) in L and (a, w, q) in L])) for w in sorted({w for a in A for w in strips(a)})}
    return {"per_architecture": ra, "mean_over_architectures": float(np.mean(v)) if v else None, "bootstrap_over_architectures_95": boot(v), "architectures_better": sum(1 for x in v if x < 0), "n_architectures": len(v),
            "cells_better": sum(1 for x in rc if x < 0), "n_cells": len(rc), "pooled": (sum(L[(a, w, p)] for a, w in cells) / sum(L[(a, w, q)] for a, w in cells) - 1) if cells else None,
            "worst_architecture": (max(ra, key=lambda a: ra[a] if ra[a] is not None else -9), max(v)) if v else None, "by_strip": by_strip, "loss_split_mean_diff": split,
            "adverse_cells_gt0.5pct": [(a, w, round(100 * (L[(a, w, p)] / L[(a, w, q)] - 1), 2)) for a, w in cells if L[(a, w, p)] / L[(a, w, q)] - 1 > 0.005]}
res = {"S": S, "GO_in_development": sel.get("GO"), "integrity": INT, "primary": None, "secondary": {}}
if S != "B": res["primary"] = {"contrast": f"{S} - B on held-out composite", **contrast("composite", S)}
else: res["primary"] = {"contrast": "no candidate selected (S = B)"}
for g in ("composite", "ood", "within_family"):
    for p in pols(g):
        if p != "B": res["secondary"][f"{g}: {p} - B"] = contrast(g, p)
    if S != "B" and S in pols(g):
        for q in pols(g):
            if q not in ("B", S): res["secondary"][f"{g}: {S} - {q}"] = contrast(g, S, q)
json.dump(res, open(OUT, "w"), indent=1, default=str)
p = res["primary"]; print("S =", S, "| integrity", {k: v for k, v in INT.items() if k != "not_ok"}, "| PRIMARY", p.get("contrast"), p.get("mean_over_architectures"), p.get("bootstrap_over_architectures_95"), "arch better", p.get("architectures_better"), "/", p.get("n_architectures"))
for k, v in res["secondary"].items(): print("  ", k, "mean", None if v["mean_over_architectures"] is None else round(100 * v["mean_over_architectures"], 2), "% | arch better", v["architectures_better"], "/", v["n_architectures"])
