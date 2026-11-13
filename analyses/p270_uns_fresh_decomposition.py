"""p270: why the unoccupied setup did not generalise on the blind set (post hoc, descriptive; PREREG_K07UNS_v1 'reported, never gating':
component decomposition).  The frozen p248 result (RESULT_K07UNS_FRESH.json) is read first and every pooled number below must equal it.
Contrasts on the 10 composite architectures (pooled) and by day-1 class: U vs G30 (the fixed G36 setup), UR vs U (learned start on top),
UR vs G30.  Change of each weighted component (comfort / energy / carbon x occupied / unoccupied) in % of the comparator's loss, as p254.
Loading as p248 (manifests K07X3_RUN_MANIFEST + K07UNS_FRESH_MANIFEST; results_k07x3 and results_k07uns_fresh; status OK, 8640 steps).
usage: python -I p270_uns_fresh_decomposition.py <out_dir>"""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
man = list(csv.DictReader(open(f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv"))) + list(csv.DictReader(open(f"{N}/k07uns/K07UNS_FRESH_MANIFEST.csv")))
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
L, F = {}, {}
for d in ("results_k07x3", "results_k07uns_fresh"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in ("G30", "U", "UR") or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); F[k] = f
def comps(f):
    z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]
    c = {k: z["comps"][:, ck.index(k)].astype(float) for k in ck}; occ = c["total_occupancy"] > 0
    return {nm: (float(np.abs(wt * c[k])[occ].mean() * occ.mean()), float(np.abs(wt * c[k])[~occ].mean() * (~occ).mean())) for nm, (k, wt) in W.items()}
cells = sorted({(a, w) for a, w, p in L if all((a, w, q) in L for q in ("G30", "U", "UR"))})
comp = [c for c in cells if grp[c[0]] == "composite"]; assert len(comp) == 30, len(comp)
C = {(a, w, p): comps(F[(a, w, p)]) for a, w in comp for p in ("G30", "U", "UR")}
def split(x, y, sel):
    tot = sum(L[(a, w, y)] for a, w in sel); acc = collections.defaultdict(float)
    for a, w in sel:
        for nm in W:
            acc[f"{nm}_occupied"] += C[(a, w, x)][nm][0] - C[(a, w, y)][nm][0]; acc[f"{nm}_unoccupied"] += C[(a, w, x)][nm][1] - C[(a, w, y)][nm][1]
    out = {k: 100 * v / tot for k, v in acc.items()}; out["total"] = 100 * (sum(L[(a, w, x)] for a, w in sel) / tot - 1)
    return out
ref = json.load(open(f"{N}/k07uns/RESULT_K07UNS_FRESH.json"))
R = {"note": "post hoc, descriptive; fresh blind set, composite; % of the comparator's pooled loss (positive = worse)", "contrasts": {}}
for x, y in (("U", "G30"), ("UR", "U"), ("UR", "G30")):
    allc = split(x, y, comp); assert abs(allc["total"] - ref[f"{x}_vs_{y}"]["composite_pooled_pct"]) < 1e-6, (x, y, allc["total"])
    R["contrasts"][f"{x}_vs_{y}"] = {"composite": allc, "by_class": {k: split(x, y, [c for c in comp if cls[c] == k]) for k in ("C", "M", "H")}}
json.dump(R, open(f"{OUT}/P270_UNS_FRESH_DECOMP.json", "w"), indent=1)
for k, v in R["contrasts"].items():
    for part in ("composite", "H", "C"):
        d = v["composite"] if part == "composite" else v["by_class"][part]
        print(f"{k:10s} {part:9s} total {d['total']:+.2f} | comfort occ {d['comfort_occupied']:+.2f} unocc {d['comfort_unoccupied']:+.2f} | energy occ {d['energy_occupied']:+.2f} unocc {d['energy_unoccupied']:+.2f} | carbon occ {d['carbon_occupied']:+.2f} unocc {d['carbon_unoccupied']:+.2f}")
print("pooled totals equal p248: OK")
