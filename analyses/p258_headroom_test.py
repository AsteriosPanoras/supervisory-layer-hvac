"""Prospective test of the headroom account (PREDICTION_ZTSUP_HEADROOM_v1); written 2026-11-11 ~13:15, before any outcome of the maintenance test (ZT) was read.
usage: python -I p258_headroom_test.py k07 <out.json>   reference: K07-G30 -> K07-G30 + layer on the K07-X3 blind set (post hoc; these runs were read before)
       python -I p258_headroom_test.py zt  <out.json>   test: ZT Best (ZB) -> ZT Best + the unchanged layer (ZS); run only after all 72 runs are collected
Per episode e (every episode with both arms, composite and OOD):
  base comfort share s_e = weighted comfort / weighted total of the BASE arm (competition weights 0.6 / 0.2 / 0.2; mean |weighted component| over the 8640 steps);
  gain g_e = 100 (1 - L_layer / L_base), L = mean |reward| over the 8640 steps (as p252).
P1: Spearman rho(g, s) > 0 with one-sided permutation p < 0.05 (10,000 permutations, seed 0).
P2: pooled gain of the third of the episodes with the lowest s lies within [-0.5, +0.5] %.
P3: pooled gain over all episodes is below the K07 reference pooled gain if the mean s is below the K07 reference mean s, and above it otherwise
    (the reference is the k07 run of this same script, out/RESULT_HEADROOM_K07.json, hashed in the ledger before the test).
Loading and filters follow p252 (status OK, 8640 steps, no illegal action, duplicates abort).  In zt mode nothing is computed unless the fidelity file says ALL_PASS and
36 complete episodes exist; the 72 runs come from host-B only (results_ztsup_fresh; the laptop and Kaggle ZT stages were not used, NOTES_ZTSUP_EXECUTION)."""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); MODE, OUT = sys.argv[1], sys.argv[2]
assert MODE in ("k07", "zt"), MODE
BASE, LAYER = ("G30", "SUP") if MODE == "k07" else ("ZB", "ZS")
man = [r for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv") for r in csv.DictReader(open(f))]
key2 = {(r["architecture"], r["weather"], r["policy_key"], r["policy"] in ("ZB", "ZS")): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
if MODE == "zt":
    if not json.load(open(f"{N}/ztsup/RESULT_ZTSUP_FIDELITY.json")).get("ALL_PASS"): sys.exit("fidelity not passed: nothing read")
    PATS = ((f"{N}/results_ztsup_fresh/**/*.summary.json", True),)
else:
    PATS = ((f"{N}/results_k07x3/**/*.summary.json", False),)
W = {"normalized_productivity_regret": 0.6, "normalized_energy_cost": 0.2, "normalized_carbon_emission": 0.2}
L, C = {}, {}
for pat, zt in PATS:
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"], zt))
        if not r or r["policy"] not in (BASE, LAYER) or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean())
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {c: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for c, wt in W.items()}
cells = sorted({(a, w) for a, w, p in L if (a, w, BASE) in L and (a, w, LAYER) in L})
if MODE == "zt" and len(cells) < 36:
    json.dump({"status": f"INCOMPLETE: {len(cells)} episodes"}, open(OUT, "w")); sys.exit(f"incomplete: {len(cells)} episodes")
s = np.array([C[(a, w, BASE)]["normalized_productivity_regret"] / sum(C[(a, w, BASE)].values()) for a, w in cells])
g = np.array([100 * (1 - L[(a, w, LAYER)] / L[(a, w, BASE)]) for a, w in cells])
dev = max(abs(sum(C[(a, w, p)].values()) / L[(a, w, p)] - 1) for a, w in cells for p in (BASE, LAYER))
rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
rho = spearman(g, s); rng = np.random.default_rng(0); perm = np.array([spearman(rng.permutation(g), s) for _ in range(10000)])
p1 = float((1 + (perm >= rho).sum()) / (1 + len(perm)))
def pooled(idx): idx = list(idx); return float(100 * (1 - sum(L[(*cells[i], LAYER)] for i in idx) / sum(L[(*cells[i], BASE)] for i in idx)))
order = np.argsort(s, kind="stable"); n3 = len(cells) // 3; thirds = [order[:n3], order[n3:2 * n3], order[2 * n3:]]
comp = [i for i, (a, w) in enumerate(cells) if grp[a] == "composite"]
res = {"mode": MODE, "base": BASE, "layer": LAYER, "episodes": len(cells), "max_rel_dev_weighted_components_vs_reward": dev,
       "mean_share": float(s.mean()), "pooled_gain_pct": pooled(range(len(cells))), "spearman_gain_vs_share": rho, "p_one_sided_perm": p1,
       "thirds_by_share": [{"mean_share": float(s[t].mean()), "pooled_gain_pct": pooled(t), "n": int(len(t))} for t in thirds],
       "composite_only_descriptive": {"episodes": len(comp), "spearman": spearman(g[comp], s[comp]), "pooled_gain_pct": pooled(comp)},
       "per_episode": [{"architecture": a, "weather": w, "group": grp[a], "base_comfort_share": float(s[i]), "gain_pct": float(g[i])} for i, (a, w) in enumerate(cells)]}
if MODE == "zt":
    ref = json.load(open(f"{H}/out/RESULT_HEADROOM_K07.json")); pred = "lower" if res["mean_share"] < ref["mean_share"] else "higher"
    res["predictions"] = {"P1_rho_gt_0_and_p_lt_0.05": bool(rho > 0 and p1 < 0.05),
                          "P2_lowest_third_within_0.5pct": bool(abs(res["thirds_by_share"][0]["pooled_gain_pct"]) <= 0.5),
                          "P3": {"reference_mean_share": ref["mean_share"], "reference_pooled_gain_pct": ref["pooled_gain_pct"], "predicted": pred,
                                 "pass": bool(res["pooled_gain_pct"] < ref["pooled_gain_pct"] if pred == "lower" else res["pooled_gain_pct"] > ref["pooled_gain_pct"])}}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=1))
