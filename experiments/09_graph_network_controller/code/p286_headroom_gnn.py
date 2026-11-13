"""p286: registered test of the headroom account on the graph-network controller (PREREG_GNNSUP_v1, predictions P1-P4), written
2026-11-12 before any GNN blind run; p278 with GN as the base and GNS as the layer, unchanged otherwise (P1-P3 as p258 mode zt,
P4 as p274; K07 reference out/RESULT_HEADROOM_K07.json).  Nothing is computed unless RESULT_GNNSUP_FIDELITY.json says ALL_PASS and
36 complete episodes exist.  usage: python -I p286_headroom_gnn.py <out.json>"""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; P = f"{N}/paperpack"
if not json.load(open(f"{H}/RESULT_GNNSUP_FIDELITY.json")).get("ALL_PASS"): sys.exit("fidelity not passed: nothing read")
man = list(csv.DictReader(open(f"{H}/GNNSUP_BLIND_MANIFEST.csv"))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
W = {"normalized_productivity_regret": 0.6, "normalized_energy_cost": 0.2, "normalized_carbon_emission": 0.2}
L, C, RW, CW = {}, {}, {}, {}
for pat in (f"{N}/results_gnnsup_blind/**/*.summary.json", f"{H}/laptop/work_gnb*/out/**/*.summary.json"):
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in ("GN", "GNS") or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); rw = np.abs(z["rewards"].astype(float)); L[k] = float(rw.mean()); RW[k] = rw
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {c: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for c, wt in W.items()}
        CW[k] = np.abs(0.6 * z["comps"][:, ck.index("normalized_productivity_regret")].astype(float))
cells = sorted({(a, w) for a, w, p in L if (a, w, "GN") in L and (a, w, "GNS") in L})
if len(cells) < 36: json.dump({"status": f"INCOMPLETE: {len(cells)} episodes"}, open(OUT, "w")); sys.exit(f"incomplete: {len(cells)} episodes")
s = np.array([C[(a, w, "GN")]["normalized_productivity_regret"] / sum(C[(a, w, "GN")].values()) for a, w in cells])
g = np.array([100 * (1 - L[(a, w, "GNS")] / L[(a, w, "GN")]) for a, w in cells])
rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
rho = spearman(g, s); rng = np.random.default_rng(0); perm = np.array([spearman(rng.permutation(g), s) for _ in range(10000)])
p1 = float((1 + (perm >= rho).sum()) / (1 + len(perm)))
def pooled(idx): idx = list(idx); return float(100 * (1 - sum(L[(*cells[i], "GNS")] for i in idx) / sum(L[(*cells[i], "GN")] for i in idx)))
order = np.argsort(s, kind="stable"); n3 = len(cells) // 3; thirds = [order[:n3], order[n3:2 * n3], order[2 * n3:]]
WK = 2016; s1 = np.array([CW[(a, w, "GN")][:WK].sum() / RW[(a, w, "GN")][:WK].sum() for a, w in cells])
g830 = np.array([100 * (1 - RW[(a, w, "GNS")][WK:].sum() / RW[(a, w, "GN")][WK:].sum()) for a, w in cells]); rho4 = spearman(g830, s1)
ref = json.load(open(f"{P}/out/RESULT_HEADROOM_K07.json")); res = {"base": "GN", "layer": "GNS", "episodes": len(cells), "mean_share": float(s.mean()), "pooled_gain_pct": pooled(range(len(cells))),
       "spearman_gain_vs_share": rho, "p_one_sided_perm": p1, "thirds_by_share": [{"mean_share": float(s[t].mean()), "pooled_gain_pct": pooled(t), "n": int(len(t))} for t in thirds],
       "first_week_share_mean": float(s1.mean()), "spearman_days8_30_gain_vs_first_week_share": rho4,
       "per_episode": [{"architecture": a, "weather": w, "group": grp[a], "base_comfort_share": float(s[i]), "gain_pct": float(g[i]), "first_week_share": float(s1[i]), "gain_days8_30_pct": float(g830[i])} for i, (a, w) in enumerate(cells)]}
pred = "lower" if res["mean_share"] < ref["mean_share"] else "higher"
res["predictions"] = {"P1_rho_gt_0_and_p_lt_0.05": bool(rho > 0 and p1 < 0.05), "P2_lowest_third_within_0.5pct": bool(abs(res["thirds_by_share"][0]["pooled_gain_pct"]) <= 0.5),
                      "P3": {"reference_mean_share": ref["mean_share"], "reference_pooled_gain_pct": ref["pooled_gain_pct"], "predicted": pred,
                             "pass": bool(res["pooled_gain_pct"] < ref["pooled_gain_pct"] if pred == "lower" else res["pooled_gain_pct"] > ref["pooled_gain_pct"])},
                      "P4_first_week_share_ranks_days8_30_gain": bool(rho4 > 0)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=1))
