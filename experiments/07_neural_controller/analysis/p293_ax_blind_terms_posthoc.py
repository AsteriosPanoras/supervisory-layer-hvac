"""p293 (post hoc, descriptive; written 2026-11-12 after the AXSUP blind result and the AX leaderboard result): for each blind architecture,
the layer's gain on the Residual PPO controller, the controller's comfort share, the change of each weighted term (% of AX loss; + = the
layer lowered it), the energy + carbon change at night (00-05) and the comfort change in the early morning (05-08); Spearman rank correlation
between the comfort share and the gain over the 12 architectures.  usage: python3 p293_ax_blind_terms_posthoc.py <out.json>"""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
man = list(csv.DictReader(open(f"{H}/AXSUP_BLIND_MANIFEST.csv"))); key = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
P = {"night": range(0, 5), "early": range(5, 8)}
D = {}
for f in glob.glob(f"{N}/results_axsup_blind/**/*.summary.json", recursive=True) + glob.glob(f"{H}/laptop/work_axb*/out/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key.get((s["building"], s["weather"], s["policy_key"]))
    if not r or r["policy"] not in ("AX", "AXS") or s["status"] != "OK" or s["steps_done"] != 8640: continue
    z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; hr = np.array([int(t[11:13]) for t in z["ts"]])
    c = {nm: np.abs(wt * z["comps"][:, ck.index(cc)].astype(float)) for nm, (cc, wt) in W.items()}
    D[(r["architecture"], r["weather"], r["policy"])] = {"c": {k: float(v.sum()) for k, v in c.items()}, "p": {f"{k}|{p}": float(v[np.isin(hr, list(h))].sum()) for k, v in c.items() for p, h in P.items()}, "grp": r["group"]}
res = {"note": "post hoc, descriptive", "architectures": {}}
for a in sorted({a for a, w, p in D}):
    cells = [(aa, w) for aa, w, p in D if aa == a and p == "AX" and (aa, w, "AXS") in D]
    AX = {k: sum(D[(aa, w, "AX")]["c"][k] for aa, w in cells) for k in W}; S = {k: sum(D[(aa, w, "AXS")]["c"][k] for aa, w in cells) for k in W}; T = sum(AX.values())
    pa = lambda pol, kp: sum(D[(aa, w, pol)]["p"][kp] for aa, w in cells)
    res["architectures"][a] = {"group": D[(cells[0][0], cells[0][1], "AX")]["grp"], "windows": len(cells), "gain_pct": 100 * (1 - sum(S.values()) / T), "AX_comfort_share_pct": 100 * AX["comfort"] / T,
        **{f"{k}_change_pct": 100 * (AX[k] - S[k]) / T for k in W},
        "night_energy_carbon_change_pct": 100 * sum(pa("AX", f"{k}|night") - pa("AXS", f"{k}|night") for k in ("energy", "carbon")) / T,
        "early_comfort_change_pct": 100 * (pa("AX", "comfort|early") - pa("AXS", "comfort|early")) / T}
g = np.array([v["gain_pct"] for v in res["architectures"].values()]); s = np.array([v["AX_comfort_share_pct"] for v in res["architectures"].values()])
rk = lambda x: np.argsort(np.argsort(x)); res["spearman_share_gain_12"] = float(np.corrcoef(rk(s), rk(g))[0, 1])
json.dump(res, open(OUT, "w"), indent=1); print("spearman(comfort share, gain) over 12 architectures:", round(res["spearman_share_gain_12"], 3))
