"""p268: the gain against the share of the term each mechanism targets (TEST_BACKLOG B47 item 2, continuing p267; post hoc, frozen runs,
nothing run).  p267 found that the learning layer's gain follows the comfort share on K07 and ZT Best but runs against it on Penguin,
whose learned precooling targets energy and carbon, not comfort.  Here the share is that of the TARGETED terms:
  K07 (G30 -> SUP) and ZT Best (ZB -> ZS): comfort;  Penguin (V0 -> PC): energy + carbon.
Per episode: targeted share h = sum of the targeted weighted components / L of the comparator; capture = removed part of the targeted
terms / their size; cost = change of the other terms in % of L.  Pooled over the composite architectures: headroom H, benefit B, cost,
net gain = B - cost, capture = B / H.  Shared-denominator null as in p267 (dL permuted, 10,000, seed 0).  Loading and filters as p267.
Official training scenes: comfort headroom and the gain it would allow at the capture rates seen on the blind set (before costs).
usage: python -I p268_targeted_term_headroom.py <out_dir>"""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
CASES = {"K07": dict(man=[f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], res=f"{N}/results_k07x3", base="G30", layer="SUP", target=("comfort",)),
         "ZT": dict(man=[f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], res=f"{N}/results_ztsup_fresh", base="ZB", layer="ZS", target=("comfort",)),
         "Penguin": dict(man=[f"{N}/pgf/PGF_RUN_MANIFEST.csv"], res=f"{N}/results_pgf", base="V0", layer="PC", target=("energy", "carbon"))}
def load(c):
    man = [r for f in c["man"] for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
    grp = {r["architecture"]: r["group"] for r in man}; L, C = {}, {}
    for f in glob.glob(f"{c['res']}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in (c["base"], c["layer"]) or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean())
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(key)].astype(float)).mean()) for nm, (key, wt) in W.items()}
    cells = sorted({(a, w) for a, w, p in L if (a, w, c["base"]) in L and (a, w, c["layer"]) in L}); assert len(cells) == 36, len(cells)
    return L, C, cells, grp
rank = lambda x: np.argsort(np.argsort(np.asarray(x), kind="stable"), kind="stable").astype(float)
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
R = {"note": "post hoc, descriptive; frozen blind runs; TEST_BACKLOG B47 item 2; continues p267", "cases": {}}
for name, c in CASES.items():
    L, C, cells, grp = load(c); b, l, T = c["base"], c["layer"], c["target"]; O = [nm for nm in W if nm not in T]
    Lb = np.array([L[(a, w, b)] for a, w in cells]); Ll = np.array([L[(a, w, l)] for a, w in cells])
    tb = np.array([sum(C[(a, w, b)][nm] for nm in T) for a, w in cells]); tl = np.array([sum(C[(a, w, l)][nm] for nm in T) for a, w in cells])
    ob = np.array([sum(C[(a, w, b)][nm] for nm in O) for a, w in cells]); ol = np.array([sum(C[(a, w, l)][nm] for nm in O) for a, w in cells])
    gain = 100 * (1 - Ll / Lb); h = tb / Lb; dL = Lb - Ll; comp = np.array([grp[a] == "composite" for a, w in cells])
    rng = np.random.default_rng(0); null = np.array([spearman(100 * rng.permutation(dL) / Lb, h) for _ in range(10000)]); obs = spearman(gain, h)
    archs = sorted({a for a, w in cells}); ga, ha = [], []
    for a in archs:
        m = np.array([x == a for x, w in cells]); ga.append(100 * (1 - Ll[m].sum() / Lb[m].sum())); ha.append(tb[m].sum() / Lb[m].sum())
    def pooled(m):
        tot = Lb[m].sum(); Hh = float(tb[m].sum() / tot); B = float(100 * (tb[m].sum() - tl[m].sum()) / tot); cost = float(100 * (ol[m].sum() - ob[m].sum()) / tot)
        return {"headroom_targeted_share": Hh, "benefit_pct": B, "other_terms_cost_pct": cost, "net_gain_pct": B - cost, "capture_of_headroom": B / (100 * Hh)}
    R["cases"][name] = {"targeted_terms": list(T), "rho_gain_vs_targeted_share": obs,
                        "null_shared_denominator": {"mean": float(null.mean()), "2.5": float(np.percentile(null, 2.5)), "97.5": float(np.percentile(null, 97.5)),
                                                    "p_null_ge_observed": float((1 + (null >= obs).sum()) / (1 + len(null)))},
                        "rho_per_architecture_12": spearman(ga, ha), "capture_median_episode": float(np.median((tb - tl) / tb)),
                        "pooled_composite": pooled(comp), "pooled_all_36": pooled(np.ones(len(cells), bool))}
k = R["cases"]["K07"]
assert round(k["pooled_composite"]["net_gain_pct"], 2) == 2.82 and round(R["cases"]["ZT"]["pooled_composite"]["net_gain_pct"], 2) == 5.71 \
    and round(R["cases"]["Penguin"]["pooled_composite"]["net_gain_pct"], 2) == 1.88, "pooled gains not reproduced"
wf = json.load(open(f"{H}/out/WHEN_FIGURE_DATA.json"))["official_scene_comfort_share"]; sh = sorted(wf.values())
capK = [k["capture_median_episode"], k["pooled_composite"]["capture_of_headroom"]]
R["official_scenes"] = {"comfort_share_min": sh[0], "comfort_share_max": sh[-1], "n": len(sh),
                        "gain_allowed_before_costs_pct_at_K07_capture": {f"capture_{c:.3f}": [100 * c * sh[0], 100 * c * sh[-1]] for c in capK}}
json.dump(R, open(f"{OUT}/P268_TARGETED_HEADROOM.json", "w"), indent=1)
for nm, o in R["cases"].items():
    p = o["pooled_composite"]; n_ = o["null_shared_denominator"]
    print(f"{nm:8s} target {'+'.join(o['targeted_terms']):14s} rho(gain, target share) {o['rho_gain_vs_targeted_share']:+.2f} (null {n_['mean']:+.2f} [{n_['2.5']:+.2f}, {n_['97.5']:+.2f}], p {n_['p_null_ge_observed']:.4f}); per-arch {o['rho_per_architecture_12']:+.2f}; "
          f"composite: headroom {100 * p['headroom_targeted_share']:.1f} %, benefit {p['benefit_pct']:.2f}, cost {p['other_terms_cost_pct']:+.2f}, net {p['net_gain_pct']:.2f} %, capture {100 * p['capture_of_headroom']:.1f} % (episode median {100 * o['capture_median_episode']:.1f} %)")
print("official scenes:", json.dumps(R["official_scenes"]))
