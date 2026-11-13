"""p297: analyses for the 12 Nov review (part 6): reliability of the headroom account and of the generalization claims, and the
sensitivity of the 2021 gate estimate to its estimand.  POST HOC and descriptive: frozen, already-read runs only; nothing is run and no
registered verdict changes.  usage: python -I p297_review6_analyses.py <profile_study_2021_episodes.csv> <out_dir>
(A) Headroom account on four controllers.  The identity, per episode e and pooled over a set S of episodes (L = sum_c w_c l_c, comparator
    b, layer l, targeted terms T, other terms O):
        g_S = 100 (1 - sum_S L_l / sum_S L_b) = B_S - C_S,   B_S = 100 sum_S (T_b - T_l) / sum_S L_b,   C_S = 100 sum_S (O_l - O_b) / sum_S L_b,
        B_S = kappa_S * h_S  with  h_S = 100 sum_S T_b / sum_S L_b (targeted share)  and  kappa_S = sum_S (T_b - T_l) / sum_S T_b <= 1,
    so B_S <= h_S, and g_S <= h_S whenever C_S >= 0.  The script checks the identity numerically.  The cases K07-G30, ZT Best and Penguin
    reproduce p268/p269 exactly (consistency gate); the Residual PPO case (AX -> AXS, comfort) is new here, with the same checks:
    rho(gain, share) over the 36 episodes, the shared-denominator null (loss reductions permuted, 10,000, seed 0), rho over the 12
    architectures, without ratios, leave-one-architecture-out range, rho(capture, share).  New for all four: a 95 % architecture-cluster
    bootstrap interval for rho over the 36 episodes (3,000, seed 1; average ranks).
(B) What to expect on a new architecture.  Per architecture, the pooled gain over its three episodes.  On the 10 composite architectures:
    gained (gain > 0, the rule of the paper's Table 3), unchanged (exactly 0), lost (< 0); the exact (Clopper-Pearson) 95 % interval of the share that gained and of the
    share that lost more than 1 %; the distribution-free prediction interval [min, max], which holds a new architecture from the same
    generator with probability (n - 1)/(n + 1) = 9/11.  Pooled over the three comfort controllers (30 pairs).
(C) 2021 gate (MG - TRIG2): the registered loss-weighted estimand against the per-episode mean (station-cluster bootstrap, 3,000, seed 1),
    and the effective number of episodes of the loss weights, n_eff = (sum L_B)^2 / sum L_B^2 (Kish), for the main sample, the extension
    and both.
(D) Do the same architectures gain on every controller?  Spearman rho over the 12 architectures between the per-architecture gains of
    the controllers, and between their per-architecture comfort shares (K07-G30, ZT Best, Residual PPO).
(E) Review part 7 (counting rule aligned with Table 3 on 12 Nov after review part 8: gain > 0 counts as gained; the first version used a
    0.01 % band and is kept as p297_review6_analyses_v1_threshold001.py): the 30 controller-architecture pairs share 10 architectures, so a binomial interval over the 30 is not valid; at the
    level of the architecture (the independent unit): on how many of the 10 composite architectures the layer gained under all three
    comfort controllers, and on how many it lost under none, each with its exact (Clopper-Pearson) 95 % interval, assuming the
    architectures are exchangeable draws from the generator."""
import sys, os, csv, glob, json, collections, numpy as np
from scipy.stats import rankdata, beta
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); CSV, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
CASES = {"K07": dict(man=[f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], res=f"{N}/results_k07x3", base="G30", layer="SUP", target=("comfort",)),
         "ZT": dict(man=[f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], res=f"{N}/results_ztsup_fresh", base="ZB", layer="ZS", target=("comfort",)),
         "Penguin": dict(man=[f"{N}/pgf/PGF_RUN_MANIFEST.csv"], res=f"{N}/results_pgf", base="V0", layer="PC", target=("energy", "carbon")),
         "RPPO": dict(man=[f"{N}/axsup/AXSUP_BLIND_MANIFEST.csv"], res=f"{N}/results_axsup_blind", base="AX", layer="AXS", target=("comfort",))}
def load(c):                                    # p268/p269 loading and filters, unchanged
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
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])        # as p268/p269 (no ties in the original data)
def spearman_avg(x, y): return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])  # average ranks, for resamples with repeated points
def cp(k, n):
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)); return [lo, hi]
R = {"note": "POST HOC, descriptive; frozen blind runs and the registered 2021 per-episode table; nothing run", "A_headroom": {}, "B_new_architecture": {}}
PA = {}
for name, c in CASES.items():
    L, C, cells, grp = load(c); b, l, T = c["base"], c["layer"], c["target"]; O = [nm for nm in W if nm not in T]
    Lb = np.array([L[(a, w, b)] for a, w in cells]); Ll = np.array([L[(a, w, l)] for a, w in cells])
    tb = np.array([sum(C[(a, w, b)][nm] for nm in T) for a, w in cells]); tl = np.array([sum(C[(a, w, l)][nm] for nm in T) for a, w in cells])
    ob = np.array([sum(C[(a, w, b)][nm] for nm in O) for a, w in cells]); ol = np.array([sum(C[(a, w, l)][nm] for nm in O) for a, w in cells])
    # the components must add up to the loss (otherwise the identity is not exact)
    assert np.allclose(tb + ob, Lb, rtol=1e-4) and np.allclose(tl + ol, Ll, rtol=1e-4), (name, "components do not add up to the loss")
    gain = 100 * (1 - Ll / Lb); h = tb / Lb; dL = Lb - Ll; cap = (tb - tl) / tb; comp = np.array([grp[a] == "composite" for a, w in cells])
    archs = sorted({a for a, w in cells}); am = {a: np.array([x == a for x, w in cells]) for a in archs}
    rng = np.random.default_rng(0); null = np.array([spearman(100 * rng.permutation(dL) / Lb, h) for _ in range(10000)]); obs = spearman(gain, h)
    ga = [100 * (1 - Ll[am[a]].sum() / Lb[am[a]].sum()) for a in archs]; ha = [tb[am[a]].sum() / Lb[am[a]].sum() for a in archs]
    loo = [spearman(gain[~am[a]], h[~am[a]]) for a in archs]
    rb = np.random.default_rng(1); boot = []
    for _ in range(3000):
        pick = rb.choice(len(archs), len(archs)); idx = np.concatenate([np.where(am[archs[i]])[0] for i in pick]); boot.append(spearman_avg(gain[idx], h[idx]))
    def pooled(m):
        tot = Lb[m].sum(); hS = 100 * tb[m].sum() / tot; B = 100 * (tb[m].sum() - tl[m].sum()) / tot; Cc = 100 * (ol[m].sum() - ob[m].sum()) / tot
        g = 100 * (1 - Ll[m].sum() / Lb[m].sum()); assert abs(g - (B - Cc)) < 1e-6 * max(1, abs(g)), (name, "identity g = B - C fails")
        kap = B / hS; assert kap <= 1 + 1e-12 and B <= hS + 1e-9
        return {"share_h_pct": hS, "benefit_B_pct": B, "cost_C_pct": Cc, "gain_pct": g, "capture_kappa": kap}
    R["A_headroom"][name] = {"targeted_terms": list(T), "rho_36": obs, "rho_36_cluster_boot_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                             "null_shared_denominator": {"mean": float(null.mean()), "2.5": float(np.percentile(null, 2.5)), "97.5": float(np.percentile(null, 97.5))},
                             "rho_12_architectures": spearman(ga, ha), "rho_without_ratios": spearman(dL, tb), "loao_rho": [float(min(loo)), float(max(loo))],
                             "rho_capture_vs_share": spearman(cap, h), "pooled_composite": pooled(comp), "pooled_all_36": pooled(np.ones(len(cells), bool))}
    PA[name] = {a: (g_, grp[a]) for a, g_ in zip(archs, ga)}
# consistency gate with the published p268/p269 numbers
P8 = json.load(open(f"{H}/out/p268/P268_TARGETED_HEADROOM.json"))["cases"]; P9 = json.load(open(f"{H}/out/p269/P269_TARGETED_ROBUSTNESS.json"))["cases"]
for name in ("K07", "ZT", "Penguin"):
    a, p8, p9 = R["A_headroom"][name], P8[name], P9[name]
    assert abs(a["rho_36"] - p8["rho_gain_vs_targeted_share"]) < 1e-12 and abs(a["rho_12_architectures"] - p8["rho_per_architecture_12"]) < 1e-12
    assert abs(a["null_shared_denominator"]["mean"] - p8["null_shared_denominator"]["mean"]) < 1e-12
    assert abs(a["pooled_composite"]["gain_pct"] - p8["pooled_composite"]["net_gain_pct"]) < 1e-6 and abs(a["pooled_composite"]["capture_kappa"] - p8["pooled_composite"]["capture_of_headroom"]) < 1e-9   # p268's net gain is B - C; here it is the loss ratio (equal to 1e-6: the identity)
    assert abs(a["rho_without_ratios"] - p9["rho_without_ratios_dL_vs_targeted_loss"]) < 1e-12 and abs(a["loao_rho"][0] - p9["loao_rho_min"]) < 1e-12 and abs(a["rho_capture_vs_share"] - p9["rho_capture_vs_share"]) < 1e-12
assert round(R["A_headroom"]["RPPO"]["pooled_composite"]["gain_pct"], 2) == 3.37, R["A_headroom"]["RPPO"]["pooled_composite"]["gain_pct"]
assert abs(R["A_headroom"]["RPPO"]["rho_36"] - 0.75) < 0.006, R["A_headroom"]["RPPO"]["rho_36"]
assert round(PA["K07"]["building_52205"][0], 2) == 6.69 and round(PA["RPPO"]["building_52209"][0], 2) == -1.35 and round(PA["RPPO"]["building_52211"][0], 2) == -1.15
# (B) a new architecture from the same generator
pool = collections.Counter()
for name, pa in PA.items():
    g = np.array([v for v, gr in pa.values() if gr == "composite"]); n = len(g); assert n == 10
    k_gain, k_same, k_lose, k_lose1 = int((g > 0).sum()), int((g == 0).sum()), int((g < 0).sum()), int((g < -1.0).sum())   # the paper's rule (Table 3: better = gain > 0)
    R["B_new_architecture"][name] = {"n_composite": n, "gained": k_gain, "unchanged": k_same, "lost": k_lose, "lost_more_than_1pct": k_lose1,
                                     "share_gained_cp95": cp(k_gain, n), "share_lost_gt1_cp95": cp(k_lose1, n),
                                     "prediction_interval_min_max": [float(g.min()), float(g.max())], "pi_coverage": (n - 1) / (n + 1),
                                     "median": float(np.median(g)), "per_architecture": {a: round(v, 3) for a, (v, gr) in sorted(pa.items())}}
    if name in ("K07", "ZT", "RPPO"):
        pool["n"] += n; pool["gained"] += k_gain; pool["lost"] += k_lose; pool["lost1"] += k_lose1
R["B_new_architecture"]["comfort_layer_three_controllers"] = {"n": pool["n"], "gained": pool["gained"], "lost": pool["lost"], "lost_more_than_1pct": pool["lost1"],
                                                               "share_gained_cp95": cp(pool["gained"], pool["n"]), "share_lost_gt1_cp95": cp(pool["lost1"], pool["n"])}
# (C) 2021 gate: estimand sensitivity
rows = list(csv.DictReader(open(CSV))); LL = collections.defaultdict(dict); st = {}
for r in rows: k = (r["cohort"], r["building"], r["weather"]); LL[k][r["arm"]] = float(r["loss"]); st[k] = r["station"]
def block(ks, seed=1, reps=3000):
    B_ = np.array([LL[k]["B"] for k in ks]); T_ = np.array([LL[k]["TRIG2"] for k in ks]); M_ = np.array([LL[k]["MG"] for k in ks]); s_ = np.array([st[k] for k in ks])
    lw = lambda i: 100 * (T_[i].sum() - M_[i].sum()) / B_[i].sum(); pe = lambda i: float((100 * (T_[i] - M_[i]) / B_[i]).mean())
    us = sorted(set(s_)); gi = {u: np.where(s_ == u)[0] for u in us}; rng = np.random.default_rng(seed); bl, bp = [], []
    for _ in range(reps):
        p = rng.choice(len(us), len(us)); i = np.concatenate([gi[us[j]] for j in p]); bl.append(lw(i)); bp.append(pe(i))
    a = np.arange(len(ks)); w = B_ / B_.sum()
    return {"n": len(ks), "stations": len(us), "loss_weighted_registered": round(float(lw(a)), 3), "lw_ci": [round(float(np.percentile(bl, q)), 3) for q in (2.5, 97.5)],
            "per_episode_mean": round(pe(a), 3), "pe_ci": [round(float(np.percentile(bp, q)), 3) for q in (2.5, 97.5)],
            "n_eff_kish": round(float(1 / (w ** 2).sum()), 1), "largest_weight_pct": round(float(100 * w.max()), 1)}
main = sorted(k for k in LL if k[0] == "H2-main"); ext = sorted(k for k in LL if k[0] == "H2-ext")
R["C_gate_2021"] = {"main": block(main), "extension": block(ext), "pooled": block(main + ext)}
c3 = R["C_gate_2021"]; assert c3["main"]["loss_weighted_registered"] == 0.504 and c3["main"]["lw_ci"] == [0.337, 0.704] and c3["extension"]["loss_weighted_registered"] == -0.004 \
    and c3["pooled"]["loss_weighted_registered"] == 0.407 and c3["pooled"]["lw_ci"] == [0.21, 0.633], c3
# (D) cross-controller consistency over the 12 architectures
SH = {}
for name in ("K07", "ZT", "RPPO"):
    L, C, cells, grp = load(CASES[name]); b = CASES[name]["base"]; archs = sorted({a for a, w in cells})
    SH[name] = {a: sum(C[(a, w, b)]["comfort"] for x, w in cells if x == a) / sum(L[(a, w, b)] for x, w in cells if x == a) for a in archs}
archs = sorted(PA["K07"]); D = {}
for i, x in enumerate(("K07", "ZT", "RPPO", "Penguin")):
    for y in ("K07", "ZT", "RPPO", "Penguin")[i + 1:]:
        D[f"gain_{x}_vs_{y}"] = spearman([PA[x][a][0] for a in archs], [PA[y][a][0] for a in archs])
for i, x in enumerate(("K07", "ZT", "RPPO")):
    for y in ("K07", "ZT", "RPPO")[i + 1:]:
        D[f"comfort_share_{x}_vs_{y}"] = spearman([SH[x][a] for a in archs], [SH[y][a] for a in archs])
R["D_cross_controller_12_architectures"] = D
comp_arch = sorted(a for a, (v, gr) in PA["K07"].items() if gr == "composite"); assert len(comp_arch) == 10
st3 = {a: [PA[c][a][0] for c in ("K07", "ZT", "RPPO")] for a in comp_arch}
all_gain = [a for a in comp_arch if all(v > 0 for v in st3[a])]; none_lost = [a for a in comp_arch if all(v >= 0 for v in st3[a])]
R["E_architecture_level"] = {"n": 10, "gained_under_all_three": len(all_gain), "cp95_all_three": cp(len(all_gain), 10), "lost_under_none": len(none_lost),
                             "cp95_none_lost": cp(len(none_lost), 10), "architectures_with_a_loss": sorted(set(comp_arch) - set(none_lost)),
                             "per_architecture_three_gains": {a: [round(v, 3) for v in st3[a]] for a in comp_arch}}
json.dump(R, open(f"{OUT}/P297_REVIEW6.json", "w"), indent=1)
print("D", json.dumps({k: round(v, 2) for k, v in D.items()}))
print("E", json.dumps({k: v for k, v in R["E_architecture_level"].items() if k != "per_architecture_three_gains"}))
for name, a in R["A_headroom"].items():
    p = a["pooled_composite"]
    print(f"{name:8s} share {p['share_h_pct']:5.1f} benefit {p['benefit_B_pct']:5.2f} cost {p['cost_C_pct']:+5.2f} gain {p['gain_pct']:5.2f} kappa {100 * p['capture_kappa']:5.1f}% | rho36 {a['rho_36']:+.2f} "
          f"[{a['rho_36_cluster_boot_95'][0]:+.2f}, {a['rho_36_cluster_boot_95'][1]:+.2f}] null {a['null_shared_denominator']['mean']:+.2f} [{a['null_shared_denominator']['2.5']:+.2f}, {a['null_shared_denominator']['97.5']:+.2f}] "
          f"rho12 {a['rho_12_architectures']:+.2f} noratio {a['rho_without_ratios']:+.2f} loao {a['loao_rho'][0]:+.2f}..{a['loao_rho'][1]:+.2f} rho(cap,share) {a['rho_capture_vs_share']:+.2f}")
for name, b in R["B_new_architecture"].items():
    print(name, {k: v for k, v in b.items() if k != "per_architecture"})
print(json.dumps(R["C_gate_2021"]))
