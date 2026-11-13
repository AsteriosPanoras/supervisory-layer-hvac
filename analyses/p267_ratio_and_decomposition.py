"""p267: post hoc checks of the comfort-share relation and a decomposition of the gains.  Frozen, already-read blind runs only; nothing is
run, no verdict changes, every number is reported whatever its direction.  usage: python -I p267_ratio_and_decomposition.py <out_dir>
Loading, filters and per-episode quantities are those of p258 (K07: G30 -> SUP on results_k07x3; ZT Best: ZB -> ZS on results_ztsup_fresh)
and p254 (Penguin: V0 -> PC on results_pgf): status OK, 8640 steps, no illegal action, duplicates abort; L = mean |reward|; weighted
components = mean |w * component| with w = 0.6 comfort, 0.2 energy, 0.2 carbon.
 0. Reproduction gate: Spearman rho(gain, comfort share) 0.80 (K07) and 0.93 (ZT) as printed; pooled gains 2.82 / 5.71 / 1.88 % (composite)
    and 2.69 / 5.97 % (all 36); K07 component changes of Table 32 (comfort -3.87, energy +0.43, carbon +0.62).  Abort if any differs.
 1. Is rho an artefact of the shared denominator (gain = 100 dL / L, share = Cq / L)?
    a) rho over the 12 architectures (episodes pooled per architecture; also the 10 composite);
    b) without any ratio: rho(dL, Cq) and rho(dCq, Cq) in absolute loss units;
    c) leave-one-architecture-out range of the episode-level rho;
    d) shared-denominator null: dL permuted across episodes (10,000 times, seed 0), rho(perm(dL) / L, Cq / L);
    e) capture rate r = dCq / Cq (share of the comfort loss removed) and the cost of the other terms, per episode.
 2. Decomposition, pooled over the composite architectures (and all 36): change of each weighted component in % of the comparator's
    loss, the comparator's shares, the headroom of the corrected term and the share of it that was used; official scenes: comfort share."""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
CASES = {"K07": dict(man=[f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], res=f"{N}/results_k07x3", base="G30", layer="SUP"),
         "ZT": dict(man=[f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], res=f"{N}/results_ztsup_fresh", base="ZB", layer="ZS"),
         "Penguin": dict(man=[f"{N}/pgf/PGF_RUN_MANIFEST.csv"], res=f"{N}/results_pgf", base="V0", layer="PC")}
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
    cells = sorted({(a, w) for a, w, p in L if (a, w, c["base"]) in L and (a, w, c["layer"]) in L})
    assert len(cells) == 36, (c["res"], len(cells))
    return L, C, cells, grp
rank = lambda x: np.argsort(np.argsort(np.asarray(x), kind="stable"), kind="stable").astype(float)
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
R = {"note": "post hoc, descriptive; frozen blind runs read before; TEST_BACKLOG B47 items 1-2; nothing new was run", "cases": {}}
for name, c in CASES.items():
    L, C, cells, grp = load(c); b, l = c["base"], c["layer"]
    dev = max(abs(sum(C[(a, w, p)].values()) / L[(a, w, p)] - 1) for a, w in cells for p in (b, l)); assert dev < 1e-6, (name, dev)
    Lb = np.array([L[(a, w, b)] for a, w in cells]); Ll = np.array([L[(a, w, l)] for a, w in cells])
    Cb = {nm: np.array([C[(a, w, b)][nm] for a, w in cells]) for nm in W}; Cl = {nm: np.array([C[(a, w, l)][nm] for a, w in cells]) for nm in W}
    gain = 100 * (1 - Ll / Lb); share = Cb["comfort"] / Lb; dL = Lb - Ll; dCq = Cb["comfort"] - Cl["comfort"]
    comp = np.array([grp[a] == "composite" for a, w in cells]); archs = sorted({a for a, w in cells}); n = len(cells)
    pooled = lambda m: float(100 * (1 - Ll[m].sum() / Lb[m].sum()))
    out = {"episodes": n, "pooled_gain_composite_pct": pooled(comp), "pooled_gain_all_pct": pooled(np.ones(n, bool)),
           "rho_episode_gain_vs_comfort_share": spearman(gain, share), "mean_comfort_share": float(share.mean())}
    # 1a per architecture
    ga, sa, ca = [], [], []
    for a in archs:
        m = np.array([x == a for x, w in cells]); ga.append(100 * (1 - Ll[m].sum() / Lb[m].sum())); sa.append(Cb["comfort"][m].sum() / Lb[m].sum()); ca.append(grp[a] == "composite")
    ga, sa, ca = np.array(ga), np.array(sa), np.array(ca)
    out["1a_rho_per_architecture"] = {"all_12": spearman(ga, sa), "composite_10": spearman(ga[ca], sa[ca]),
                                     "per_architecture": [{"architecture": a, "gain_pct": float(g_), "comfort_share": float(s_)} for a, g_, s_ in zip(archs, ga, sa)]}
    # 1b absolute
    out["1b_absolute"] = {"rho_dL_vs_comfort_loss": spearman(dL, Cb["comfort"]), "rho_dComfort_vs_comfort_loss": spearman(dCq, Cb["comfort"]),
                          "rho_total_loss_vs_comfort_loss": spearman(Lb, Cb["comfort"])}
    # 1c leave one architecture out
    loo = []
    for a in archs:
        m = np.array([x != a for x, w in cells]); loo.append(spearman(gain[m], share[m]))
    out["1c_leave_one_architecture_out_rho"] = {"min": float(min(loo)), "max": float(max(loo))}
    # 1d shared-denominator null
    rng = np.random.default_rng(0); null = np.array([spearman(100 * rng.permutation(dL) / Lb, share) for _ in range(10000)])
    obs = out["rho_episode_gain_vs_comfort_share"]
    out["1d_shared_denominator_null"] = {"observed_rho": obs, "null_mean": float(null.mean()), "null_2.5": float(np.percentile(null, 2.5)),
                                        "null_97.5": float(np.percentile(null, 97.5)), "p_null_ge_observed": float((1 + (null >= obs).sum()) / (1 + len(null)))}
    # 1e capture rate and cost
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(Cb["comfort"] > 0, dCq / Cb["comfort"], np.nan)
    cost = 100 * ((Cl["energy"] - Cb["energy"]) + (Cl["carbon"] - Cb["carbon"])) / Lb
    out["1e_capture_and_cost"] = {"capture_median": float(np.nanmedian(cap)), "capture_q25": float(np.nanpercentile(cap, 25)), "capture_q75": float(np.nanpercentile(cap, 75)),
                                  "rho_capture_vs_share": spearman(np.nan_to_num(cap), share), "other_terms_cost_pct_median": float(np.median(cost)),
                                  "rho_cost_vs_share": spearman(cost, share)}
    # 2 decomposition (pooled)
    def decomp(m):
        tot = Lb[m].sum()
        d = {nm: float(100 * (Cl[nm][m].sum() - Cb[nm][m].sum()) / tot) for nm in W}
        sh = {nm: float(Cb[nm][m].sum() / tot) for nm in W}
        return {"change_pct_of_comparator_loss": d, "net_change_pct": float(sum(d.values())), "comparator_shares": sh}
    out["2_decomposition"] = {"composite": decomp(comp), "all_36": decomp(np.ones(n, bool))}
    R["cases"][name] = out
# 0. reproduction gate (numbers as printed in the paper)
K, Z, P = R["cases"]["K07"], R["cases"]["ZT"], R["cases"]["Penguin"]
checks = {"rho_K07_0.80": round(K["rho_episode_gain_vs_comfort_share"], 2) == 0.80, "rho_ZT_0.93": round(Z["rho_episode_gain_vs_comfort_share"], 2) == 0.93,
          "gain_K07_comp_2.82": round(K["pooled_gain_composite_pct"], 2) == 2.82, "gain_ZT_comp_5.71": round(Z["pooled_gain_composite_pct"], 2) == 5.71,
          "gain_Penguin_comp_1.88": round(P["pooled_gain_composite_pct"], 2) == 1.88, "gain_K07_all_2.69": round(K["pooled_gain_all_pct"], 2) == 2.69,
          "gain_ZT_all_5.97": round(Z["pooled_gain_all_pct"], 2) == 5.97}
kd = K["2_decomposition"]["composite"]["change_pct_of_comparator_loss"]
checks["table32_K07"] = (round(kd["comfort"], 2), round(kd["energy"], 2), round(kd["carbon"], 2)) == (-3.87, 0.43, 0.62)
R["0_reproduction"] = checks
wf = json.load(open(f"{H}/out/WHEN_FIGURE_DATA.json")); R["official_scenes_comfort_share"] = wf["official_scene_comfort_share"]
json.dump(R, open(f"{OUT}/P267_RATIO_DECOMP.json", "w"), indent=1)
print(json.dumps(checks)); assert all(checks.values()), "reproduction failed: nothing is interpreted"
for nm, o in R["cases"].items():
    print(f"\n== {nm}: pooled gain composite {o['pooled_gain_composite_pct']:.2f} %, all {o['pooled_gain_all_pct']:.2f} %, mean comfort share {o['mean_comfort_share']:.3f}")
    print(f"   rho episodes {o['rho_episode_gain_vs_comfort_share']:.2f} | 1a per-arch 12: {o['1a_rho_per_architecture']['all_12']:.2f}, composite 10: {o['1a_rho_per_architecture']['composite_10']:.2f}")
    print(f"   1b absolute: rho(dL, Cq) {o['1b_absolute']['rho_dL_vs_comfort_loss']:.2f}, rho(dCq, Cq) {o['1b_absolute']['rho_dComfort_vs_comfort_loss']:.2f}, rho(L, Cq) {o['1b_absolute']['rho_total_loss_vs_comfort_loss']:.2f}")
    print(f"   1c LOAO rho {o['1c_leave_one_architecture_out_rho']['min']:.2f} .. {o['1c_leave_one_architecture_out_rho']['max']:.2f}")
    d = o["1d_shared_denominator_null"]; print(f"   1d null: mean {d['null_mean']:.2f} [{d['null_2.5']:.2f}, {d['null_97.5']:.2f}], p(null >= obs) {d['p_null_ge_observed']:.4f}")
    e = o["1e_capture_and_cost"]; print(f"   1e capture median {e['capture_median']:.2f} (IQR {e['capture_q25']:.2f}-{e['capture_q75']:.2f}), rho(capture, share) {e['rho_capture_vs_share']:.2f}; other-terms cost median {e['other_terms_cost_pct_median']:+.2f} %, rho(cost, share) {e['rho_cost_vs_share']:.2f}")
    for k in ("composite", "all_36"):
        x = o["2_decomposition"][k]; print(f"   2 {k}: change {json.dumps({a: round(b, 2) for a, b in x['change_pct_of_comparator_loss'].items()})} net {x['net_change_pct']:+.2f}; shares {json.dumps({a: round(b, 3) for a, b in x['comparator_shares'].items()})}")
print("\nofficial scenes comfort share:", json.dumps(R["official_scenes_comfort_share"])[:300])
