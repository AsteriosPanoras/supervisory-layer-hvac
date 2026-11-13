"""p269: the remaining robustness checks of the targeted-term relation, for the appendix table of paper v49_51 (TEST_BACKLOG B47 items 1-2;
post hoc, frozen blind runs, nothing run).  p267 checked the comfort share for K07 and ZT Best; p268 added Penguin with its targeted terms
(energy + carbon) but without the absolute and leave-one-architecture-out checks.  For each case and its targeted terms T:
  rho(gain, h) with h = T share of the comparator's loss; rho without ratios rho(dL, T_base); leave-one-architecture-out range of
  rho(gain, h); rho(capture_T, h) with capture_T = removed part of T / T.
Consistency gate: for K07 and ZT (T = comfort) every value must equal p267's; rho(gain, h) and the pooled numbers must equal p268's.
Loading and filters as p267/p268.  usage: python -I p269_targeted_robustness.py <out_dir>"""
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
P7 = json.load(open(f"{H}/out/p267/P267_RATIO_DECOMP.json"))["cases"]; P8 = json.load(open(f"{H}/out/p268/P268_TARGETED_HEADROOM.json"))["cases"]
R = {"note": "post hoc, descriptive; frozen blind runs; completes p267/p268 for the appendix table of v49_51", "cases": {}}
for name, c in CASES.items():
    L, C, cells, grp = load(c); b, l, T = c["base"], c["layer"], c["target"]
    Lb = np.array([L[(a, w, b)] for a, w in cells]); Ll = np.array([L[(a, w, l)] for a, w in cells])
    tb = np.array([sum(C[(a, w, b)][nm] for nm in T) for a, w in cells]); tl = np.array([sum(C[(a, w, l)][nm] for nm in T) for a, w in cells])
    gain = 100 * (1 - Ll / Lb); h = tb / Lb; dL = Lb - Ll; cap = (tb - tl) / tb; archs = sorted({a for a, w in cells})
    loo = [spearman(gain[np.array([x != a for x, w in cells])], h[np.array([x != a for x, w in cells])]) for a in archs]
    R["cases"][name] = {"targeted_terms": list(T), "rho_gain_vs_share": spearman(gain, h), "rho_without_ratios_dL_vs_targeted_loss": spearman(dL, tb),
                        "loao_rho_min": float(min(loo)), "loao_rho_max": float(max(loo)), "rho_capture_vs_share": spearman(cap, h)}
# consistency gate
for name in ("K07", "ZT"):
    r, p7 = R["cases"][name], P7[name]
    assert abs(r["rho_gain_vs_share"] - p7["rho_episode_gain_vs_comfort_share"]) < 1e-12
    assert abs(r["rho_without_ratios_dL_vs_targeted_loss"] - p7["1b_absolute"]["rho_dL_vs_comfort_loss"]) < 1e-12
    assert abs(r["loao_rho_min"] - p7["1c_leave_one_architecture_out_rho"]["min"]) < 1e-12 and abs(r["loao_rho_max"] - p7["1c_leave_one_architecture_out_rho"]["max"]) < 1e-12
    assert abs(r["rho_capture_vs_share"] - p7["1e_capture_and_cost"]["rho_capture_vs_share"]) < 1e-12
for name in R["cases"]:
    assert abs(R["cases"][name]["rho_gain_vs_share"] - P8[name]["rho_gain_vs_targeted_share"]) < 1e-12
json.dump(R, open(f"{OUT}/P269_TARGETED_ROBUSTNESS.json", "w"), indent=1)
for name, r in R["cases"].items():
    print(f"{name:8s} {'+'.join(r['targeted_terms']):14s} rho {r['rho_gain_vs_share']:+.2f} | without ratios {r['rho_without_ratios_dL_vs_targeted_loss']:+.2f} | "
          f"LOAO {r['loao_rho_min']:+.2f} .. {r['loao_rho_max']:+.2f} | rho(capture, share) {r['rho_capture_vs_share']:+.2f}")
print("consistency with p267/p268: OK")
