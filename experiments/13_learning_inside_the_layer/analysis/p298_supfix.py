"""p298: K07SUPFIX analysis (PREREG_K07SUPFIX_v1; written and hashed BEFORE any SUPFIX run).  usage: python -I p298_supfix.py <out.json>
Arms on the 36 blind scenes of the K07 test (frozen runs unless stated): B and ROSFIX (fixed start alone; K07X3H), G30 and ROS (learned
start alone; K07X3 stage 1), SUP (learned start + pressure reset; K07X3 stage 2, Kaggle) and the new SUPFIX (fixed start + pressure
reset; laptop).  Loss = mean |reward| over 8640 steps; filters as p268 (status OK, 8640 steps, no illegal actions).
PRIMARY: SUP over SUPFIX, gain pooled over the episodes of the 10 composite architectures, 95 % architecture-cluster bootstrap
(3,000 replicates, seed 1; each drawn architecture enters with all its episodes as often as drawn: the p289/p277b logic).
DECISION: LEARNING ADDS VALUE INSIDE THE LAYER if the interval lies above 0; NOT DETECTED if it contains 0; LEARNING HARMS if it lies
below 0.  Reported with it, no multiplicity claim: the exact Wilcoxon signed-rank and sign tests over the 10 per-architecture gains,
better/worse counts, the OOD pair, the change of the comfort, energy and carbon terms, SUPFIX over G30 and over ROSFIX (the pressure
reset over the fixed start), and the comparison with learning without the reset (ROS over ROSFIX).
CHECKS (must reproduce the published numbers): SUP over G30 = 2.82 %, ROS over ROSFIX = 0.95 %, SUP over ROS = 1.58 %.
REPRODUCIBILITY (descriptive): the two SUP jobs re-run on the laptop against their Kaggle originals (bitwise, max |difference|)."""
import sys, os, csv, glob, json, collections, numpy as np
from scipy.stats import wilcoxon, binomtest
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
SRC = [(f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/results_k07x3"), (f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/results_k07x3"),
       (f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv", f"{N}/results_k07x3h"), (f"{H}/K07SUPFIX_MANIFEST.csv", f"{N}/results_k07supfix")]
L, C, grp = {}, {}, {}
for man, res in SRC:
    rows = list(csv.DictReader(open(man))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in rows}
    for r in rows: grp[r["architecture"]] = r["group"]
    pols = {r["policy"] for r in rows}
    for f in glob.glob(f"{res}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in pols or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean())
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for nm, (c, wt) in W.items()}
n = collections.Counter(k[2] for k in L)
cells = sorted({(a, w) for a, w, p in L if p == "G30"}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
assert len(cells) == 36 and len(CO) == 10 and len(OOD) == 2, (len(cells), len(CO), len(OOD))
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def Ssum(p, archs): return sum(L[(a, w, p)] for aa in archs for a, w in cells if a == aa)      # multiplicity, as p289
def gain(x, y, archs=CO): return 100 * (1 - Ssum(x, archs) / Ssum(y, archs))
def contrast(x, y):
    ga = {a: gain(x, y, [a]) for a in CO + OOD}; reps = [gain(x, y, [CO[i] for i in b]) for b in BOOT]; g = np.array([ga[a] for a in CO]); nz = g[g != 0]
    out = {"gain_pct": gain(x, y), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "better": int((g > 0).sum()), "worse": int((g < 0).sum()),
           "wilcoxon_exact_p": float(wilcoxon(nz, alternative="two-sided", method="exact").pvalue) if len(nz) else None,
           "sign_p": float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else None,
           "ood_gain_pct": gain(x, y, OOD), "per_architecture_gain_pct": {a: round(v, 4) for a, v in ga.items()}}
    tot = sum(L[(a, w, y)] for a, w in cells if a in CO)
    out["term_change_points"] = {nm: 100 * sum(C[(a, w, x)][nm] - C[(a, w, y)][nm] for a, w in cells if a in CO) / tot for nm in W}   # negative = the term fell
    return out
res = {"runs": dict(n), "checks": {}}
for x, y, want in (("SUP", "G30", 2.82), ("ROS", "ROSFIX", 0.95), ("SUP", "ROS", 1.58)):
    g = gain(x, y); res["checks"][f"{x}_over_{y}"] = round(g, 2); assert round(g, 2) == want, (x, y, g)
if n.get("SUPFIX", 0) < 36:
    res["status"] = f"INCOMPLETE: SUPFIX runs {n.get('SUPFIX', 0)} of 36"
else:
    p = contrast("SUP", "SUPFIX"); res["PRIMARY_SUP_over_SUPFIX"] = p
    res["DECISION"] = "LEARNING ADDS VALUE INSIDE THE LAYER" if p["ci95"][0] > 0 else "LEARNING HARMS" if p["ci95"][1] < 0 else "NOT DETECTED"
    res["SUPFIX_over_G30"] = contrast("SUPFIX", "G30"); res["SUPFIX_over_ROSFIX_pressure_reset_on_fixed_start"] = contrast("SUPFIX", "ROSFIX")
    res["ROS_over_ROSFIX_learning_without_reset"] = contrast("ROS", "ROSFIX")
    # the same contrasts in points of B-referenced gain (the additive chain of the paper)
    pts = lambda x: 100 * (1 - Ssum(x, CO) / Ssum("B", CO))
    res["chain_points_of_B"] = {a: round(pts(a), 3) for a in ("G30", "ROSFIX", "ROS", "SUPFIX", "SUP")}
    res["learning_points_with_reset"] = round(pts("SUP") - pts("SUPFIX"), 3); res["learning_points_without_reset"] = round(pts("ROS") - pts("ROSFIX"), 3)
# reproducibility of SUP on the laptop (descriptive)
rep = []
for f in glob.glob(f"{N}/results_k07supfix_repro/**/*.npz", recursive=True):
    g = glob.glob(f"{N}/results_k07x3/**/{os.path.basename(f)}", recursive=True)
    if g:
        a, b = np.load(f, allow_pickle=True)["rewards"], np.load(g[0], allow_pickle=True)["rewards"]
        rep.append({"file": os.path.basename(f), "bitwise": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max()),
                    "rel_loss_diff": float(abs(np.abs(a).mean() - np.abs(b).mean()) / np.abs(b).mean())})
res["repro_SUP_laptop_vs_kaggle"] = rep
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k in ("runs", "checks", "status", "DECISION", "repro_SUP_laptop_vs_kaggle")}))
if "PRIMARY_SUP_over_SUPFIX" in res:
    p = res["PRIMARY_SUP_over_SUPFIX"]; print(f"SUP over SUPFIX {p['gain_pct']:+.2f} % [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] better {p['better']} worse {p['worse']} Wilcoxon {p['wilcoxon_exact_p']:.3f} sign {p['sign_p']:.3f}")
