"""Paper package 1 (11 Nov 2026): every number from FROZEN, already-read fresh-set results; nothing new is run.  usage: python -I p251_paper_package.py <out_dir>
Fresh blind set K07-X3: 12 new architectures (10 composite + 2 OOD) x 3 weather windows (day-1 class C/M/H) = 36 scenes, 30-day episodes (8640 steps).
Loss per run = mean |reward| over 8640 steps.  Gain of arm x over reference y (paper Eq. 2): G = 100 (1 - sum L_x / sum L_y) over the composite cells (positive = better).
95 % interval: architecture-cluster percentile bootstrap (the 10 composite architectures resampled with replacement, all 3 cells of each kept, 3000 replicates, seed 1).
Also: per-architecture wins, worst architecture, gain by day-1 class, OOD gain, B-referenced gains (Eq. 3 differences in pp) and the weighted-component decomposition.
Outputs: PAPER_PACKAGE_1.json, per_scene_losses_fresh.csv, tables.md."""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
MAN = [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv", f"{N}/k07supw2/K07SUPW2_FRESH_MANIFEST.csv", f"{N}/pgf/PGF_RUN_MANIFEST.csv"]
DIRS = ["results_k07x3", "results_k07x3h", "results_k07supw2_fresh", "results_pgf"]
NAME = {"B": "K07-B", "G30": "K07-G30", "ROS": "K07-G30+LOS", "ROSFIX": "K07-G30+FOS", "SUP": "K07-G30+SUP", "SUPW2": "K07-G30+SUP+WC", "V0": "PENGUIN", "PC": "PENGUIN+PC"}
man = [r for f in MAN for r in csv.DictReader(open(f))]
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
L, C = {}, {}
for d in DIRS:
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or r["policy"] not in NAME or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean())
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for nm, (c, wt) in W.items()}
n = collections.Counter(k[2] for k in L); assert all(n[p] == 36 for p in NAME), n
cells = sorted({(a, w) for a, w, p in L}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def S(p, archs, cl=None): return sum(L[(a, w, p)] for a, w in cells if a in archs and (cl is None or cls[(a, w)] == cl))
def gain(x, y, archs=CO, cl=None): return 100 * (1 - S(x, archs, cl) / S(y, archs, cl))
def contrast(x, y):
    ga = {a: gain(x, y, [a]) for a in CO + OOD}; reps = [gain(x, y, [CO[i] for i in b]) for b in BOOT]
    return {"x": NAME[x], "ref": NAME[y], "gain_pct": gain(x, y), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "composite_better": sum(1 for a in CO if ga[a] > 0), "composite_changed": sum(1 for a in CO if ga[a] != 0), "composite_n": len(CO),
            "worst_architecture_gain_pct": min(ga[a] for a in CO), "by_class_gain_pct": {c: gain(x, y, CO, c) for c in ("C", "M", "H")},
            "ood_gain_pct": gain(x, y, OOD), "ood_better": sum(1 for a in OOD if ga[a] > 0), "per_architecture_gain_pct": ga}
res = {"runs": dict(n), "composite_architectures": CO, "ood_architectures": OOD, "names": NAME}
PAIRS = [("G30", "B"), ("SUP", "G30"), ("SUP", "B"), ("ROS", "G30"), ("ROSFIX", "G30"), ("ROS", "ROSFIX"), ("SUP", "ROS"), ("SUPW2", "SUP"), ("PC", "V0"), ("V0", "G30"), ("V0", "SUP")]
res["contrasts"] = {f"{x}_vs_{y}": contrast(x, y) for x, y in PAIRS}
gB = {p: gain(p, "B") for p in ("G30", "ROSFIX", "ROS", "SUP", "SUPW2")}; res["B_referenced_gain_pct"] = gB
res["B_referenced_differences_pp"] = {"engineering (G30 - B)": gB["G30"], "fixed start (FOS - G30)": gB["ROSFIX"] - gB["G30"], "learning (LOS - FOS)": gB["ROS"] - gB["ROSFIX"],
                                     "pressure reset (SUP - LOS)": gB["SUP"] - gB["ROS"], "method total (SUP - G30)": gB["SUP"] - gB["G30"], "combination (SUP - B)": gB["SUP"]}
def comp_change(x, y):
    tot = sum(sum(C[(a, w, y)].values()) for a, w in cells if a in CO)
    return {nm: 100 * sum(C[(a, w, x)][nm] - C[(a, w, y)][nm] for a, w in cells if a in CO) / tot for nm in W}
res["component_change_pct_of_ref_loss"] = {f"{x}_vs_{y}": comp_change(x, y) for x, y in (("G30", "B"), ("SUP", "G30"), ("ROS", "ROSFIX"), ("PC", "V0"))}
json.dump(res, open(f"{OUT}/PAPER_PACKAGE_1.json", "w"), indent=1)
with open(f"{OUT}/per_scene_losses_fresh.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["architecture", "group", "weather", "day1_class"] + [NAME[p] for p in NAME])
    for a, w in cells: wr.writerow([a, grp[a], w, cls[(a, w)]] + [f"{L[(a, w, p)]:.8f}" for p in NAME])
f = lambda v: f"{v:+.2f}"
md = ["| Contrast | Gain (%) | 95% CI | Better (composite) | Worst arch. | C | M | H | OOD |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in res["contrasts"].items():
    md.append(f"| {v['x']} vs {v['ref']} | {f(v['gain_pct'])} | [{f(v['ci95'][0])}, {f(v['ci95'][1])}] | {v['composite_better']}/{v['composite_n']} | {f(v['worst_architecture_gain_pct'])} | "
              f"{f(v['by_class_gain_pct']['C'])} | {f(v['by_class_gain_pct']['M'])} | {f(v['by_class_gain_pct']['H'])} | {f(v['ood_gain_pct'])} ({v['ood_better']}/2) |")
md += ["", "| B-referenced decomposition | pp |", "|---|---|"] + [f"| {k} | {f(v)} |" for k, v in res["B_referenced_differences_pp"].items()]
md += ["", "| Component change (% of reference loss) | energy | carbon | comfort |", "|---|---|---|---|"] + [f"| {k} | {f(v['energy'])} | {f(v['carbon'])} | {f(v['comfort'])} |" for k, v in res["component_change_pct_of_ref_loss"].items()]
open(f"{OUT}/tables.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
