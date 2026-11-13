"""p281: correction of the architecture-cluster bootstrap of the blind-set contrasts (found 11 Nov 2026 ~21:40 while building the
supplementary package: supplementary_v1/scripts/reproduce_tables.py reproduced every point estimate but not the intervals).
p254 (Table 3, Table tab:tests, evidence table) and p252 (ZT Best) drew 10 architectures with replacement but summed the episodes with
'a in archs', so an architecture drawn twice counted once: each replicate was a random SUBSET of distinct architectures, not a cluster
bootstrap sample, and the intervals were too narrow.  This script recomputes, with the same seed and draws (rng = default_rng(1),
BOOT = 3,000 x integers(0, 10, 10)), both versions: 'once' (as printed; must reproduce the paper exactly, gate) and 'multiplicity'
(the standard cluster bootstrap: every drawn architecture enters with all its episodes as many times as drawn).  Also the bootstrap
p-values (2 min(P(rep <= 0), P(rep >= 0))) and their Holm adjustment over the six secondary contrasts of p254.  Point estimates,
Wilcoxon and sign tests do not depend on the bootstrap and are unchanged.  Loading as p254 / p252.  usage: python -I p281_bootstrap_correction.py <out.json>"""
import sys, os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
SRC = [("K07", [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv", f"{N}/k07supw2/K07SUPW2_FRESH_MANIFEST.csv", f"{N}/pgf/PGF_RUN_MANIFEST.csv"],
        ["results_k07x3", "results_k07x3h", "results_k07supw2_fresh", "results_pgf"]),
       ("ZT", [f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], ["results_ztsup_fresh"])]
L, grp = {}, {}
for tag, mans, dirs in SRC:
    man = [r for f in mans for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
    for r in man: grp[r["architecture"]] = r["group"]
    for d in dirs:
        for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
            s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
            if not r or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
            k = (tag, r["architecture"], r["weather"], r["policy"])
            if k in L: raise SystemExit(f"duplicate {k}")
            L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
CO = sorted({a for a in grp if grp[a] == "composite"}); rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def gain(tag, x, y, archs, multi):
    cells = sorted({(a, w) for t, a, w, p in L if t == tag and p == x})
    sel = [(a, w) for aa in archs for a, w in cells if a == aa] if multi else [(a, w) for a, w in cells if a in archs]
    return 100 * (1 - sum(L[(tag, a, w, x)] for a, w in sel) / sum(L[(tag, a, w, y)] for a, w in sel))
C = [("K07", "SUP", "G30"), ("K07", "G30", "B"), ("K07", "SUP", "B"), ("K07", "ROS", "G30"), ("K07", "ROSFIX", "G30"), ("K07", "ROS", "ROSFIX"), ("K07", "SUP", "ROS"),
     ("K07", "SUPW2", "SUP"), ("K07", "PC", "V0"), ("ZT", "ZS", "ZB")]
SEC = [("K07", "ROS", "G30"), ("K07", "ROSFIX", "G30"), ("K07", "ROS", "ROSFIX"), ("K07", "SUP", "ROS"), ("K07", "SUPW2", "SUP"), ("K07", "PC", "V0")]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
res = {"note": "blind-set bootstrap: 'once' = as printed (each drawn architecture counted once), 'multiplicity' = standard cluster bootstrap", "contrasts": {}}
for multi, lab in ((False, "once"), (True, "multiplicity")):
    for c in C:
        reps = np.array([gain(*c, [CO[i] for i in b], multi) for b in BOOT]); k = f"{c[1]}_vs_{c[2]}" + ("" if c[0] == "K07" else "_ZT")
        res["contrasts"].setdefault(k, {"gain_pct": gain(*c, CO, True)})[lab] = {"ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
                                                                                  "p_boot": float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))}
    ad = holm([res["contrasts"][f"{c[1]}_vs_{c[2]}"][lab]["p_boot"] for c in SEC])
    for c, a in zip(SEC, ad): res["contrasts"][f"{c[1]}_vs_{c[2]}"][lab]["p_holm"] = a
PRINTED = {"G30_vs_B": [6.56, 11.02], "SUP_vs_G30": [1.53, 3.73], "SUP_vs_B": [8.37, 13.71], "ROS_vs_ROSFIX": [0.12, 1.53], "SUP_vs_ROS": [0.63, 2.28], "PC_vs_V0": [1.24, 2.92], "ZS_vs_ZB_ZT": [3.94, 7.14]}
gate = {k: [round(x, 2) for x in res["contrasts"][k]["once"]["ci95"]] == v for k, v in PRINTED.items()}
res["gate_once_reproduces_printed_intervals"] = gate; assert all(gate.values()), gate
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["contrasts"].items():
    print(f"{k:16s} gain {v['gain_pct']:+.2f} | printed: [{v['once']['ci95'][0]:+.2f}, {v['once']['ci95'][1]:+.2f}] p {v['once']['p_boot']:.4f} holm {v['once'].get('p_holm', float('nan')):.4f}"
          f" | corrected: [{v['multiplicity']['ci95'][0]:+.2f}, {v['multiplicity']['ci95'][1]:+.2f}] p {v['multiplicity']['p_boot']:.4f} holm {v['multiplicity'].get('p_holm', float('nan')):.4f}")
