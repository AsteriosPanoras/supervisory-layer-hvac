"""TR development gate (PREREG_K07TR_v1, analysis 1).  usage: python -I p215_ros_dev_gate.py <out.json> <results_dir> [...]
Per development architecture (both strips pooled): r = L_TR / L_G30 - 1.  GATE: composite mean r < 0 AND every development architecture r <= +1 %.  Also the frozen selection
procedure p205 (candidates B, G30, MG, MGG30, ROS; reference B) on the development manifest.  Descriptive: extended mornings, mean lead, final tau, fallback per run."""
import json, glob, csv, sys, subprocess, numpy as np, os
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; DIRS = sys.argv[2:]
man = list(csv.DictReader(open(f"{H}/K07TR_SEL_DEV_MANIFEST.csv"))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
L, ROSD = {}, {}
for d in DIRS:
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if r and s["status"] == "OK" and s["steps_done"] == 8640 and s.get("illegal_actions", 0) == 0:
            L[(r["architecture"], r["weather"], r["policy"])] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
            if r["policy"] == "TR":
                ROSD[(r["architecture"], r["weather"])] = {"override_share": (sum(s["tr"]["override_steps"]) / max(1, sum(s["tr"]["pi_steps"]))), "ahus_with_override": sum(1 for x in s["tr"]["override_steps"] if x > 0), "n_ahus": len(s["tr"]["override_steps"]), "frozen_ahus": s["tr"]["frozen_ahus"], "unresponsive_zones": len(s["tr"]["unresponsive_zones"])}
cells = sorted({(r["architecture"], r["weather"]) for r in man}); ok = [c for c in cells if (c[0], c[1], "TR") in L and (c[0], c[1], "G30") in L]
archs = sorted({c[0] for c in ok}); r = {a: sum(L[(c[0], c[1], "TR")] for c in ok if c[0] == a) / sum(L[(c[0], c[1], "G30")] for c in ok if c[0] == a) - 1 for a in archs}
comp = [a for a in archs if grp[a] == "composite"]; cm = float(np.mean([r[a] for a in comp])) if comp else None
gate = {"composite_mean_lt_0": cm is not None and cm < 0, "every_dev_architecture_le_1pct": all(v <= 0.01 for v in r.values()), "all_18_cells": len(ok) == 18}
sel = subprocess.run([sys.executable, "-I", f"{N}/selrule/p205_select_general.py", f"{H}/RESULT_K07TR_SELECTION.json", f"{H}/K07TR_SEL_DEV_MANIFEST.csv", "B", "B,G30,MG,MGG30,TR"] + DIRS, capture_output=True, text=True)
res = {"cells": len(ok), "missing": [c for c in cells if c not in ok], "r_TR_vs_G30_pct": {a: 100 * v for a, v in r.items()}, "composite_mean_pct": None if cm is None else 100 * cm,
       "per_cell_pct": {f"{c[0]}|{c[1]}": 100 * (L[(c[0], c[1], "TR")] / L[(c[0], c[1], "G30")] - 1) for c in ok}, "tr_behaviour": {f"{a}|{w}": v for (a, w), v in ROSD.items()},
       "gate": gate, "GATE_PASS": all(gate.values()), "selection_stdout": sel.stdout.strip()[-400:], "selection_rc": sel.returncode}
json.dump(res, open(OUT, "w"), indent=1); print("GATE_PASS", res["GATE_PASS"], gate, "| composite mean %", res["composite_mean_pct"], "|", sel.stdout.strip()[-200:])
