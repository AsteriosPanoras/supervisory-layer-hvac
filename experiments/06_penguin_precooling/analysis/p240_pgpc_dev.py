"""Penguin night pre-cooling, development analysis (PREREG_PGPC_v1); written before any PC outcome.  usage: python -I p240_pgpc_dev.py <out.json>
Loss per run = mean |reward| over 8640 steps.  Per development architecture r = sum over its strips L_PC / sum L_V0 - 1 (V0 = native Penguin, pgx@static@).
Gate: mean r over the composite development architectures < 0 AND no development architecture worse than +1 %.  Also: per scene; component split (energy, carbon, comfort, per step,
weights 0.2/0.2/0.6 as p227) PC - V0; pre-cooling steps, learned need and frozen flag per scene."""
import sys, json, glob, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
coh = json.load(open(f"{N}/k07x/K07X_V2_COHORT.json"))["cells"]; dev = {(c["architecture"], c["weather"]) for c in coh if c["split"] == "dev"}; grp = {c["architecture"]: c["group"] for c in coh}
KEYS = {"pgx@static@": "V0", "pgx@sup@ROS=0,TR=0,PC=1": "PC"}; W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
L, CP, BEH = {}, {}, {}
for d in ("results_pgx", "results_pgxsup", "results_pgpc"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        b = os.path.basename(f)
        if not any(b.startswith(f"{a}_{w}_") for a, w in dev): continue
        s = json.load(open(f)); p = KEYS.get(s["policy_key"]); c = (s["building"], s["weather"])
        if p is None or c not in dev or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; L[(*c, p)] = float(np.abs(z["rewards"]).mean())
        CP[(*c, p)] = {nm: float(np.abs(wt * z["comps"][:, ck.index(k)]).mean()) for nm, (k, wt) in W.items()}
        if p == "PC": BEH[f"{c[0][-5:]}|{c[1]}"] = {"precool_steps": s["pc"]["precool_steps"], "need_final_k": s["pc"]["need_final_k"], "frozen": s["pc"]["frozen"]}
cells = sorted(c for c in dev if (*c, "PC") in L and (*c, "V0") in L); archs = sorted({a for a, w in cells}); co = [a for a in archs if grp[a] == "composite"]
ra = {a: sum(L[(a, w, "PC")] for aa, w in cells if aa == a) / sum(L[(a, w, "V0")] for aa, w in cells if aa == a) - 1 for a in archs}
cm = float(np.mean([ra[a] for a in co])) if co else None
gate = {"composite_mean_lt_0": cm is not None and cm < 0, "every_dev_architecture_le_1pct": bool(ra) and all(v <= 0.01 for v in ra.values()), "all_18_cells": len(cells) == 18}
res = {"cells": len(cells), "r_PC_vs_V0_pct": {a: 100 * v for a, v in ra.items()}, "composite_mean_pct": None if cm is None else 100 * cm, "gate": gate, "GATE_PASS": all(gate.values()),
       "per_scene_pct": {f"{a[-5:]}|{w}": 100 * (L[(a, w, "PC")] / L[(a, w, "V0")] - 1) for a, w in cells},
       "component_diff_mean_per_step": {nm: float(np.mean([CP[(a, w, "PC")][nm] - CP[(a, w, "V0")][nm] for a, w in cells])) for nm in W} if cells else {}, "behaviour": BEH}
json.dump(res, open(OUT, "w"), indent=1)
print("GATE_PASS", res["GATE_PASS"], gate, "| composite mean %", res["composite_mean_pct"]); print("per scene %:", {k: round(v, 2) for k, v in res["per_scene_pct"].items() if abs(v) > 0.005})
print("component diff:", res["component_diff_mean_per_step"]); print("behaviour:", {k: v for k, v in BEH.items() if v["precool_steps"]})
