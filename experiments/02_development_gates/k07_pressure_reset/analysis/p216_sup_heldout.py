"""SUP held-out (PREREG_K07SUP_COMBINED_v1, analysis 2; run ONLY after the development gate passed).  usage: python -I p216_ros_heldout.py <out.json> <results_dir> [...]
Per held-out architecture (both minneapolis strips pooled): r = L_SUP / L_G30 - 1.  Rule (as p206, ROS vs G30): composite pooled < 0, ROS better on >= 3 of 4 composite architectures,
no held-out architecture worse than +2 %.  Plus per-cell values, ROS behaviour and the comfort / energy / carbon split (mean per step)."""
import json, glob, csv, sys, numpy as np, os
H = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]; DIRS = sys.argv[2:]
W = {"energy": ("normalized_energy_cost", -0.2), "carbon": ("normalized_carbon_emission", -0.2), "comfort": ("normalized_productivity_regret", 0.6)}
man = list(csv.DictReader(open(f"{H}/K07SUP_SEL_HELDOUT_MANIFEST.csv"))); man = [r for r in man if r["policy"] in ("G30", "SUP")]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
L, CP, BEH = {}, {}, {}
for d in DIRS:
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if r and s["status"] == "OK" and s["steps_done"] == 8640 and s.get("illegal_actions", 0) == 0:
            z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; k = (r["architecture"], r["weather"], r["policy"])
            L[k] = float(np.abs(z["rewards"]).mean()); CP[k] = {nm: float((-w * z["comps"][:, ck.index(kk)]).mean()) for nm, (kk, w) in W.items()}
            if r["policy"] == "SUP":
                BEH[f"{r['architecture']}|{r['weather']}"] = {"override_share": (sum(s["tr"]["override_steps"]) / max(1, sum(s["tr"]["pi_steps"]))), "ahus_with_override": sum(1 for x in s["tr"]["override_steps"] if x > 0), "n_ahus": len(s["tr"]["override_steps"]), "frozen_ahus": s["tr"]["frozen_ahus"], "unresponsive_zones": len(s["tr"]["unresponsive_zones"]), "ros_extended_mornings": sum(1 for e in s["ros"]["log"] if e["event"] == "start" and e["why"] == "need"), "ros_frozen": s["ros"]["frozen"]}
cells = sorted({(r["architecture"], r["weather"]) for r in man}); ok = [c for c in cells if (c[0], c[1], "SUP") in L and (c[0], c[1], "G30") in L]; archs = sorted({c[0] for c in ok})
ra = {a: sum(L[(c[0], c[1], "SUP")] for c in ok if c[0] == a) / sum(L[(c[0], c[1], "G30")] for c in ok if c[0] == a) - 1 for a in archs}; comp = [a for a in archs if grp[a] == "composite"]
pooled = sum(L[(c[0], c[1], "SUP")] for c in ok if c[0] in comp) / sum(L[(c[0], c[1], "G30")] for c in ok if c[0] in comp) - 1 if comp else None
rule = {"composite_pooled_lt_0": pooled is not None and pooled < 0, "composite_better_ge_3_of_4": sum(1 for a in comp if ra[a] < 0) >= 3, "no_architecture_worse_than_2pct": all(v <= 0.02 for v in ra.values())}
res = {"cells": len(ok), "missing": [c for c in cells if c not in ok], "r_SUP_vs_G30_pct": {a: 100 * v for a, v in ra.items()}, "composite_pooled_pct": None if pooled is None else 100 * pooled,
       "composite_mean_pct": 100 * float(np.mean([ra[a] for a in comp])) if comp else None, "per_cell_pct": {f"{c[0]}|{c[1]}": 100 * (L[(c[0], c[1], "SUP")] / L[(c[0], c[1], "G30")] - 1) for c in ok},
       "split_mean_diff_SUP_minus_G30": {nm: float(np.mean([CP[(c[0], c[1], "SUP")][nm] - CP[(c[0], c[1], "G30")][nm] for c in ok])) for nm in W}, "sup_behaviour": BEH, "rule": rule, "SUP_PASSES_HELDOUT": all(rule.values())}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k: res[k] for k in ("cells", "r_SUP_vs_G30_pct", "composite_pooled_pct", "rule", "SUP_PASSES_HELDOUT")}))
