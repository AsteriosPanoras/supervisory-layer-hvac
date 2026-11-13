"""p277b = p277 with the corrected cluster bootstrap (PREREG_AXSUP_v1 AMENDMENT 1, registered before any blind outcome was read).  p277
inherited from p252 a resampling in which an architecture drawn twice counted once (sum over 'a in archs'); here every drawn architecture
enters with all its episodes as often as drawn (as p281), and the interval of the old rule is reported next to it for transparency.
Everything else is p277 unchanged.  The unchanged layer on the public Residual PPO controller, blind set (PREREG_AXSUP_v1); written 2026-11-11 before any blind outcome of
this controller.  Same rules as p252 (ZT Best).  usage: python -I p277_axsup_blind.py <out.json>
Runs only if RESULT_AXSUP_FIDELITY.json says ALL_PASS and all 36 AX and 36 AXS runs are complete; otherwise it reads no AX blind file.
Gain = 100 (1 - sum L_x / sum L_ref) pooled over the 10 composite architectures (L = mean |reward| over 8640 steps); 95 % architecture-
cluster bootstrap (3000, seed 1).  Primary rule 'the layer helps the neural controller': AXS vs AX composite pooled gain > 0 AND a
majority of the changed composite architectures better AND no architecture worse than 1 %.
Also: by day-1 class, OOD, weighted-component decomposition, supervisor activity, and descriptive AX vs K07-G30, AX vs ZT Best, and the
layer's gain on the three controllers on the same scenes.  Sources are told apart by folder: K07 (results_k07x3), ZT Best
(results_ztsup_fresh), Residual PPO (results_axsup_blind and axsup/laptop/work_axb*), because ZT Best and AX share policy keys."""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
fid = json.load(open(f"{H}/RESULT_AXSUP_FIDELITY.json")) if os.path.exists(f"{H}/RESULT_AXSUP_FIDELITY.json") else {}
if not fid.get("ALL_PASS"): json.dump({"status": "NOT RUN: fidelity not passed"}, open(OUT, "w")); sys.exit(print("fidelity not passed: AX blind not read"))
SRC = {"k07": [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], "zt": [f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], "ax": [f"{H}/AXSUP_BLIND_MANIFEST.csv"]}
key2, grp, cls = {}, {}, {}
for src, fs in SRC.items():
    for f in fs:
        for r in csv.DictReader(open(f)):
            key2[(r["architecture"], r["weather"], r["policy_key"], src)] = r; grp[r["architecture"]] = r["group"]; cls[(r["architecture"], r["weather"])] = r["day1_class"]
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
L, C, S = {}, {}, {}
PATS = ((f"{N}/results_k07x3/**/*.summary.json", "k07"), (f"{N}/results_ztsup_fresh/**/*.summary.json", "zt"),
        (f"{N}/results_axsup_blind/**/*.summary.json", "ax"), (f"{H}/laptop/work_axb*/out/**/*.summary.json", "ax"))
for pat, src in PATS:
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"], src))
        if not r or r["policy"] not in ("G30", "SUP", "ZB", "ZS", "AX", "AXS") or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean()); S[k] = s
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for nm, (c, wt) in W.items()}
n = collections.Counter(k[2] for k in L)
if n["AX"] < 36 or n["AXS"] < 36: json.dump({"status": f"INCOMPLETE {dict(n)}"}, open(OUT, "w")); sys.exit(print("incomplete", dict(n)))
cells = sorted({(a, w) for a, w, p in L if p == "AX"}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def Ssum(p, archs, cl=None): return sum(L[(a, w, p)] for aa in archs for a, w in cells if a == aa and (cl is None or cls[(a, w)] == cl))      # multiplicity
def Ssum_once(p, archs, cl=None): return sum(L[(a, w, p)] for a, w in cells if a in archs and (cl is None or cls[(a, w)] == cl))     # the p277 rule, reported only
def gain(x, y, archs=CO, cl=None): return 100 * (1 - Ssum(x, archs, cl) / Ssum(y, archs, cl))
def contrast(x, y):
    ga = {a: gain(x, y, [a]) for a in CO + OOD}; reps = [gain(x, y, [CO[i] for i in b]) for b in BOOT]; ch = [a for a in CO if ga[a] != 0.0]
    reps_once = [100 * (1 - Ssum_once(x, [CO[i] for i in b]) / Ssum_once(y, [CO[i] for i in b])) for b in BOOT]
    return {"gain_pct": gain(x, y), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "ci95_p277_rule_once": [float(np.percentile(reps_once, 2.5)), float(np.percentile(reps_once, 97.5))], "composite_changed": len(ch), "composite_changed_better": sum(1 for a in ch if ga[a] > 0),
            "worst_architecture_gain_pct": min(ga.values()), "by_class_gain_pct": {c: gain(x, y, CO, c) for c in ("C", "M", "H")}, "ood_gain_pct": gain(x, y, OOD), "per_architecture_gain_pct": ga}
res = {"runs": dict(n), "AXS_vs_AX": contrast("AXS", "AX"), "AX_vs_G30": contrast("AX", "G30"), "AX_vs_ZB": contrast("AX", "ZB"),
       "SUP_vs_G30_same_cells": contrast("SUP", "G30"), "ZS_vs_ZB_same_cells": contrast("ZS", "ZB")}
v = res["AXS_vs_AX"]; res["rule_layer_helps"] = {"composite_pooled_gain_gt_0": v["gain_pct"] > 0, "majority_of_changed_better": v["composite_changed"] > 0 and 2 * v["composite_changed_better"] > v["composite_changed"],
                                                 "no_architecture_worse_than_1pct": v["worst_architecture_gain_pct"] >= -1.0}
res["LAYER_HELPS"] = all(res["rule_layer_helps"].values())
tot = sum(sum(C[(a, w, "AX")].values()) for a, w in cells if a in CO)
res["component_change_pct_of_AX_loss"] = {nm: 100 * sum(C[(a, w, "AXS")][nm] - C[(a, w, "AX")][nm] for a, w in cells if a in CO) / tot for nm in W}
res["AX_comfort_share_composite"] = sum(C[(a, w, "AX")]["comfort"] for a, w in cells if a in CO) / tot
ext = 0; ovr = []
for (a, w, p), s in S.items():
    if p != "AXS": continue
    ext += sum(1 for e in (s.get("ros") or {}).get("log", []) if e["event"] == "start" and e.get("why") == "need")
    tr = s.get("tr") or {}; ps = sum(tr.get("pi_steps") or [0]); os_ = sum(tr.get("override_steps") or [0]); ovr.append(os_ / ps if ps else 0.0)
res["supervisor_activity"] = {"LOS_need_extensions_total": ext, "TR_override_share_mean": float(np.mean(ovr)) if ovr else None}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("AXS_vs_AX", "AX_vs_G30", "AX_vs_ZB", "SUP_vs_G30_same_cells", "ZS_vs_ZB_same_cells"):
    v = res[k]; print(k, "| gain %", round(v["gain_pct"], 3), "CI", [round(x, 3) for x in v["ci95"]], "| changed/better", v["composite_changed"], v["composite_changed_better"],
                      "| worst", round(v["worst_architecture_gain_pct"], 3), "| OOD", round(v["ood_gain_pct"], 3))
print("LAYER_HELPS", res["LAYER_HELPS"], res["rule_layer_helps"], "| components", {k: round(x, 3) for k, x in res["component_change_pct_of_AX_loss"].items()},
      "| AX comfort share", round(res["AX_comfort_share_composite"], 4), "| activity", res["supervisor_activity"])
