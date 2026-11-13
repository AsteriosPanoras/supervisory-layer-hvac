"""p285: the unchanged layer on the graph-network controller (GN = the Residual PPO base rule with the imitation-trained GraphNets residual;
GNS = GN with the unchanged supervisor), blind set (PREREG_GNNSUP_v1).  Written 2026-11-12 before any GNN blind run; the logic is p277b
(standard architecture-cluster bootstrap, 3000 replicates, seed 1; gain pooled over the 10 composite architectures) with the arms
renamed.  Primary rule (same as AXSUP): GNS vs GN composite pooled gain > 0 AND a majority of the changed composite architectures
better AND no architecture worse than 1 %.  Descriptive: GN vs the teacher (AX), GN vs K07-G30, the layer on AX and on K07-G30 on
the same scenes.  Runs only if RESULT_GNNSUP_FIDELITY.json says ALL_PASS and all 36 GN and 36 GNS runs are complete.
usage: python -I p285_gnnsup_blind.py <out.json>"""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
fid = json.load(open(f"{H}/RESULT_GNNSUP_FIDELITY.json")) if os.path.exists(f"{H}/RESULT_GNNSUP_FIDELITY.json") else {}
if not fid.get("ALL_PASS"): json.dump({"status": "NOT RUN: fidelity not passed"}, open(OUT, "w")); sys.exit(print("fidelity not passed: GNN blind not read"))
SRC = {"k07": [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], "zt": [f"{N}/ztsup/ZTSUP_FRESH_MANIFEST.csv"], "ax": [f"{N}/axsup/AXSUP_BLIND_MANIFEST.csv"], "gn": [f"{H}/GNNSUP_BLIND_MANIFEST.csv"]}
key2, grp, cls = {}, {}, {}
for src, fs in SRC.items():
    for f in fs:
        for r in csv.DictReader(open(f)):
            key2[(r["architecture"], r["weather"], r["policy_key"], src)] = r; grp[r["architecture"]] = r["group"]; cls[(r["architecture"], r["weather"])] = r["day1_class"]
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
L, C, S = {}, {}, {}
PATS = ((f"{N}/results_k07x3/**/*.summary.json", "k07"), (f"{N}/results_ztsup_fresh/**/*.summary.json", "zt"),
        (f"{N}/results_axsup_blind/**/*.summary.json", "ax"), (f"{N}/axsup/laptop/work_axb*/out/**/*.summary.json", "ax"),
        (f"{N}/results_gnnsup_blind/**/*.summary.json", "gn"), (f"{H}/laptop/work_gnb*/out/**/*.summary.json", "gn"))
for pat, src in PATS:
    for f in glob.glob(pat, recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"], src))
        if not r or r["policy"] not in ("G30", "SUP", "ZB", "ZS", "AX", "AXS", "GN", "GNS") or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"])
        if k in L: raise SystemExit(f"duplicate {k}: {f}")
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); L[k] = float(np.abs(z["rewards"]).mean()); S[k] = s
        ck = [str(x) for x in z["comp_keys"]]; C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for nm, (c, wt) in W.items()}
n = collections.Counter(k[2] for k in L)
if n["GN"] < 36 or n["GNS"] < 36: json.dump({"status": f"INCOMPLETE {dict(n)}"}, open(OUT, "w")); sys.exit(print("incomplete", dict(n)))
cells = sorted({(a, w) for a, w, p in L if p == "GN"}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def Ssum(p, archs, cl=None): return sum(L[(a, w, p)] for aa in archs for a, w in cells if a == aa and (cl is None or cls[(a, w)] == cl))      # multiplicity
def Ssum_once(p, archs, cl=None): return sum(L[(a, w, p)] for a, w in cells if a in archs and (cl is None or cls[(a, w)] == cl))     # the p277 rule, reported only
def gain(x, y, archs=CO, cl=None): return 100 * (1 - Ssum(x, archs, cl) / Ssum(y, archs, cl))
def contrast(x, y):
    ga = {a: gain(x, y, [a]) for a in CO + OOD}; reps = [gain(x, y, [CO[i] for i in b]) for b in BOOT]; ch = [a for a in CO if ga[a] != 0.0]
    reps_once = [100 * (1 - Ssum_once(x, [CO[i] for i in b]) / Ssum_once(y, [CO[i] for i in b])) for b in BOOT]
    return {"gain_pct": gain(x, y), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "ci95_p277_rule_once": [float(np.percentile(reps_once, 2.5)), float(np.percentile(reps_once, 97.5))], "composite_changed": len(ch), "composite_changed_better": sum(1 for a in ch if ga[a] > 0),
            "worst_architecture_gain_pct": min(ga.values()), "by_class_gain_pct": {c: gain(x, y, CO, c) for c in ("C", "M", "H")}, "ood_gain_pct": gain(x, y, OOD), "per_architecture_gain_pct": ga}
res = {"runs": dict(n), "GNS_vs_GN": contrast("GNS", "GN"), "GN_vs_AX": contrast("GN", "AX"), "GN_vs_G30": contrast("GN", "G30"),
       "AXS_vs_AX_same_cells": contrast("AXS", "AX"), "SUP_vs_G30_same_cells": contrast("SUP", "G30")}
v = res["GNS_vs_GN"]; res["rule_layer_helps"] = {"composite_pooled_gain_gt_0": v["gain_pct"] > 0, "majority_of_changed_better": v["composite_changed"] > 0 and 2 * v["composite_changed_better"] > v["composite_changed"],
                                                 "no_architecture_worse_than_1pct": v["worst_architecture_gain_pct"] >= -1.0}
res["LAYER_HELPS"] = all(res["rule_layer_helps"].values())
tot = sum(sum(C[(a, w, "GN")].values()) for a, w in cells if a in CO)
res["component_change_pct_of_GN_loss"] = {nm: 100 * sum(C[(a, w, "GNS")][nm] - C[(a, w, "GN")][nm] for a, w in cells if a in CO) / tot for nm in W}
res["GN_comfort_share_composite"] = sum(C[(a, w, "GN")]["comfort"] for a, w in cells if a in CO) / tot
ext = 0; ovr = []
for (a, w, p), s in S.items():
    if p != "GNS": continue
    ext += sum(1 for e in (s.get("ros") or {}).get("log", []) if e["event"] == "start" and e.get("why") == "need")
    tr = s.get("tr") or {}; ps = sum(tr.get("pi_steps") or [0]); os_ = sum(tr.get("override_steps") or [0]); ovr.append(os_ / ps if ps else 0.0)
res["supervisor_activity"] = {"LOS_need_extensions_total": ext, "TR_override_share_mean": float(np.mean(ovr)) if ovr else None}
json.dump(res, open(OUT, "w"), indent=1)
for k in ("GNS_vs_GN", "GN_vs_AX", "GN_vs_G30", "AXS_vs_AX_same_cells", "SUP_vs_G30_same_cells"):
    v = res[k]; print(k, "| gain %", round(v["gain_pct"], 3), "CI", [round(x, 3) for x in v["ci95"]], "| changed/better", v["composite_changed"], v["composite_changed_better"],
                      "| worst", round(v["worst_architecture_gain_pct"], 3), "| OOD", round(v["ood_gain_pct"], 3))
print("LAYER_HELPS", res["LAYER_HELPS"], res["rule_layer_helps"], "| components", {k: round(x, 3) for k, x in res["component_change_pct_of_GN_loss"].items()},
      "| GN comfort share", round(res["GN_comfort_share_composite"], 4), "| activity", res["supervisor_activity"])
