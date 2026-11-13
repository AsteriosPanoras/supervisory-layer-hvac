"""Paper analyses on the FROZEN fresh-set runs (already read; nothing new is run).  usage: python -I p253_paper_analyses.py <out_dir>
A. What each building taught the supervisor: the learned recovery constant tau (h) per scene and architecture for LOS and SUP, informative mornings, safety freezes.
B. Error analysis, activity vs gain: per scene, LOS activity (morning extensions 'need', mean start lead) and T&R activity (override share of occupied steps), against the per-scene
   gain vs G30; scenes where the supervisor never acted vs scenes where it acted; Spearman rank correlation.
C. Overhead: policy time per step (ms) and wall time for G30, LOS and SUP.
Outputs: PAPER_ANALYSES_2.json and analyses_2.md."""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
man = [r for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv") for r in csv.DictReader(open(f))]
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
L, S = {}, {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if not r or r["policy"] not in ("G30", "ROS", "SUP") or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
    k = (r["architecture"], r["weather"], r["policy"]); L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); S[k] = s
cells = sorted({(a, w) for a, w, p in L}); assert all((a, w, p) in L for a, w in cells for p in ("G30", "ROS", "SUP")) and len(cells) == 36
def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
res = {"A_learned_tau": {}, "B_activity_vs_gain": {}, "C_overhead": {}}
# A. learned tau
for p in ("ROS", "SUP"):
    rows = {}
    for a, w in cells:
        ros = S[(a, w, p)].get("ros") or {}; lg = ros.get("log", [])
        rows[f"{a}|{w}"] = {"tau_final_h": ros.get("tau_final_h"), "n_informative": ros.get("n_informative"), "frozen": ros.get("frozen"), "class": cls[(a, w)], "group": grp[a],
                            "need_extensions": sum(1 for e in lg if e["event"] == "start" and e.get("why") == "need")}
    taus = [v["tau_final_h"] for v in rows.values() if v["n_informative"] and v["n_informative"] >= 2]
    per_arch = collections.defaultdict(list)
    for k, v in rows.items():
        if v["n_informative"] and v["n_informative"] >= 2: per_arch[k.split("|")[0]].append(v["tau_final_h"])
    res["A_learned_tau"][p] = {"per_scene": rows, "scenes_with_ge2_informative_mornings": len(taus), "tau_h_min_median_max": [float(min(taus)), float(np.median(taus)), float(max(taus))] if taus else None,
                               "per_architecture_mean_tau_h": {a: float(np.mean(v)) for a, v in sorted(per_arch.items())}, "safety_freezes": sum(1 for v in rows.values() if v["frozen"])}
# B. activity vs gain
def tr_share(s):
    tr = s.get("tr") or {}; ps = sum(tr.get("pi_steps") or [0]); return (sum(tr.get("override_steps") or [0]) / ps) if ps else 0.0
rows = []
for a, w in cells:
    s = S[(a, w, "SUP")]; lg = (s.get("ros") or {}).get("log", []); ext = [e for e in lg if e["event"] == "start" and e.get("why") == "need"]
    rows.append({"scene": f"{a}|{w}", "class": cls[(a, w)], "group": grp[a], "los_extensions": len(ext), "tr_override_share": tr_share(s),
                 "gain_SUP_vs_G30_pct": 100 * (1 - L[(a, w, "SUP")] / L[(a, w, "G30")]), "gain_LOS_vs_G30_pct": 100 * (1 - L[(a, w, "ROS")] / L[(a, w, "G30")])})
inact = [r for r in rows if r["los_extensions"] == 0 and r["tr_override_share"] == 0.0]; act = [r for r in rows if not (r["los_extensions"] == 0 and r["tr_override_share"] == 0.0)]
g = lambda R, k: [r[k] for r in R]
res["B_activity_vs_gain"] = {"per_scene": rows, "inactive_scenes": len(inact), "inactive_gain_SUP_pct_min_max": [min(g(inact, "gain_SUP_vs_G30_pct")), max(g(inact, "gain_SUP_vs_G30_pct"))] if inact else None,
                             "active_scenes": len(act), "active_gain_SUP_pct_mean": float(np.mean(g(act, "gain_SUP_vs_G30_pct"))) if act else None,
                             "active_scenes_worse_than_0p1pct": sum(1 for r in act if r["gain_SUP_vs_G30_pct"] < -0.1), "active_scenes_better": sum(1 for r in act if r["gain_SUP_vs_G30_pct"] > 0),
                             "spearman_tr_share_vs_gain_SUP": spearman(g(rows, "tr_override_share"), g(rows, "gain_SUP_vs_G30_pct")),
                             "spearman_los_extensions_vs_gain_LOS": spearman(g(rows, "los_extensions"), g(rows, "gain_LOS_vs_G30_pct")),
                             "by_class": {c: {"scenes": sum(1 for r in rows if r["class"] == c), "mean_los_extensions": float(np.mean([r["los_extensions"] for r in rows if r["class"] == c])),
                                              "mean_tr_override_share": float(np.mean([r["tr_override_share"] for r in rows if r["class"] == c]))} for c in ("C", "M", "H")}}
# C. overhead
for p in ("G30", "ROS", "SUP"):
    ms = [S[(a, w, p)].get("policy_ms_per_step") for a, w in cells if S[(a, w, p)].get("policy_ms_per_step") is not None]; ws = [S[(a, w, p)].get("wall_s") for a, w in cells if S[(a, w, p)].get("wall_s")]
    res["C_overhead"][p] = {"policy_ms_per_step_median": float(np.median(ms)) if ms else None, "policy_ms_per_step_max": float(max(ms)) if ms else None, "wall_s_median": float(np.median(ws)) if ws else None}
json.dump(res, open(f"{OUT}/PAPER_ANALYSES_2.json", "w"), indent=1)
A, B, C = res["A_learned_tau"], res["B_activity_vs_gain"], res["C_overhead"]
md = ["## A. Learned recovery constant (h)", "| Arm | scenes with >= 2 informative mornings | tau min / median / max | safety freezes |", "|---|---|---|---|"]
for p in ("ROS", "SUP"): md.append(f"| {'LOS' if p == 'ROS' else 'SUP'} | {A[p]['scenes_with_ge2_informative_mornings']}/36 | {' / '.join(f'{x:.2f}' for x in A[p]['tau_h_min_median_max']) if A[p]['tau_h_min_median_max'] else '-'} | {A[p]['safety_freezes']} |")
md += ["", "Per-architecture mean tau (SUP): " + ", ".join(f"{a[-5:]}: {v:.2f}" for a, v in A["SUP"]["per_architecture_mean_tau_h"].items()), "",
       "## B. Activity vs gain (SUP vs G30, per scene)", f"- scenes where the supervisor never acted: {B['inactive_scenes']} (gain range {B['inactive_gain_SUP_pct_min_max']})",
       f"- scenes where it acted: {B['active_scenes']} (mean gain {B['active_gain_SUP_pct_mean']:.2f} %; better {B['active_scenes_better']}; worse than -0.1 %: {B['active_scenes_worse_than_0p1pct']})",
       f"- Spearman: T&R override share vs gain {B['spearman_tr_share_vs_gain_SUP']:.2f}; LOS extensions vs LOS gain {B['spearman_los_extensions_vs_gain_LOS']:.2f}",
       "- by first-day class: " + "; ".join(f"{c}: extensions {v['mean_los_extensions']:.1f}/scene, T&R share {100 * v['mean_tr_override_share']:.1f} %" for c, v in B["by_class"].items()), "",
       "## C. Overhead", "| Arm | policy ms/step median (max) | wall s median |", "|---|---|---|"] + [f"| {p} | {v['policy_ms_per_step_median']:.2f} ({v['policy_ms_per_step_max']:.2f}) | {v['wall_s_median']:.0f} |" for p, v in C.items()]
open(f"{OUT}/analyses_2.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
