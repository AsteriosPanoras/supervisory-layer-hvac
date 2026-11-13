"""p279: the neural controller on the six official training scenes, before the leaderboard check (the same scenes and the same
diagnosis as for K07: Appendix J, Table 'unoccupied setup on six official training scenes').  Written 2026-11-11 before any of these
runs finished.  Research runner run_episode_k07_v9supe_ax.py with the PLAIN kit (the competition code, as the gateway and the
leaderboard), 8640 steps; arms AX (k07@) and AXS (k07sup@;ROS=1,TR=1).
Per scene and pooled: gain of AXS over AX = 100 (1 - sum L_AXS / sum L_AX), L = mean |reward|; AX's weighted shares (comfort 0.6,
energy 0.2, carbon 0.2) and the share of its energy + carbon spent in unoccupied hours; the supervisor's activity.
These numbers are the basis of the leaderboard prediction PREDICTION_AX_LB_v1.md, written and hashed before any submission.
Caveat stated in advance: the network was trained on official buildings 1, 2, 4, 5 and 7, so these scenes are in sample for it.
usage: python -I p279_ax_official_diagnosis.py <diag_dir> <out.json>"""
import sys, glob, json, numpy as np
D, OUT = sys.argv[1], sys.argv[2]
SC = [("building_train_1", "denver_7"), ("building_train_1", "fairbanks_1"), ("building_train_1", "phoenix_5"), ("building_train_1", "singapore_8"),
      ("building_train_2", "fairbanks_2"), ("building_train_4", "phoenix_8")]
KEYS = {"AX": "k07__", "AXS": "k07sup__;ROS-1_TR-1"}
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
R, S = {}, {}
for b, w in SC:
    for arm, sk in KEYS.items():
        f = glob.glob(f"{D}/{b}_{w}_{sk}_8640.summary.json"); assert len(f) == 1, (b, w, arm, f)
        s = json.load(open(f[0])); assert s["status"] == "OK" and s["steps_done"] == 8640 and s["illegal_actions"] == 0, (b, w, arm, s["status"])
        z = np.load(f[0].replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; occ = z["occupancy"].astype(float) > 0
        comp = {nm: np.abs(wt * z["comps"][:, ck.index(c)].astype(float)) for nm, (c, wt) in W.items()}
        R[(b, w, arm)] = {"L": float(np.abs(z["rewards"]).mean()), "comp": {k: float(v.mean()) for k, v in comp.items()},
                         "unocc_ec": float((comp["energy"][~occ] + comp["carbon"][~occ]).sum() / np.abs(z["rewards"]).sum())}
        S[(b, w, arm)] = s
g = lambda sel: 100 * (1 - sum(R[(b, w, "AXS")]["L"] for b, w in sel) / sum(R[(b, w, "AX")]["L"] for b, w in sel))
res = {"note": "official training scenes (in sample for the network); plain kit; 8640 steps", "pooled_gain_pct": g(SC), "per_scene": {}}
for b, w in SC:
    ax = R[(b, w, "AX")]; tot = sum(ax["comp"].values()); s3 = S[(b, w, "AXS")]; tr = s3.get("tr") or {}
    res["per_scene"][f"{b}/{w}"] = {"gain_pct": g([(b, w)]), "AX_loss": ax["L"], "AX_share_comfort": ax["comp"]["comfort"] / tot, "AX_share_energy": ax["comp"]["energy"] / tot,
                                   "AX_share_carbon": ax["comp"]["carbon"] / tot, "AX_unoccupied_energy_carbon_share": ax["unocc_ec"],
                                   "learned_starts": sum(1 for e in (s3.get("ros") or {}).get("log", []) if e.get("event") == "start" and e.get("why") == "need"),
                                   "pressure_override_steps": sum(tr.get("override_steps") or [0])}
tot = {k: sum(R[(b, w, "AX")]["comp"][k] for b, w in SC) for k in W}; T = sum(tot.values())
res["pooled_AX_shares"] = {k: v / T for k, v in tot.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"pooled gain of the layer on the official scenes: {res['pooled_gain_pct']:+.3f} %; AX shares {({k: round(v, 3) for k, v in res['pooled_AX_shares'].items()})}")
for k, v in res["per_scene"].items():
    print(f"  {k:30s} gain {v['gain_pct']:+.2f} %  comfort share {v['AX_share_comfort']:.3f}  unocc E+C {v['AX_unoccupied_energy_carbon_share']:.3f}  starts {v['learned_starts']}  overrides {v['pressure_override_steps']}")
