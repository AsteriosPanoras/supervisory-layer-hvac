"""p284: ZT Best alone (ZB, key k07@) and with the unchanged layer (ZS, key k07sup@;ROS=1,TR=1) on the six official training scenes, the
closest available proxy of the hidden leaderboard scenes.
Written and hashed before any of these runs exists.  Setting identical to the K07 official-scene stage of the paper (kernel_runner_v3 with
the KRE-v2 overlay on the competition-tracker kit, i.e. the corrected tracker 443aec77; the leaderboard itself runs the competition's
original tracker 87f69478, so this is a proxy), runner run_episode_k07_v9supe_zt.py (2b8144cb), ZT classes a0246bd4, 8640 steps.
Per scene and pooled: gain of ZS over ZB = 100 (1 - sum L_ZS / sum L_ZB), L = mean |reward|; ZB's weighted shares (comfort 0.6,
energy 0.2, carbon 0.2); the supervisor's activity.  These numbers feed a leaderboard prediction for ZT Best +/- layer, to be written by the
rule of PREDICTION_RULE_AX_LB_v1 (|g| < 0.5 % -> unchanged; >= +0.5 % -> better; <= -0.5 % -> worse) and hashed and timestamped before
any submission.  usage: python -I p284_zt_official_diagnosis.py <results_dir> <out.json>"""
import sys, glob, json, numpy as np
D, OUT = sys.argv[1], sys.argv[2]
SC = [("building_train_1", "denver_7"), ("building_train_1", "fairbanks_1"), ("building_train_1", "phoenix_5"), ("building_train_1", "singapore_8"),
      ("building_train_2", "fairbanks_2"), ("building_train_4", "phoenix_8")]
KEYS = {"ZB": "k07@", "ZS": "k07sup@;ROS=1,TR=1"}
W = {"comfort": ("normalized_productivity_regret", 0.6), "energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2)}
F = {}
for f in glob.glob(f"{D}/**/*.summary.json", recursive=True):
    s = json.load(open(f)); arm = next((a for a, k in KEYS.items() if s["policy_key"] == k), None)
    if arm and (s["building"], s["weather"]) in SC:
        assert (s["building"], s["weather"], arm) not in F, ("duplicate", f); F[(s["building"], s["weather"], arm)] = (f, s)
R = {}
for b, w in SC:
    for arm in KEYS:
        f, s = F[(b, w, arm)]; assert s["status"] == "OK" and s["steps_done"] == 8640 and s["illegal_actions"] == 0, (b, w, arm, s["status"])
        z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]
        R[(b, w, arm)] = {"L": float(np.abs(z["rewards"]).mean()), "comp": {k: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for k, (c, wt) in W.items()}, "s": s}
g = lambda sel: 100 * (1 - sum(R[(b, w, "ZS")]["L"] for b, w in sel) / sum(R[(b, w, "ZB")]["L"] for b, w in sel))
res = {"note": "official training scenes; corrected tracker (KRE-v2), as the paper's K07 official-scene stage; 8640 steps", "pooled_gain_pct": g(SC), "per_scene": {}}
for b, w in SC:
    zb = R[(b, w, "ZB")]; tot = sum(zb["comp"].values()); s3 = R[(b, w, "ZS")]["s"]; tr = s3.get("tr") or {}
    res["per_scene"][f"{b}/{w}"] = {"gain_pct": g([(b, w)]), "ZB_share_comfort": zb["comp"]["comfort"] / tot, "ZB_share_energy": zb["comp"]["energy"] / tot, "ZB_share_carbon": zb["comp"]["carbon"] / tot,
                                   "learned_starts": sum(1 for e in (s3.get("ros") or {}).get("log", []) if e.get("event") == "start" and e.get("why") == "need"),
                                   "pressure_override_steps": sum(tr.get("override_steps") or [0])}
tot = {k: sum(R[(b, w, "ZB")]["comp"][k] for b, w in SC) for k in W}; T = sum(tot.values()); res["pooled_ZB_shares"] = {k: v / T for k, v in tot.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"ZT Best: pooled gain of the layer on the official scenes {res['pooled_gain_pct']:+.3f} %; ZB shares {({k: round(v, 3) for k, v in res['pooled_ZB_shares'].items()})}")
for k, v in res["per_scene"].items():
    print(f"  {k:30s} gain {v['gain_pct']:+.2f} %  comfort share {v['ZB_share_comfort']:.3f}  starts {v['learned_starts']}  overrides {v['pressure_override_steps']}")
