"""p283: where each controller loses on the official training scenes (the closest available proxy of the hidden leaderboard scenes), by
term x occupancy x hour band, to choose tomorrow's mechanism by the decomposition rule of Section 5.3 / future work.  Read-only on stored runs; nothing is run.
Controllers and runs (plain competition kit or research runner as stored):
  K07-G30  k07uns official stage (6 scenes, 8640 steps, key k07@<G30>);
  ResPPO   the public Residual PPO controller alone, official_adapter/dry_ax/diag (6 scenes, 8640 steps, key k07@);
  Penguin  deep-native: official_adapter/dry_pg/v0 (atlanta_5) and the learned-precooling references dry_pg/pc (2880 steps), used as
           Penguin itself only where the precooling never acted (checked: its pc_on log must be all zero).
Shares are of the weighted loss (0.6 comfort, 0.2 energy, 0.2 carbon; mean |weighted component|), pooled over a controller's scenes.
Hour bands: night 00-06, morning 06-09, day 09-17, evening 17-24.  The energy term is split into electricity and gas by the raw costs.
usage: python -I p283_official_term_hour.py <out.json>"""
import sys, os, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
G30 = "k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
def runs():
    R = {"K07-G30": [], "ResPPO": [], "Penguin": []}
    for f in glob.glob(f"{N}/k07uns/laptop/work_off/out/**/*.summary.json", recursive=True):
        s = json.load(open(f))
        if s["policy_key"] == G30 and s["status"] == "OK": R["K07-G30"].append(f)
    for f in glob.glob(f"{N}/official_adapter/dry_ax/diag/*k07___8640.summary.json"):
        if json.load(open(f))["status"] == "OK": R["ResPPO"].append(f)
    v0 = glob.glob(f"{N}/official_adapter/dry_pg/v0/ref_*/*.summary.json"); R["Penguin"] += v0
    v0_scenes = {json.load(open(f))["weather"] for f in v0}
    for f in glob.glob(f"{N}/official_adapter/dry_pg/pc/ref_*/*.summary.json"):
        s = json.load(open(f)); z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)
        if s["weather"] in v0_scenes: continue
        pc = z["pc_on"] if "pc_on" in z.files else None
        assert pc is not None and not np.asarray(pc).astype(bool).any(), f"precooling acted in {s['building']}/{s['weather']}: not Penguin itself"
        R["Penguin"].append(f)
    return R
BANDS = [("night 00-06", 0, 6), ("morning 06-09", 6, 9), ("day 09-17", 9, 17), ("evening 17-24", 17, 24)]
res = {"note": "read-only; official training scenes; shares of the weighted loss pooled over each controller's scenes", "controllers": {}}
for ctrl, files in runs().items():
    acc = {}; tot = 0.0; elec = gas = 0.0; scenes = []
    for f in files:
        s = json.load(open(f)); z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); ck = [str(x) for x in z["comp_keys"]]; scenes.append(f"{s['building']}/{s['weather']}")
        hours = np.array([int(str(t)[11:13]) for t in z["ts"]]); occ = z["occupancy"].astype(float) > 0
        w = {"comfort": np.abs(0.6 * z["comps"][:, ck.index("normalized_productivity_regret")]), "energy": np.abs(0.2 * z["comps"][:, ck.index("normalized_energy_cost")]),
             "carbon": np.abs(0.2 * z["comps"][:, ck.index("normalized_carbon_emission")])}
        e_el, e_gas = np.abs(z["comps"][:, ck.index("electricity_energy_cost")]), np.abs(z["comps"][:, ck.index("natural_gas_energy_cost")])
        share_el = e_el / np.where(e_el + e_gas > 0, e_el + e_gas, 1.0); elec += float((w["energy"] * share_el).sum()); gas += float((w["energy"] * (1 - share_el)).sum())
        for term, v in w.items():
            for b, h0, h1 in BANDS:
                m = (hours >= h0) & (hours < h1)
                for o, lab in ((True, "occupied"), (False, "unoccupied")):
                    k = (term, lab, b); acc[k] = acc.get(k, 0.0) + float(v[m & (occ == o)].sum())
            tot += float(v.sum())
    cells = sorted(((v / tot, k) for k, v in acc.items()), reverse=True)
    res["controllers"][ctrl] = {"scenes": scenes, "n": len(scenes),
                                "by_term": {t: sum(v for k, v in acc.items() if k[0] == t) / tot for t in ("comfort", "energy", "carbon")},
                                "by_occupancy": {o: sum(v for k, v in acc.items() if k[1] == o) / tot for o in ("occupied", "unoccupied")},
                                "energy_split": {"electricity": elec / (elec + gas), "gas": gas / (elec + gas)},
                                "top_cells": [{"term": k[0], "occupancy": k[1], "hours": k[2], "share": s_} for s_, k in cells[:8]]}
json.dump(res, open(OUT, "w"), indent=1)
for ctrl, v in res["controllers"].items():
    print(f"== {ctrl} ({v['n']} scenes): terms {({k: round(x, 3) for k, x in v['by_term'].items()})}, occupancy {({k: round(x, 3) for k, x in v['by_occupancy'].items()})}, energy split {({k: round(x, 2) for k, x in v['energy_split'].items()})}")
    for c in v["top_cells"][:6]: print(f"     {c['share']:.3f}  {c['term']:8s} {c['occupancy']:10s} {c['hours']}")
