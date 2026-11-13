"""Mean zone vs. worst zone after the unoccupied setup (post hoc, descriptive; frozen, already-read runs).  usage: python -I p260_mean_zone_check.py <out.json>
Official training scenes of the unoccupied-setup stage, K07-G30 (G30) vs. K07-G30 + G36 unoccupied cooling setup (U) and U + learned start (UR).
The start rule acts on the MEAN deviation of the zones from the controller's comfort band; the band is not assumed: it is verified first, by recomputing the
deficit logged by the rule at every logged decision of the UR runs from the saved zone temperatures (must match to 1e-6).
For weekday steps 06:00-09:00 (first occupied hours) it reports, per scene and arm: mean over zones of the deviation (what the rule sees), mean over steps of the largest
zone deviation, and the share of zone-steps outside the band; plus the deficit at 04, 05 and 06 h (weekday means)."""
import sys, os, glob, json, datetime as dt, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
KEYS = {f"k07@{G}": "G30", f"k07sup@{G};ROS=0,TR=0,UNS_COOL_K=305.4": "U", f"k07sup@{G};ROS=1,TR=0,UNS_COOL_K=305.4": "UR"}
LO, HI = 294.0, 297.0
Z, S = {}, {}
for f in glob.glob(f"{N}/k07uns/laptop/work_off/out/**/*.summary.json", recursive=True):
    s = json.load(open(f)); p = KEYS.get(s["policy_key"])
    if p and s["status"] == "OK" and s["steps_done"] == 8640: Z[(s["building"], s["weather"], p)] = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); S[(s["building"], s["weather"], p)] = s
dev_of = lambda T: np.maximum(np.maximum(LO - T, 0), T - HI)
# 1) verify the band against every deficit the rule logged
nchk, worst = 0, 0.0
for (b, w, p), s in S.items():
    if p != "UR": continue
    T = Z[(b, w, p)]["zone_t_k"].astype(float)
    for e in (s.get("ros") or {}).get("log", []):
        if "D_k" in e and e["D_k"] == e["D_k"]:
            worst = max(worst, abs(float(np.nanmean(dev_of(T[e["step"]]))) - e["D_k"])); nchk += 1
assert nchk > 0 and worst < 1e-6, (nchk, worst)
res = {"note": "post hoc descriptive; frozen already-read runs", "band_K": [LO, HI], "band_verified_on_logged_deficits": nchk, "max_abs_error": worst, "scenes": {}}
for b, w in sorted({(b, w) for b, w, p in Z}):
    if not all((b, w, p) in Z for p in ("G30", "U", "UR")): continue
    tt = [dt.datetime.fromisoformat(str(x)) for x in Z[(b, w, "G30")]["ts"]]; hr = np.array([t.hour + t.minute / 60 for t in tt]); dow = np.array([t.weekday() for t in tt])
    mor = (dow <= 4) & (hr >= 6) & (hr < 9); r = {}
    for p in ("G30", "U", "UR"):
        D = dev_of(Z[(b, w, p)]["zone_t_k"].astype(float))
        at = {h: float(np.nanmean(np.nanmean(D, axis=1)[(dow <= 4) & (np.abs(hr - h) < 1e-6)])) for h in (4, 5, 6)}
        r[p] = {"weekday_06_09_mean_zone_deviation_K": float(np.nanmean(D[mor])), "weekday_06_09_largest_zone_deviation_mean_K": float(np.nanmean(np.nanmax(D[mor], axis=1))),
                "weekday_06_09_share_zone_steps_outside_pct": float(100 * np.mean(D[mor] > 0)), "weekday_mean_deficit_at_04_05_06_K": at}
    res["scenes"][f"{b}|{w}"] = r
json.dump(res, open(OUT, "w"), indent=1)
print("band verified on", nchk, "logged deficits, max error", worst)
for k, r in res["scenes"].items():
    print(k, {p: (round(v["weekday_06_09_mean_zone_deviation_K"], 3), round(v["weekday_06_09_largest_zone_deviation_mean_K"], 2), round(v["weekday_06_09_share_zone_steps_outside_pct"], 1)) for p, v in r.items()})
