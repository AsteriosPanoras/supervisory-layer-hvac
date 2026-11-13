"""Fidelity of runner v9supe (G36 unoccupied cooling setup); development scenes 52010 phoenix_7 and 52103 fairbanks_3; no reward value printed.
usage: python -I p246_uns_fidelity.py <v9supd dir (SUP 864 phoenix_7)> <v9supe dir> <out.json>
F1 existing SUP key: v9supe == v9supd (all arrays, supervisor log).  F2 setup only vs G30 (both scenes): identical before the first differing action; at that step K07 is outside
its PI hours and some zone is above the original eco cooling limit 298 K (the base would have cooled), or the runs are identical.  F3 setup + ROS vs setup only: identical before the
first ROS morning extension, and ROS extensions are logged."""
import sys, json, glob, numpy as np
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[(s["building"], s["weather"], s["policy_key"], s["steps_done"])] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
A, B = ld(sys.argv[1]), ld(sys.argv[2]); OUT = sys.argv[3]
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
SUP = f"k07sup@{G};ROS=1,TR=1"; G30 = f"k07@{G}"; U2 = f"k07sup@{G};ROS=0,TR=0,UNS_COOL_K=305.4"; U3 = f"k07sup@{G};ROS=1,TR=0,UNS_COOL_K=305.4"
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def diff(z1, z2, n=None, skip=("ros_",)):
    bad = []
    for a in sorted(set(z1.files) & set(z2.files)):
        if any(a.startswith(p) for p in skip): continue
        x, y = z1[a], z2[a]
        if n is not None:
            if getattr(x, "ndim", 0) < 1 or x.shape[0] < n: continue
            x, y = x[:n], y[:n]
        if not eq(x, y): bad.append(a)
    return bad
res = {}
s1, z1 = A[("building_52010", "phoenix_7", SUP, 864)]; t1, y1 = B[("building_52010", "phoenix_7", SUP, 864)]
d = diff(z1, y1, skip=()); res["F1"] = {"differing": d, "log_equal": s1["ros"]["log"] == t1["ros"]["log"], "pass": not d and s1["ros"]["log"] == t1["ros"]["log"]}
f2 = {}
for (b, w, n) in (("building_52010", "phoenix_7", 2016), ("building_52103", "fairbanks_3", 864)):
    sg, zg = B[(b, w, G30, n)]; su, zu = B[(b, w, U2, n)]
    ac_g, ac_u = zg["actions_cont"], zu["actions_cont"]; dif = np.nonzero(np.any(ac_g != ac_u, axis=1))[0]
    if len(dif) == 0: f2[w] = {"identical": True, "pass": not diff(zg, zu)}; continue
    k = int(dif[0]); pre = diff(zg, zu, n=k); pi = bool(zg["pi_active_c"][k]); zt = zg["zone_t_k"][k].astype(float)
    f2[w] = {"first_diff_step": k, "prefix_differing": pre, "pi_active_at_first_diff": pi, "max_zone_k_at_first_diff": float(np.nanmax(zt)), "pass": not pre and not pi and float(np.nanmax(zt)) > 298.0 - 1e-6}
res["F2"] = {**f2, "pass": all(v["pass"] for v in f2.values())}
s2, z2 = B[("building_52010", "phoenix_7", U2, 2016)]; s3, z3 = B[("building_52010", "phoenix_7", U3, 2016)]
starts = [e["step"] for e in s3["ros"]["log"] if e["event"] == "start" and e.get("why") == "need"]; k = starts[0] if starts else None
pre = diff(z2, z3, n=k) if k is not None else diff(z2, z3)
res["F3"] = {"ros_extensions": len(starts), "first_extension_step": k, "prefix_differing": pre, "pass": not pre}
res["ALL_PASS"] = all(res[x]["pass"] for x in ("F1", "F2", "F3")); json.dump(res, open(OUT, "w"), indent=1)
for x in ("F1", "F2", "F3"): print(x, "PASS" if res[x]["pass"] else "FAIL", {kk: v for kk, v in res[x].items() if kk != "pass"})
print("ALL_PASS", res["ALL_PASS"])
