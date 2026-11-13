"""Fidelity of the standby-band actuator (ACT=standby; no reward printed).  usage: python -I p223_pgxsup_standby_fidelity.py <old_out(v3)> <new_out(v3sup)> <out.json>
S1 the degenerate key (LMAX=LMAX_ID=1, empty window) == native pgx@static@ (all common arrays).  S2 the active key == native before its first knob change, and the first action
difference is at or after it.  S3 every logged start is the first window step where (opening - tod) <= tau*ln(max(D,D*)/D*) (logged ros arrays); at a start the standby band is set to
the comfort band and precondition_hours to 24; every other change is a restore of the native values at noon or after a freeze."""
import sys, json, glob, math, numpy as np
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[s["policy_key"]] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
O, Nw = ld(sys.argv[1]), ld(sys.argv[2]); s0, z0 = O["pgx@static@"]; res = {}
skip = lambda a: a.startswith("ros_")
sd, zd = Nw["pgx@sup@ROS=1,TR=0,ACT=standby,LMAX=1.0,LMAX_ID=1.0"]; bad = [a for a in sorted(set(z0.files) & set(zd.files)) if not skip(a) and not eq(z0[a], zd[a])]
res["S1_degenerate_eq_native"] = {"differing": bad, "switches": len(sd["switch_log"]), "pass": not bad and sd["status"] == "OK" and not sd["switch_log"]}
sa, za = Nw["pgx@sup@ROS=1,TR=0,ACT=standby"]; sw = sa["switch_log"]; first = min((e["step"] for e in sw), default=len(za["rewards"]))
pre = [a for a in sorted(set(z0.files) & set(za.files)) if not skip(a) and getattr(z0[a], "ndim", 0) >= 1 and len(z0[a]) >= first and not eq(z0[a][:first], za[a][:first])]
ad = np.nonzero(np.any(z0["actions_cont"] != za["actions_cont"], axis=1))[0]; fdiff = int(ad[0]) if len(ad) else None
res["S2_equal_until_first_change"] = {"first_change_step": first, "prefix_differing": pre, "first_action_diff": fdiff, "pass": not pre and (fdiff is None or fdiff >= first) and sa["status"] == "OK"}
R = sa["ros"]; P = R["params"]; ms = R["opening_hour"]; lo, hi = R["comfort_band_k"]; nat = R["native_values"]; tod, D, tau = za["ros_tod"], za["ros_D"], za["ros_tau"]; ok = True; chk = []
starts = [e for e in R["log"] if e["event"] == "start"]
for e in starts:
    s = int(e["step"]); lmax = float(e["lmax_h"]); w0 = ms - lmax
    win = [j for j in range(max(0, s - 60), s + 1) if w0 - 1e-9 <= tod[j] < ms - P["LMIN"] - 1e-9 and (s - j) < 60 and abs((tod[s] - tod[j]) - (s - j) / 12.0) < 1e-6]
    fire = [j for j in win if (ms - tod[j]) <= tau[j] * math.log(max(D[j], P["DSTAR"]) / P["DSTAR"])]
    setv = {x["knob"]: x["new"] for x in sw if x["step"] == s}
    good = bool(fire) and fire[0] == s and abs(setv.get("standby_minimum", -1) - lo) < 1e-9 and abs(setv.get("standby_maximum", -1) - hi) < 1e-9 and setv.get("precondition_hours") == 24.0
    ok &= good; chk.append({"step": s, "tod": round(float(tod[s]), 3), "lead_h": round(ms - float(tod[s]), 3), "first_fire_ok": bool(fire) and fire[0] == s, "set_ok": good})
others = [x for x in sw if x["step"] not in {int(e["step"]) for e in starts}]
rest_ok = all(abs(x["new"] - nat[x["knob"]]) < 1e-9 and (tod[x["step"]] >= 12.0 - 1e-9 or R["frozen"]) for x in others)
res["S3_decisions"] = {"starts": chk, "n_other_changes": len(others), "others_are_noon_or_freeze_restores": bool(rest_ok), "pass": bool(ok and rest_ok and starts)}
res["VERDICT"] = "PASS" if all(v["pass"] for k, v in res.items() if k.startswith("S")) else "FAIL"; json.dump(res, open(sys.argv[3], "w"), indent=1, default=str); print(res["VERDICT"], json.dumps(res, default=str)[:1200])
