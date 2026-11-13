"""ROS fidelity (no reward printed).  usage: python -I p214_ros_fidelity.py <old_out(v9s)> <new_out(v9ros)> <out.json>
F1 k07@G30 by v9ros == by v9s (all common npz arrays, NaN-aware).  F2 k07ros with an empty window (LMAX=2) == G30 (all common arrays).  F3 k07ros default == G30 up to the first step where
base_prestart_c != 2; pi_active turns on at the logged start step; logged ros_D == D recomputed from zone_t_k (band 294-297 K) within 1e-4 K.  F4 the first window step where
(6 - tod) <= tau*ln(max(D,0.3)/0.3) equals the logged start step; base_prestart changes only at logged starts and noon resets."""
import sys, json, glob, math, numpy as np
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[s["policy_key"]] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
O, Nw = ld(sys.argv[1]), ld(sys.argv[2]); G = [k for k in O if k.startswith("k07@")][0]; R = "k07ros@" + G[4:]; R2 = R + ";LMAX=2,LMAX_ID=2"; res = {}
(s0, z0) = O[G]; skip = {"ros_D", "ros_tau"}
for nm, k in (("F1_G30_new_vs_old", G), ("F2_ros_empty_window_vs_G30", R2)):
    s1, z1 = Nw[k]; com = sorted((set(z0.files) & set(z1.files)) - skip); bad = [a for a in com if not eq(z0[a], z1[a])]
    res[nm] = {"arrays": len(com), "differing": bad, "status_ok": s1["status"] == "OK", "pass": not bad and s1["status"] == "OK"}
s2, z2 = Nw[R]; bp = z2["base_prestart_c"]; first = int(np.argmax(bp != 2)) if (bp != 2).any() else len(bp)
pre = {a: bool(eq(z0[a][:first], z2[a][:first])) for a in sorted(set(z0.files) & set(z2.files) - skip) if getattr(z0[a], "ndim", 0) >= 1 and z0[a].shape[0] == len(bp)}
log = s2["ros"]["log"]; starts = [e for e in log if e["event"] == "start"]; tk = z2["zone_t_k"].astype(float); Drec = np.mean(np.maximum(np.maximum(294.0 - tk, 0.0), tk - 297.0), axis=1)
pi = z2["pi_active_c"]; pi_on = [int(e["step"]) for e in starts]; pi_ok = all(pi[s] == 1.0 and pi[s - 1] == 0.0 for s in pi_on)
res["F3_equal_until_first_change"] = {"first_change_step": first, "prefix_equal_all": all(pre.values()), "n_arrays": len(pre), "starts": [(e["step"], round(e["tod_h"], 3), round(e["D_k"], 3), round(e["tau_h"], 3), e["prestart_set"], e["why"]) for e in starts],
                                      "pi_on_at_start_steps": pi_ok, "max_abs_D_log_vs_recomputed": float(np.nanmax(np.abs(z2["ros_D"] - Drec))), "n_updates": sum(1 for e in log if e["event"] == "update"), "frozen": s2["ros"]["frozen"]}
ts = [str(x) for x in z2["ts"]]; tod = np.array([int(t[11:13]) + int(t[14:16]) / 60.0 for t in ts])
if starts:                                                     # the tracker timestamps may be those after the step: align them on the logged observation time of the first start
    _off = (tod[int(starts[0]["step"])] - float(starts[0]["tod_h"])) % 24.0; _off = _off - 24.0 if _off > 12 else _off; tod = (tod - _off) % 24.0
    res["F4_time_offset_h"] = float(_off)
dec_ok = True; checked = 0; fz = [int(e["step"]) + 1 for e in log if e["event"] == "frozen"]
for e in starts:
    s = int(e["step"]); lo = 6.0 - float(e["lmax_h"])
    w = [j for j in range(max(0, s - 40), s + 1) if lo - 1e-9 <= tod[j] < 4.0 - 1e-9 and (s - j) < 30]
    firsts = [j for j in w if (6.0 - tod[j]) <= z2["ros_tau"][j] * math.log(max(z2["ros_D"][j], 0.3) / 0.3)]
    if e["why"] == "need": dec_ok &= bool(firsts) and firsts[0] == s and (bool(e["capped"]) == (abs(tod[s] - lo) < 1.0 / 12 and e["need_h"] > 6.0 - tod[s])); checked += 1
chg = np.nonzero(np.diff(bp) != 0)[0] + 1; allowed = set(pi_on) | {j for j in chg if tod[j] >= 12.0 - 1e-9} | set(fz)
res["F4_decisions"] = {"need_starts_checked": checked, "first_crossing_equals_logged": dec_ok, "base_prestart_changes": [int(j) for j in chg], "only_at_starts_or_noon": set(int(j) for j in chg) <= allowed,
                       "note": "tod from the npz timestamps; a start is the first window step where the rule fires"}
ok = res["F1_G30_new_vs_old"]["pass"] and res["F2_ros_empty_window_vs_G30"]["pass"] and res["F3_equal_until_first_change"]["prefix_equal_all"] and pi_ok and res["F3_equal_until_first_change"]["max_abs_D_log_vs_recomputed"] < 1e-4 and dec_ok and res["F4_decisions"]["only_at_starts_or_noon"]
res["VERDICT"] = "PASS" if ok else "FAIL"; json.dump(res, open(sys.argv[3], "w"), indent=1, default=str); print(res["VERDICT"], json.dumps(res, default=str)[:1500])
