"""Fidelity of runner v9supe_zt (maintenance test PREREG_ZTSUP_v1) on development scenes 52010 phoenix_7 and 52103 fairbanks_3, 864 steps; no reward value is printed.
usage: python -I p250_zt_fidelity.py <old runner v9_zt work dir> <new runner v9supe_zt work dir> <out.json>
F1: zt-best 'k07@' under v9supe_zt equals the earlier ZT runner v9_zt (every array present in both runs, bitwise).
F2: 'k07sup@;ROS=0,TR=0' (supervisor off) equals 'k07@' under v9supe_zt (all arrays).
F3: 'k07sup@;ROS=1,TR=1' runs (status OK, no illegal action) and is identical to 'k07@' up to its first differing action; at that step the supervisor has acted (a ROS
start at or before it, or a TR override at it), or the runs are identical.  Activity counts of both supervisor parts are reported for both scenes."""
import sys, json, glob, numpy as np
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[(s["building"], s["weather"], s["policy_key"], s["steps_done"])] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
A, B = ld(sys.argv[1]), ld(sys.argv[2]); OUT = sys.argv[3]
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def diff(z1, z2, n=None, skip=()):
    bad = []
    for a in sorted(set(z1.files) & set(z2.files)):
        if any(a.startswith(p) for p in skip): continue
        x, y = z1[a], z2[a]
        if n is not None:
            if getattr(x, "ndim", 0) < 1 or x.shape[0] < n: continue
            x, y = x[:n], y[:n]
        if not eq(x, y): bad.append(a)
    return bad
res = {}; P = ("building_52010", "phoenix_7")
so, zo = A[(*P, "k07@", 864)]; sn, zn = B[(*P, "k07@", 864)]
d1 = diff(zo, zn); res["F1"] = {"common_arrays": sorted(set(zo.files) & set(zn.files)), "differing": d1, "status": [so["status"], sn["status"]], "pass": not d1 and so["status"] == sn["status"] == "OK"}
s0, z0 = B[(*P, "k07sup@;ROS=0,TR=0", 864)]; d2 = diff(zn, z0); res["F2"] = {"differing": d2, "pass": not d2 and s0["status"] == "OK"}
f3 = {}
for sc in (P, ("building_52103", "fairbanks_3")):
    s, z = B[(*sc, "k07sup@;ROS=1,TR=1", 864)]; ros = (s.get("ros") or {}).get("log", []); tr = (s.get("tr") or {}); trl = tr.get("log", [])
    act = {"status": s["status"], "illegal_actions": s.get("illegal_actions", 0), "ros_events": {e["event"] + ":" + str(e.get("why")): sum(1 for x in ros if x["event"] == e["event"] and x.get("why") == e.get("why")) for e in ros},
           "tr_override_steps": tr.get("override_steps"), "tr_log_events": len(trl)}
    if sc == P:
        dif = np.nonzero(np.any(zn["actions_cont"] != z["actions_cont"], axis=1))[0]
        if len(dif) == 0: act["identical_to_zt"] = True; ok = True
        else:
            k = int(dif[0]); pre = diff(zn, z, n=k, skip=("ros_",)); starts = [e["step"] for e in ros if e["event"] == "start"]; ovr = [e["step"] for e in trl if e.get("event") == "override_on"]
            act.update({"first_diff_step": k, "prefix_differing": pre, "ros_start_at_or_before": any(x <= k for x in starts), "tr_override_at": k in ovr}); ok = not pre and (act["ros_start_at_or_before"] or act["tr_override_at"])
        act["pass"] = ok and s["status"] == "OK" and not s.get("illegal_actions", 0)
    else: act["pass"] = s["status"] == "OK" and not s.get("illegal_actions", 0)
    f3[sc[1]] = act
res["F3"] = {**f3, "pass": all(v["pass"] for v in f3.values())}
res["ALL_PASS"] = all(res[x]["pass"] for x in ("F1", "F2", "F3")); json.dump(res, open(OUT, "w"), indent=1)
for x in ("F1", "F2", "F3"): print(x, "PASS" if res[x]["pass"] else "FAIL", {k: v for k, v in res[x].items() if k not in ("pass", "common_arrays")})
print("ALL_PASS", res["ALL_PASS"])
