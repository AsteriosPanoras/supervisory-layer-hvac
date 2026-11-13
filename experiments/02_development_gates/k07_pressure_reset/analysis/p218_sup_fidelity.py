"""Combined-supervisor fidelity (runner v9sup; no reward printed).  usage: python -I p218_sup_fidelity.py <old_out(v9s)> <new_out(v9sup)> <ros_ref_out(v9ros fidelity)> <out.json>
F1 k07@G30 by v9sup == by v9s.  F2 k07tr;SPMAX=200 == G30 and requests occur.  F3 k07tr == G30 before its first override; at that step only the overridden AHUs' pressure columns
differ, with value sp/1000 - 1.  F4 offline replay of the trim & respond rule from the logged zone temperatures, PI outputs and PI state reproduces tr_req and tr_sp_pa.
F5 k07ros by v9sup == by v9ros.  F6 k07sup (ROS=1, TR=1) == G30 before its first change, and both supervisors logged."""
import sys, json, glob, numpy as np
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[(s["building"], s["policy_key"])] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
O, Nw, RR = ld(sys.argv[1]), ld(sys.argv[2]), ld(sys.argv[3]); res = {}
G = [k for (b, k) in O if k.startswith("k07@")][0]; OV = G[4:]; B = "building_52010"; s0, z0 = O[(B, G)]
skip = lambda a: a.startswith("tr_") or a.startswith("ros_")
def same(z1, n=None):
    com = sorted(a for a in set(z0.files) & set(z1.files) if not skip(a))
    if n is None: return [a for a in com if not eq(z0[a], z1[a])]
    return [a for a in com if getattr(z0[a], "ndim", 0) >= 1 and z0[a].shape[0] >= n and z1[a].shape[0] >= n and not eq(z0[a][:n], z1[a][:n])]
s1, z1 = Nw[(B, G)]; res["F1"] = {"differing": same(z1), "pass": not same(z1) and s1["status"] == "OK"}
s2, z2 = Nw[(B, f"k07tr@{OV};SPMAX=200")]; res["F2"] = {"differing": same(z2), "requests_seen": int((z2["tr_req"] > 0).sum()), "pass": not same(z2) and int((z2["tr_req"] > 0).sum()) > 0 and s2["status"] == "OK"}
s3, z3 = Nw[(B, f"k07tr@{OV}")]; ov = z3["tr_ovr"]; first = int(np.argmax(ov.any(axis=1))) if ov.any() else len(ov); amap = s3["tr"]["ahu_map"]
pre = same(z3, first); col_ok = True
if first < len(ov):
    d = np.nonzero(z3["actions_cont"][first] != z0["actions_cont"][first])[0].tolist(); exp_cols = sorted(amap[a]["pressure_idx"] for a in np.nonzero(ov[first])[0])
    vals_ok = all(abs(float(z3["actions_cont"][first][amap[a]["pressure_idx"]]) - (float(z3["tr_sp_pa"][first][a]) / 1000 - 1)) < 1e-6 for a in np.nonzero(ov[first])[0])
    col_ok = sorted(d) == exp_cols and vals_ok
res["F3"] = {"first_override_step": first, "prefix_differing": pre, "only_pressure_columns_at_first": col_ok, "pass": not pre and col_ok and first < len(ov)}
# F4 replay
P = s3["tr"]["params"]; tk, zu, pi = z3["zone_t_k"].astype(float), z3["zone_u"].astype(float), z3["pi_active_c"]; HI = 297.0
zones = []; [zones.extend([(a, v) for v in m["vavs"]]) for a, m in enumerate(amap)]; NA = len(amap)
sp = [P["SPMIN"]] * NA; req_on = {}; m = {}; prev = False; R_ok = SP_ok = True; ovr_ok = True; run_at_max = {}; marked = {}
for i in range(len(pi)):
    p_ = bool(pi[i])
    if p_ and not prev: sp = [P["SPMIN"]] * NA
    for a in range(NA):
        R = 0.0; rz = {}
        for j, (aa, v) in enumerate(zones):
            if aa != a: continue
            T, u = tk[i, j], zu[i, j]
            if not (np.isfinite(T) and np.isfinite(u)) or not p_: req_on[v] = False; continue
            op = u >= P["U_REQ"] or (req_on.get(v, False) and u >= P["U_REL"]); on = op and T > HI; req_on[v] = on
            r = 0 if not on else (2 if T > HI + P["HI_EXTRA_K"] else 1); rz[v] = r; R += m.get(v, 1.0) * r
        if p_:
            sp[a] = min(sp[a] + min(P["RES"] * (R - P["I"]), P["RESMAX"]), P["SPMAX"]) if R > P["I"] else max(sp[a] - P["TRIM"], P["SPMIN"])
            for v, r in rz.items():
                run_at_max[v] = run_at_max.get(v, 0) + 1 if (r > 0 and sp[a] >= P["SPMAX"]) else 0
                if run_at_max[v] >= P["UNRESP_STEPS"]:
                    marked.setdefault(v, set()).add(i // 288); run_at_max[v] = 0
                    if len(marked[v]) >= P["UNRESP_DAYS"]: m[v] = 0.0
        R_ok &= int(R) == int(z3["tr_req"][i][a]); SP_ok &= abs(sp[a] - float(z3["tr_sp_pa"][i][a])) < 1e-3
        kp = float(z3["tr_k07_p"][i][a]); exp_ovr = p_ and np.isfinite(kp) and sp[a] > max(kp, 200.0) + 0.5
        ovr_ok &= (not bool(z3["tr_ovr"][i][a])) or exp_ovr                      # an override only where the rule allows one (the AHU run flag is not logged, so not the converse)
    prev = p_
res["F4"] = {"tr_req_reproduced": bool(R_ok), "tr_sp_reproduced": bool(SP_ok), "overrides_only_where_allowed": bool(ovr_ok), "no_override_outside_PI": bool(not (z3["tr_ovr"].any(axis=1) & (pi == 0)).any()), "pass": bool(R_ok and SP_ok and ovr_ok)}
kr = [k for (b, k) in RR if k.startswith("k07ros@") and "LMAX=2" not in k][0]; sr, zr = RR[("building_52103", kr)]; s5, z5 = Nw[("building_52103", kr)]
d5 = [a for a in sorted(set(zr.files) & set(z5.files)) if not eq(zr[a], z5[a])]; res["F5"] = {"differing": d5, "pass": not d5 and s5["status"] == "OK"}
s6, z6 = Nw[(B, f"k07sup@{OV};ROS=1,TR=1")]; c1 = int(np.argmax(z6["tr_ovr"].any(axis=1))) if z6["tr_ovr"].any() else len(z6["tr_ovr"]); bp = z6["base_prestart_c"]; c2 = int(np.argmax(bp != 2)) if (bp != 2).any() else len(bp)
fc = min(c1, c2); res["F6"] = {"first_change": fc, "prefix_differing": same(z6, fc), "ros_logged": s6.get("ros") is not None, "tr_logged": s6.get("tr") is not None, "pass": not same(z6, fc) and s6.get("ros") is not None and s6.get("tr") is not None and s6["status"] == "OK"}
res["VERDICT"] = "PASS" if all(res[k]["pass"] for k in ("F1", "F2", "F3", "F4", "F5", "F6")) else "FAIL"
json.dump(res, open(sys.argv[4], "w"), indent=1, default=str); print(res["VERDICT"], json.dumps(res, default=str)[:1600])
