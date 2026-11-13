"""Fidelity of the Penguin night-precooling runner v3pc; development scenes only; no reward value printed.
usage: python -I p239_pgpc_fidelity.py <v3sup ref dir (static fairbanks_3 864, static phoenix_2 2016)> <v3pc dir (static fairbanks_3 864, PC fairbanks_3 864, PC phoenix_2 2016)> <out.json>
F1 static key: v3pc == v3sup (all arrays).  F2 PC on a cold scene: every night 'cold', no pre-cooling step, all common arrays == native.  F3 PC on 52105 phoenix_2: arrays == native before the
first pre-cooling step; at every pre-cooling step pc_max == max(hi - 0.5, max zone T - 0.25, lo + 0.5) and outside air (oat_c of the previous step) <= mean zone T - 2; pre-cooling only
inside the weekday night window after a 'warm' gate with a positive learned need; standby maximum native at every other step.  F4 replay of every gate class from oat_c."""
import sys, json, glob, numpy as np
def ld(d):
    o = {}
    for f in glob.glob(f"{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); o[(s["building"], s["weather"], s["policy_key"], s["steps_done"])] = (s, np.load(f.replace(".summary.json", ".npz"), allow_pickle=True))
    return o
R, C = ld(sys.argv[1]), ld(sys.argv[2]); OUT = sys.argv[3]; PC = "pgx@sup@ROS=0,TR=0,PC=1"; res = {}
eq = lambda x, y: x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
def diff(z1, z2, n=None, skip=("pc_",)):
    bad = []
    for a in sorted(set(z1.files) & set(z2.files)):
        if any(a.startswith(p) for p in skip): continue
        x, y = z1[a], z2[a]
        if n is not None:
            if getattr(x, "ndim", 0) < 1 or x.shape[0] < n: continue
            x, y = x[:n], y[:n]
        if not eq(x, y): bad.append(a)
    return bad
sr, zr = R[("building_52103", "fairbanks_3", "pgx@static@", 864)]; sn, zn = C[("building_52103", "fairbanks_3", "pgx@static@", 864)]
d = diff(zr, zn); res["F1"] = {"differing": d, "pass": not d}
sc, zc = C[("building_52103", "fairbanks_3", PC, 864)]; g = [e["cls"] for e in sc["pc"]["log"] if e["event"] == "gate"]
d = diff(zr, zc); res["F2"] = {"gate_classes": g, "precool_steps": int(zc["pc_on"].sum()), "differing": d, "pass": bool(g) and all(x == "cold" for x in g) and int(zc["pc_on"].sum()) == 0 and not d}
s0, z0 = R[("building_52105", "phoenix_2", "pgx@static@", 2016)]; s1, z1 = C[("building_52105", "phoenix_2", PC, 2016)]; P = s1["pc"]["params"]; lo, hi = s1["pc"]["comfort_band_k"]; nat = s1["pc"]["native_standby_maximum"]
on = z1["pc_on"].astype(bool); first = int(np.argmax(on)) if on.any() else None; zt = z1["pc_zt"]; mx = z1["pc_max"]; oat = z1["oat_c"].astype(float)
pre = diff(z0, z1, n=first) if first is not None else diff(z0, z1)
ok_val = all(abs(mx[i] - max(hi - P["PC_BELOW_HI_K"], float(np.max(zt[i])) - P["PC_STEP_K"], lo + 0.5)) < 1e-9 for i in np.nonzero(on)[0])
ok_econ = all(i > 0 and (oat[i - 1] + 273.15) <= float(np.mean(zt[i])) - P["PC_ECON_K"] for i in np.nonzero(on)[0])
ok_native = all(mx[i] == nat for i in np.nonzero(~on)[0])
res["F3"] = {"first_precool_step": first, "precool_steps": int(on.sum()), "prefix_differing": pre, "value_rule_ok": ok_val, "economizer_rule_ok": ok_econ, "native_elsewhere_ok": ok_native,
             "need_final_k": s1["pc"]["need_final_k"], "gate_classes": [e["cls"] for e in s1["pc"]["log"] if e["event"] == "gate"], "frozen": s1["pc"]["frozen"],
             "pass": first is not None and not pre and ok_val and ok_econ and ok_native}
f4 = []
for s, z in ((sc, zc), (s1, z1)):
    o = z["oat_c"].astype(float)
    for e in s["pc"]["log"]:
        if e["event"] != "gate": continue
        w = o[max(0, e["step"] - 288):e["step"]]; w = w[np.isfinite(w)]; m = float(np.mean(w)) if len(w) else float("nan")
        c = "nan" if not np.isfinite(m) else ("warm" if m >= s["pc"]["params"]["PC_GATE_C"] else "cold"); f4.append(bool(abs(m - e["oat_mean_c"]) < 1e-9 and c == e["cls"]))
res["F4"] = {"gate_events": len(f4), "matching": sum(f4), "pass": bool(f4) and all(f4)}
res["ALL_PASS"] = all(res[k]["pass"] for k in ("F1", "F2", "F3", "F4")); json.dump(res, open(OUT, "w"), indent=1)
for k in ("F1", "F2", "F3", "F4"): print(k, "PASS" if res[k]["pass"] else "FAIL", {kk: v for kk, v in res[k].items() if kk != "pass"})
print("ALL_PASS", res["ALL_PASS"])
