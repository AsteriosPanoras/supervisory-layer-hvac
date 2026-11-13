"""p222 (INTERNAL): where does NATIVE Penguin ('pgx@static@', 30 days) lose on the new architectures?  DEVELOPMENT scenes only: the (architecture, weather) cells with
split == "dev" in k07x/K07X_V2_COHORT.json; files are located by constructing each dev cell's own file name, so no held-out file is listed or read.
Per scene, relative numbers only (shares of that scene's own loss L = |reward| = 0.2 E + 0.2 C - 0.6 P per step; the decomposition is asserted):
the comfort (productivity-regret) share of L, the weekday hour of the comfort peak, the weekday 5-8 h comfort share of L (gate P0 of k07fb_design section 5: > 0.5 % of L),
the late-recovery comfort (weekday 6-12 h comfort above the scene's 12-17 h hourly level, % of L), and the energy + carbon share of L in Penguin's own pre-conditioning hour (weekday 5-6 h)
and in the night standby (weekday 0-5 h).  Zone temperatures were not logged by the Penguin runner, so cold vs hot comfort cannot be separated here.
usage: python -I p222_pgx_native_dev_anatomy.py <N10S_design dir> <out.json>"""
import datetime as dt, glob, json, sys
import numpy as np

N, OUT = sys.argv[1:3]
cells = sorted((c["architecture"], c["weather"], c["group"]) for c in json.load(open(f"{N}/k07x/K07X_V2_COHORT.json"))["cells"] if c["split"] == "dev")
rows = []
for b, w, grp in cells:
    fs = sorted(glob.glob(f"{N}/results_pgx/*/out/**/{b}_{w}_pgx__static___8640.npz", recursive=True))
    if not fs: rows.append({"architecture": b, "weather": w, "group": grp, "status": "no native Penguin-X dev run"}); continue
    z = np.load(fs[0], allow_pickle=False); ck = [str(x) for x in z["comp_keys"]]; C = z["comps"]; rew = z["rewards"]
    if len(rew) != 8640: rows.append({"architecture": b, "weather": w, "group": grp, "status": f"{len(rew)} steps"}); continue
    E = 0.2 * C[:, ck.index("normalized_energy_cost")]; CB = 0.2 * C[:, ck.index("normalized_carbon_emission")]; P = -0.6 * C[:, ck.index("normalized_productivity_regret")]
    if not np.allclose(E + CB + P, -rew, atol=1e-9): rows.append({"architecture": b, "weather": w, "group": grp, "status": "reward decomposition mismatch"}); continue
    L = float(np.abs(rew).sum())
    ts = [dt.datetime.fromisoformat(t) - dt.timedelta(minutes=5) for t in z["ts"]]; hr = np.array([t.hour for t in ts]); wk = np.array([t.weekday() < 5 for t in ts])
    hp = np.array([P[wk & (hr == h)].sum() for h in range(24)]); pct = lambda x: round(100.0 * float(x) / L, 2)
    aft = hp[12:17].mean(); late = np.maximum(hp[6:12] - aft, 0.0).sum()
    rows.append({"architecture": b, "weather": w, "group": grp, "status": "OK", "file": fs[0].split("/results_pgx/")[1],
                 "comfort_share_pct": pct(P.sum()), "energy_share_pct": pct(E.sum()), "carbon_share_pct": pct(CB.sum()),
                 "weekday_comfort_peak_hour": int(np.argmax(hp)) if hp.max() > 0 else None, "wd_comfort_5_8h_pct": pct(hp[5:8].sum()), "wd_comfort_8_12h_pct": pct(hp[8:12].sum()),
                 "wd_comfort_12_19h_pct": pct(hp[12:19].sum()), "late_recovery_comfort_pct": pct(late),
                 "wd_precondition_hour_5_6_energy_carbon_pct": pct((E + CB)[wk & (hr == 5)].sum()), "wd_standby_0_5_energy_carbon_pct": pct((E + CB)[wk & (hr < 5)].sum()),
                 "weekend_pct": pct(np.abs(rew)[~wk].sum()), "oat_mean_c": round(float(np.nanmean(z["oat_c"])), 1)})
ok = [r for r in rows if r["status"] == "OK"]
gate = {"rule": "k07fb_design section 5 gate P0: weekday 5-8 h comfort regret > 0.5 % of L in >= 3 dev scenes", "n_scenes_with_data": len(ok),
        "n_scenes_above": sum(r["wd_comfort_5_8h_pct"] > 0.5 for r in ok), "passes": sum(r["wd_comfort_5_8h_pct"] > 0.5 for r in ok) >= 3}
json.dump({"rows": rows, "gate_P0": gate}, open(OUT, "w"), indent=1)
print(f"{'architecture':15s} {'weather':12s} grp  comfort%L peak_h  5-8h%L 8-12h%L 12-19h%L late%L  pre5-6h_EC%L night0-5_EC%L weekend%L  oat")
for r in rows:
    if r["status"] != "OK": print(f"{r['architecture']:15s} {r['weather']:12s} {r['group'][:4]}  {r['status']}"); continue
    print(f"{r['architecture']:15s} {r['weather']:12s} {r['group'][:4]}  {r['comfort_share_pct']:8.2f} {str(r['weekday_comfort_peak_hour']):>6s} {r['wd_comfort_5_8h_pct']:7.2f} {r['wd_comfort_8_12h_pct']:7.2f} "
          f"{r['wd_comfort_12_19h_pct']:8.2f} {r['late_recovery_comfort_pct']:6.2f} {r['wd_precondition_hour_5_6_energy_carbon_pct']:12.2f} {r['wd_standby_0_5_energy_carbon_pct']:13.2f} {r['weekend_pct']:9.2f} {r['oat_mean_c']:5.1f}")
print(json.dumps(gate))
