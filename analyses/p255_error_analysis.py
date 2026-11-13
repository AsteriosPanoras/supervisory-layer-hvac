"""Error analysis of the learning layer on the blind set (post hoc, descriptive; frozen, already-read runs; no verdict changes).  usage: python -I p255_error_analysis.py <out_dir>
Paired comparison layer (SUP) vs G30 on the same 36 blind episodes (identical weather and simulator).  'Gain' over a subset of steps = 100 (1 - sum loss_SUP / sum loss_G30) over those steps,
pooled over the 10 composite architectures unless stated; OOD reported separately where useful.
T  Time: by week of the episode, by day type (Monday after the weekend, Tue-Fri, weekend), by hour of day with the weighted components, first five weekday mornings vs later.
X  Weather extremes and variability, per calendar day: daily mean outdoor temperature bins; the coldest/hottest 10 % of days; day-to-day change of the daily mean
   (cold snaps / warm surges); within-day range.
D  Decisions under weather that was unknown at decision time: every learned-start morning ('need' start; capped starts counted apart); predicted vs actual deficit at opening;
   weather surprise = mean outdoor temperature from decision to opening minus the value at decision, and the next occupied day's mean minus the previous day's; cost of misses
   (loss of that occupied day and comfort 06-09 vs G30); safety freezes.
B  Buildings: Spearman correlation over the 12 blind architectures between the layer's gain and the learned time constant, envelope heat capacity, facade U-value, floors, zones, AHUs."""
import sys, os, csv, glob, json, math, collections, re, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
man = [r for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv") for r in csv.DictReader(open(f))]
key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
Z, S = {}, {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if not r or r["policy"] not in ("G30", "SUP") or s["status"] != "OK" or s["steps_done"] != 8640: continue
    Z[(r["architecture"], r["weather"], r["policy"])] = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); S[(r["architecture"], r["weather"], r["policy"])] = s
cells = sorted({(a, w) for a, w, p in Z if (a, w, "G30") in Z and (a, w, "SUP") in Z}); assert len(cells) == 36, len(cells)
CO = {a for a, w in cells if grp[a] == "composite"}
WN = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
def comp(z, nm): ck = [str(x) for x in z["comp_keys"]]; k, wt = WN[nm]; return np.abs(wt * z["comps"][:, ck.index(k)].astype(float))
EP = {}
for a, w in cells:
    g, u = Z[(a, w, "G30")], Z[(a, w, "SUP")]; ts = [str(x) for x in g["ts"]]; assert ts == [str(x) for x in u["ts"]]
    date = np.array([t[:10] for t in ts]); hour = np.array([int(t[11:13]) for t in ts])
    import datetime as _dt
    dow = np.array([_dt.date.fromisoformat(d).weekday() for d in date])
    day_idx = np.arange(8640) // 288
    EP[(a, w)] = {"lg": np.abs(g["rewards"]), "lu": np.abs(u["rewards"]), "oat": g["oat_c"].astype(float), "date": date, "hour": hour, "dow": dow, "day_idx": day_idx,
                  "cg": {nm: comp(g, nm) for nm in WN}, "cu": {nm: comp(u, nm) for nm in WN}, "tsl": ts}
def gain_mask(maskf, archs=CO):
    num = den = 0.0
    for (a, w), e in EP.items():
        if a not in archs: continue
        m = maskf(e); num += e["lu"][m].sum(); den += e["lg"][m].sum()
    return (100 * (1 - num / den) if den > 0 else None), den
res = {"note": "post hoc, descriptive analyses of frozen, already-read blind runs; pre-registered rules and verdicts unchanged"}
# ---------------- T: time ----------------
T = {}
T["by_week"] = {lab: gain_mask(lambda e, lo=lo, hi=hi: (e["day_idx"] >= lo) & (e["day_idx"] < hi))[0] for lab, lo, hi in (("days 1-7", 0, 7), ("days 8-14", 7, 14), ("days 15-21", 14, 21), ("days 22-30", 21, 30))}
T["by_day_type"] = {"Monday": gain_mask(lambda e: e["dow"] == 0)[0], "Tue-Fri": gain_mask(lambda e: (e["dow"] >= 1) & (e["dow"] <= 4))[0], "weekend": gain_mask(lambda e: e["dow"] >= 5)[0]}
hourly = {}
for h in range(24):
    tot = sum(e["lg"][e["hour"] == h].sum() for (a, w), e in EP.items() if a in CO); allg = sum(e["lg"].sum() for (a, w), e in EP.items() if a in CO)
    d = {nm: sum((e["cu"][nm][e["hour"] == h] - e["cg"][nm][e["hour"] == h]).sum() for (a, w), e in EP.items() if a in CO) for nm in WN}
    hourly[h] = {"delta_pct_of_total_G30_loss": {nm: 100 * v / allg for nm, v in d.items()}, "gain_within_hour_pct": 100 * (1 - sum(e["lu"][e["hour"] == h].sum() for (a, w), e in EP.items() if a in CO) / tot)}
T["by_hour"] = hourly
def weekday_morning_index(e):
    # index of the weekday morning (0 = first weekday of the episode) for each step; -1 at weekends
    idx = np.full(8640, -1); k = -1; last = None
    for i in range(8640):
        if e["dow"][i] <= 4:
            if e["date"][i] != last: k += 1; last = e["date"][i]
            idx[i] = k
    return idx
for e in EP.values(): e["wmi"] = weekday_morning_index(e)
T["first_5_weekdays_vs_later"] = {"first 5 weekdays": gain_mask(lambda e: (e["wmi"] >= 0) & (e["wmi"] < 5))[0], "later weekdays": gain_mask(lambda e: e["wmi"] >= 5)[0]}
res["T_time"] = T
# ---------------- X: weather extremes and variability (calendar days) ----------------
days = []
for (a, w), e in EP.items():
    ud = list(dict.fromkeys(e["date"]))
    dm = {d: e["oat"][e["date"] == d].mean() for d in ud}
    for k, d in enumerate(ud):
        m = e["date"] == d
        if m.sum() < 200: continue                      # partial first/last day
        prev = dm.get(ud[k - 1]) if k > 0 else None
        days.append({"arch": a, "w": w, "date": d, "comp": a in CO, "tmean": float(e["oat"][m].mean()), "trange": float(e["oat"][m].max() - e["oat"][m].min()),
                     "dT": None if prev is None else float(dm[d] - prev), "lg": float(e["lg"][m].sum()), "lu": float(e["lu"][m].sum()), "dow": int(e["dow"][m][0])})
def gain_days(sel): sel = [d for d in sel if d["comp"]]; den = sum(d["lg"] for d in sel); return (100 * (1 - sum(d["lu"] for d in sel) / den) if den else None, len(sel))
X = {}
X["by_daily_mean_oat"] = {lab: gain_days([d for d in days if lo <= d["tmean"] < hi]) for lab, lo, hi in (("< -5 C", -99, -5), ("-5..5 C", -5, 5), ("5..15 C", 5, 15), ("15..25 C", 15, 25), (">= 25 C", 25, 99))}
q10, q90 = np.percentile([d["tmean"] for d in days], [10, 90])
X["extreme_days"] = {f"coldest 10% (< {q10:.1f} C)": gain_days([d for d in days if d["tmean"] < q10]), "middle 80%": gain_days([d for d in days if q10 <= d["tmean"] <= q90]), f"hottest 10% (> {q90:.1f} C)": gain_days([d for d in days if d["tmean"] > q90])}
X["by_day_to_day_change"] = {lab: gain_days([d for d in days if d["dT"] is not None and lo <= d["dT"] < hi]) for lab, lo, hi in (("cold snap <= -6 K", -99, -6), ("-6..-2 K", -6, -2), ("stable -2..2 K", -2, 2), ("2..6 K", 2, 6), ("warm surge >= 6 K", 6, 99))}
r33, r67 = np.percentile([d["trange"] for d in days], [33.3, 66.7])
X["by_within_day_range"] = {f"small (< {r33:.1f} K)": gain_days([d for d in days if d["trange"] < r33]), "medium": gain_days([d for d in days if r33 <= d["trange"] < r67]), f"large (>= {r67:.1f} K)": gain_days([d for d in days if d["trange"] >= r67])}
dg = [100 * (1 - d["lu"] / d["lg"]) for d in days if d["comp"] and d["lg"] > 0]
X["daily_gain_distribution_pct"] = {"days": len(dg), "share_better": float(np.mean([x > 0 for x in dg])), "share_worse_than_1pct": float(np.mean([x < -1 for x in dg])), "p5": float(np.percentile(dg, 5)), "median": float(np.median(dg)), "p95": float(np.percentile(dg, 95))}
worst = sorted([d for d in days if d["comp"] and d["lg"] > 0], key=lambda d: 1 - d["lu"] / d["lg"])[:10]
X["worst_10_days"] = [{"episode": f"{d['arch']}|{d['w']}", "date": d["date"], "gain_pct": 100 * (1 - d["lu"] / d["lg"]), "tmean": d["tmean"], "dT": d["dT"], "trange": d["trange"], "dow": d["dow"]} for d in worst]
res["X_weather"] = X
# ---------------- D: learned-start decisions under unknown weather ----------------
mor = []; capped = 0; freezes = []
for (a, w), e in EP.items():
    s = S[(a, w, "SUP")]; lg = (s.get("ros") or {}).get("log", [])
    if (s.get("ros") or {}).get("frozen"): freezes.append(f"{a}|{w}")
    starts = [x for x in lg if x["event"] == "start" and x.get("why") == "need"]; checks = [x for x in lg if x["event"] == "check"]
    capped += sum(1 for x in starts if x.get("capped"))
    for st in starts:
        if st.get("capped"): continue
        ck = next((c for c in checks if c["step"] > st["step"] and c["step"] - st["step"] <= 12 * 7), None)
        if ck is None: continue
        i0, i1 = st["step"], ck["step"]; oat = e["oat"]
        # the occupied day that follows (06:00-19:00 of the opening date) vs the previous calendar day's same hours
        d0 = e["date"][i1]; occ = (e["date"] == d0) & (e["hour"] >= 6) & (e["hour"] < 19)
        dates = list(dict.fromkeys(e["date"])); k = dates.index(d0); prevd = dates[k - 1] if k > 0 else None
        pocc = (e["date"] == prevd) & (e["hour"] >= 6) & (e["hour"] < 19) if prevd else None
        early = (e["date"] == d0) & (e["hour"] >= 6) & (e["hour"] < 9)
        mor.append({"episode": f"{a}|{w}", "comp": a in CO, "class": cls[(a, w)], "decision_tod_h": st["tod_h"], "lead_h": (i1 - i0) / 12.0, "D_decision": st["D_k"], "tau_h": st["tau_h"],
                    "D_pred": ck["pred_k"], "D_actual": ck["D_k"], "err": ck["D_k"] - ck["pred_k"],
                    "surprise_recovery_K": float(oat[i0:i1].mean() - oat[i0]), "surprise_day_K": (float(oat[occ].mean() - oat[pocc].mean()) if pocc is not None and pocc.any() else None),
                    "oat_decision": float(oat[i0]), "day_gain_pct": float(100 * (1 - e["lu"][occ].sum() / e["lg"][occ].sum())),
                    "comfort_06_09_delta": float((e["cu"]["comfort"][early] - e["cg"]["comfort"][early]).sum()), "energy_pre_delta": float(sum((e["cu"][nm][i0:i1] - e["cg"][nm][i0:i1]).sum() for nm in ("energy", "carbon")))})
D = {"mornings": len(mor), "capped_starts": capped, "safety_freezes": freezes}
err = np.array([m["err"] for m in mor])
D["prediction_error_K"] = {"median": float(np.median(err)), "p10": float(np.percentile(err, 10)), "p90": float(np.percentile(err, 90)), "share_within_0p5K": float(np.mean(np.abs(err) <= 0.5)), "share_miss_gt_0p5K": float(np.mean(err > 0.5))}
def bucket(vals, edges, labels, key):
    out = {}
    for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
        sel = [m for m in mor if m[key] is not None and lo <= m[key] < hi]
        if not sel: out[lab] = {"n": 0}; continue
        out[lab] = {"n": len(sel), "median_err_K": float(np.median([m["err"] for m in sel])), "share_miss": float(np.mean([m["err"] > 0.5 for m in sel])), "mean_day_gain_pct": float(np.mean([m["day_gain_pct"] for m in sel]))}
    return out
D["by_recovery_surprise"] = bucket(None, [-99, -2, -0.5, 0.5, 2, 99], ["colder by > 2 K", "colder 0.5-2 K", "stable +-0.5 K", "warmer 0.5-2 K", "warmer by > 2 K"], "surprise_recovery_K")
D["by_next_day_surprise"] = bucket(None, [-99, -6, -2, 2, 6, 99], ["day colder by > 6 K", "colder 2-6 K", "stable +-2 K", "warmer 2-6 K", "warmer by > 6 K"], "surprise_day_K")
sr = [m for m in mor if m["surprise_day_K"] is not None]
def spear(x, y): rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
D["spearman_err_vs_recovery_surprise"] = spear([m["surprise_recovery_K"] for m in mor], [m["err"] for m in mor])
D["spearman_err_vs_next_day_surprise"] = spear([m["surprise_day_K"] for m in sr], [m["err"] for m in sr])
miss = [m for m in mor if m["err"] > 0.5]; hit = [m for m in mor if m["err"] <= 0.5]
D["cost_of_misses"] = {"misses": len(miss), "hits": len(hit), "mean_day_gain_pct_miss": float(np.mean([m["day_gain_pct"] for m in miss])) if miss else None, "mean_day_gain_pct_hit": float(np.mean([m["day_gain_pct"] for m in hit])) if hit else None,
                       "mean_comfort_06_09_delta_miss": float(np.mean([m["comfort_06_09_delta"] for m in miss])) if miss else None, "mean_comfort_06_09_delta_hit": float(np.mean([m["comfort_06_09_delta"] for m in hit])) if hit else None}
D["looked_right_but_missed"] = {"n": sum(1 for m in mor if m["D_pred"] <= 0.3 + 1e-9 and m["err"] > 0.5), "of_mornings_predicted_at_target": sum(1 for m in mor if m["D_pred"] <= 0.3 + 1e-9)}
res["D_decisions"] = D; res["D_mornings"] = mor
# ---------------- B: buildings ----------------
props = {}
for l in open(f"{N}/k07x3/K07X3_MANIFEST.md"):
    if l.startswith("| building_"):
        f = [x.strip() for x in l.strip().strip("|").split("|")]; props[f[0]] = {"floors": float(f[5]), "zones": float(f[6]), "ahus": float(f[7]), "facade_U": float(f[8]), "envelope_C": float(f[9])}
archs = sorted({a for a, w in cells})
ga = {a: 100 * (1 - sum(EP[(a, w)]["lu"].sum() for aa, w in cells if aa == a) / sum(EP[(a, w)]["lg"].sum() for aa, w in cells if aa == a)) for a in archs}
tau = {a: float(np.mean([S[(a, w, "SUP")]["ros"]["tau_final_h"] for aa, w in cells if aa == a])) for a in archs}
B = {"gain_by_architecture_pct": ga, "spearman_gain_vs": {"learned_tau": spear([tau[a] for a in archs], [ga[a] for a in archs]), **{k: spear([props[a][k] for a in archs], [ga[a] for a in archs]) for k in ("envelope_C", "facade_U", "floors", "zones", "ahus")}}}
res["B_buildings"] = B
json.dump(res, open(f"{OUT}/PAPER_ERROR_ANALYSIS.json", "w"), indent=1, default=float)
f2 = lambda v: "-" if v is None else f"{v:+.2f}"
md = ["## T. Time (gain of the layer vs G30, composite)", f"- by week: { {k: f2(v) for k, v in T['by_week'].items()} }", f"- by day type: { {k: f2(v) for k, v in T['by_day_type'].items()} }",
      f"- first 5 weekdays vs later: { {k: f2(v) for k, v in T['first_5_weekdays_vs_later'].items()} }", "- by hour (change in % of total G30 loss: energy / carbon / comfort; gain within hour):"]
for h, v in hourly.items(): md.append(f"  - {h:02d}h: {v['delta_pct_of_total_G30_loss']['energy']:+.3f} / {v['delta_pct_of_total_G30_loss']['carbon']:+.3f} / {v['delta_pct_of_total_G30_loss']['comfort']:+.3f}; within-hour gain {v['gain_within_hour_pct']:+.2f} %")
md += ["", "## X. Weather extremes and variability (calendar days, composite)"] + [f"- {k}: " + "; ".join(f"{kk}: {f2(vv[0])} % (n={vv[1]})" for kk, vv in X[k].items()) for k in ("by_daily_mean_oat", "extreme_days", "by_day_to_day_change", "by_within_day_range")]
md += [f"- daily gain distribution: {X['daily_gain_distribution_pct']}", "- worst 10 days: " + "; ".join(f"{d['episode']} {d['date']} {d['gain_pct']:+.1f} % (Tmean {d['tmean']:.1f}, dT {d['dT'] if d['dT'] is None else round(d['dT'], 1)}, dow {d['dow']})" for d in X["worst_10_days"])]
md += ["", "## D. Learned-start decisions under weather unknown at decision time", f"- mornings {D['mornings']}, capped {D['capped_starts']}, freezes {D['safety_freezes']}", f"- prediction error at opening (K): {D['prediction_error_K']}",
       f"- by surprise during recovery: {D['by_recovery_surprise']}", f"- by surprise of the next occupied day: {D['by_next_day_surprise']}",
       f"- Spearman error vs recovery surprise {D['spearman_err_vs_recovery_surprise']:.2f}; vs next-day surprise {D['spearman_err_vs_next_day_surprise']:.2f}", f"- cost of misses: {D['cost_of_misses']}", f"- looked right but missed: {D['looked_right_but_missed']}",
       "", "## B. Buildings", f"- Spearman gain vs: { {k: round(v, 2) for k, v in B['spearman_gain_vs'].items()} }"]
open(f"{OUT}/error_analysis.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
