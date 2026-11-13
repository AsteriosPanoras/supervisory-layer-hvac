"""H3 design (written 2026-11-12, before any H3 run): a pre-registered, independent replication of the 2021 M-gate test (H2, PREREG_H2_v1),
made after H2's 20 extension cells gave -0.004 points against +0.504 [0.337, 0.704] in its 98 complete main cells.  40 NEW cells; nothing is
chosen by outcome (this script reads no reward).
- Windows: the H2 eligibility unchanged (Sunday 06:00 starts; exact-window QC Rule W of noaa_year.py on the raw NOAA 2021 file; 735 h
  available; the same 42 stations), classes by T19 = mean of the first 19 hourly temperatures (C < 11, M < 24.5, H >= 24.5; E < -15 excluded).
  The pool of a class holds the windows that start at least 35 days from every H2 window of their station (no overlap with any H2 episode).
  Quotas in the H2 proportions: M 18, C 12, H 10.  Within a class the pool is sorted by T19 and the windows at np.linspace positions are
  taken, else the nearest feasible one (the H2 rule); feasible = its station gives at most 2 H3 windows, at least 35 days apart.
- Buildings: alternating GEN12/GEN14 over the sorted T19 of each class (the H2 rule, so both subsets span the same range).  GEN12: the same
  36 buildings as H2; GEN14: the 60 variants minus b2_49016 and b2_49017, which the simulator cannot initialise (H2 amendment 1).  Each sorted
  list is walked with stride 7 from index 3 (7 is prime to 36 and 58), so no building is used twice in H3.
- Arms: B, TRIG2, MG with the frozen H2 keys (h2_arms.json); the H2 runner run_episode_k07_v8_mg.py, unchanged.
Writes cells_h3.csv (cohort H3, interleaved M/C/H as in H2), cross_h3.csv, h3_arms.json, plan_h3_jobs.json (cell-major: B, TRIG2, MG).
usage (from DABU): python -I N10S_design/make_design_h3.py"""
import os, csv, json, datetime as dt, collections, numpy as np
N = "N10S_design"; J = "../../.."; R21 = f"{J}/noaa_2021_out"; st = json.load(open(f"{N}/noaa_stations_43.json")); st.pop("birmingham", None)
OUTS = ("cells_h3.csv", "cross_h3.csv", "h3_arms.json", "plan_h3_jobs.json")
for f in OUTS: assert not os.path.exists(f"{N}/{f}"), f"{f} exists: the design is written once"
src = open(f"{N}/noaa_year.py").read(); exec(src[:src.index("def main():")])
cls = lambda T: "E" if T < -15 else "C" if T < 11 else "M" if T < 24.5 else "H"; W = []
for city, (sid, off, label, country) in st.items():
    raw = f"{J}/noaa_isd_full_2021_raw/{sid}-2021.gz"; df = read_full_isd(raw); local, h, filled = to_local(df, 2021, off); T = h["t"].to_numpy(); ts = list(local)
    for i, t in enumerate(ts):
        if t.weekday() == 6 and t.hour == 6 and i + 735 <= len(T) and window_ok(local, h, filled, t):
            W.append(dict(city=city, station=city + "21", start=t.strftime("%Y-%m-%d 06:00"), month=t.month, T19=float(T[i:i + 19].mean()), cls=cls(float(T[i:i + 19].mean())), d=dt.datetime(t.year, t.month, t.day)))
H2 = list(csv.DictReader(open(f"{N}/cells_h2.csv"))); assert len(H2) == 120
Wkey = {(w["station"], w["start"]): w for w in W}
assert all((c["station"], c["start"]) in Wkey and Wkey[(c["station"], c["start"])]["cls"] == c["cls"] for c in H2), "the H2 windows are not reproduced"
used = collections.defaultdict(list)
for c in H2: used[c["station"]].append(dt.datetime.strptime(c["start"][:10], "%Y-%m-%d"))
far = lambda w: all(abs((w["d"] - d).days) >= 35 for d in used[w["station"]])
print("eligible exact-QC windows", dict(collections.Counter(w["cls"] for w in W)), "| not overlapping H2", dict(collections.Counter(w["cls"] for w in W if far(w))))
QUOTA = {"M": 18, "C": 12, "H": 10}; cnt = collections.Counter(); chosen = []
def ok(w): return cnt[w["city"]] < 2 and all(abs((w["d"] - c["d"]).days) >= 35 for c in chosen if c["city"] == w["city"])
for k in ("M", "C", "H"):
    pool = sorted([w for w in W if w["cls"] == k and far(w)], key=lambda w: w["T19"]); sel = []
    for j in np.linspace(0, len(pool) - 1, QUOTA[k]).astype(int):
        for d in range(len(pool)):
            hit = False
            for jj in (j + d, j - d):
                if 0 <= jj < len(pool) and pool[jj] not in sel and ok(pool[jj]): sel.append(pool[jj]); chosen.append(pool[jj]); cnt[pool[jj]["city"]] += 1; hit = True; break
            if hit: break
    assert len(sel) == QUOTA[k], (k, len(sel), len(pool))
b12 = sorted({r["building"] for r in csv.DictReader(open(f"{N}/cells_h0.csv"))} | {r["building"] for r in csv.DictReader(open(f"{N}/cells_h0s.csv"))}); assert len(b12) == 36, len(b12)
R14 = json.load(open(f"{J}/b14_gen14/b2c_record_gen14.json"))["sets"]["gen14"]["buildings"]; b14all = sorted(r["scenario_dir"].split("/")[-2] for r in R14); assert len(b14all) == 60
base = {r["scenario_dir"].split("/")[-2]: r["base"] for r in R14}
assert {c["building"] for c in H2 if c["generator"] == "gen12"} <= set(b12) and {c["building"] for c in H2 if c["generator"] == "gen14"} <= set(b14all)
b14 = [b for b in b14all if b not in ("b2_49016", "b2_49017")]; assert len(b14) == 58
wid = lambda w: f"{w['station']}_w{w['start'][5:7]}{w['start'][8:10]}"
out, c12, c14 = [], 0, 0
for k in "CMH":
    for i, w in enumerate(sorted([w for w in chosen if w["cls"] == k], key=lambda w: w["T19"])):
        gen = "gen12" if i % 2 == 0 else "gen14"
        if gen == "gen12": b = b12[(3 + 7 * c12) % 36]; c12 += 1
        else: b = b14[(3 + 7 * c14) % 58]; c14 += 1
        out.append(dict(cls=k, building=b, generator=gen, base_index=int(base[b].split("_")[-1]) if b in base and base[b][-1].isdigit() else -1,
                        weather=wid(w), station=w["station"], T19_file=round(w["T19"], 2), start=w["start"], cohort="H3"))
def interleave(cs):
    by = {k: [c for c in cs if c["cls"] == k] for k in "CMH"}; o = []
    for i in range(max(len(v) for v in by.values())):
        for k in "MCH":
            if i < len(by[k]): o.append(by[k][i])
    return o
cells = interleave(out)
assert len(cells) == 40 and len({c["building"] for c in cells}) == 40 and len({c["weather"] for c in cells}) == 40
assert not ({(c["building"], c["weather"]) for c in cells} & {(c["building"], c["weather"]) for c in H2}) and not ({c["weather"] for c in cells} & {c["weather"] for c in H2})
with open(f"{N}/cells_h3.csv", "w", newline="") as fh: w_ = csv.DictWriter(fh, fieldnames=list(cells[0])); w_.writeheader(); w_.writerows(cells)
rec = json.load(open(f"{R21}/record.json"))["stations"]
open(f"{N}/cross_h3.csv", "w").write("weather,station,month,start,convection_coefficient\n" + "".join(
    f"{c['weather']},{c['station']},{int(c['start'][5:7])},{c['start']},{rec[c['station'][:-2]]['convection_coefficient']}\n" for c in cells))
a2 = json.load(open(f"{N}/h2_arms.json")); arms = {a: a2[a] for a in ("B", "TRIG2", "MG")}; json.dump(arms, open(f"{N}/h3_arms.json", "w"), indent=1)
jobs = [f"{c['building']} {c['weather']} {arms[a]} 8640" for c in cells for a in ("B", "TRIG2", "MG")]
json.dump({"cells": len(cells), "jobs": jobs}, open(f"{N}/plan_h3_jobs.json", "w"), indent=1)
mind = min(min(abs((dt.datetime.strptime(c["start"][:10], "%Y-%m-%d") - d).days) for d in used[c["station"]]) if used[c["station"]] else 999 for c in cells)
print("cells", len(cells), {k: sum(c["cls"] == k for c in cells) for k in "CMH"}, "| generators", dict(collections.Counter(c["generator"] for c in cells)),
      "| stations", len({c["station"] for c in cells}), "max per station", max(collections.Counter(c["station"] for c in cells).values()),
      "| T19 ranges", {k: (min(c["T19_file"] for c in cells if c["cls"] == k), max(c["T19_file"] for c in cells if c["cls"] == k)) for k in "CMH"},
      "| months", dict(sorted(collections.Counter(int(c["start"][5:7]) for c in cells).items())), "| min days to an H2 window of the same station", mind, "| jobs", len(jobs))
