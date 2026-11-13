"""H2 design (written before any H2 run): prospective M-gate test on NOAA 2021 weather (no earlier experiment used 2021: start years in all cross/scenario tables are 2023 and 2022 only).  100 main cells + 20 pre-frozen extension cells (fixed order, extension used only if throughput allows, never chosen by outcome).  Exact-window QC (Rule W of noaa_year.py on the raw NOAA file for the exact 720 h of every candidate Sunday-06:00 window), classes by T19 = mean of the first 19 hourly temperatures (E<-15, C<11, M<24.5, H>=24.5; E excluded), quotas follow the reference weights (C .31, M .43, H .26 -> main 30/44/26, extension 6/9/5), spaced over the T19 range of each class, at most 3 windows per station overall, at least 35 days between two windows of one station (no overlapping 30-day windows).  Buildings: half of every class on GEN12 (seen in H0 under 2022 weather: isolates the weather shift), half on GEN14 (new variants: separate variant-stress subset), alternating over the sorted T19 so both subsets span the same range.
usage (from DABU): python -I N10S_design/make_design_h2.py"""
import csv, json, glob, datetime as dt, collections, numpy as np
N = "N10S_design"; J = "../../.."; R21 = f"{J}/noaa_2021_out"; st = json.load(open(f"{N}/noaa_stations_43.json")); st.pop("birmingham", None)
src = open(f"{N}/noaa_year.py").read(); exec(src[:src.index("def main():")])
cls = lambda T: "E" if T < -15 else "C" if T < 11 else "M" if T < 24.5 else "H"; W = []
for city, (sid, off, label, country) in st.items():
    raw = f"{J}/noaa_isd_full_2021_raw/{sid}-2021.gz"; df = read_full_isd(raw); local, h, filled = to_local(df, 2021, off); T = h["t"].to_numpy(); ts = list(local)
    for i, t in enumerate(ts):
        if t.weekday() == 6 and t.hour == 6 and i + 735 <= len(T) and window_ok(local, h, filled, t):
            W.append(dict(city=city, station=city + "21", start=t.strftime("%Y-%m-%d 06:00"), month=t.month, T19=float(T[i:i + 19].mean()), cls=cls(float(T[i:i + 19].mean())), d=dt.datetime(t.year, t.month, t.day)))
print("eligible exact-QC windows by class", dict(collections.Counter(w["cls"] for w in W)), "stations", len({w["city"] for w in W}))
QUOTA = {"M": 44 + 9, "C": 30 + 6, "H": 26 + 5}; cnt = collections.Counter(); chosen = []
def ok(w): return cnt[w["city"]] < 3 and all(abs((w["d"] - c["d"]).days) >= 35 for c in chosen if c["city"] == w["city"])
for k in ("M", "C", "H"):
    pool = sorted([w for w in W if w["cls"] == k], key=lambda w: w["T19"]); sel = []
    for j in np.linspace(0, len(pool) - 1, QUOTA[k]).astype(int):
        for d in range(len(pool)):
            hit = False
            for jj in (j + d, j - d):
                if 0 <= jj < len(pool) and pool[jj] not in sel and ok(pool[jj]): sel.append(pool[jj]); chosen.append(pool[jj]); cnt[pool[jj]["city"]] += 1; hit = True; break
            if hit: break
    assert len(sel) == QUOTA[k], (k, len(sel))
EXT = {"M": 9, "C": 6, "H": 5}; byc = {k: sorted([w for w in chosen if w["cls"] == k], key=lambda w: w["T19"]) for k in "CMH"}; main, ext = [], []
for k in "CMH":
    p = byc[k]; ei = set(np.linspace(1, len(p) - 2, EXT[k]).astype(int)); main += [(k, w) for i, w in enumerate(p) if i not in ei]; ext += [(k, w) for i, w in enumerate(p) if i in ei]
g12 = sorted(json.load(open(f"{N}/cells_h0.csv".replace("cells_h0.csv", "cells_h0.csv")) ) if False else []) 
b12 = sorted({r["building"] for r in csv.DictReader(open(f"{N}/cells_h0.csv"))} | {r["building"] for r in csv.DictReader(open(f"{N}/cells_h0s.csv"))}); assert len(b12) == 36, len(b12)
R14 = json.load(open(f"{J}/b14_gen14/b2c_record_gen14.json"))["sets"]["gen14"]["buildings"]; b14 = sorted(r["scenario_dir"].split("/")[-2] for r in R14); assert len(b14) == 60
base = {r["scenario_dir"].split("/")[-2]: r["base"] for r in R14}
cells, c12, c14 = [], 0, 0; wid = lambda w: f"{w['station']}_w{w['start'][5:7]}{w['start'][8:10]}"
def build(group, name):
    global c12, c14
    out = []; per = collections.defaultdict(list)
    for k, w in group: per[k].append(w)
    for k in "CMH":
        for i, w in enumerate(sorted(per[k], key=lambda w: w["T19"])):
            gen = "gen12" if i % 2 == 0 else "gen14"
            if gen == "gen12": b = b12[(c12 * 5) % 36]; c12 += 1
            else: b = b14[c14]; c14 += 1
            out.append(dict(cls=k, building=b, generator=gen, base_index=int(base[b].split("_")[-1]) if b in base and base[b][-1].isdigit() else -1, weather=wid(w), station=w["station"], T19_file=round(w["T19"], 2), start=w["start"], cohort=name))
    return out
main_c = build(main, "H2-main"); ext_c = build(ext, "H2-ext")
def interleave(cs): 
    by = {k: [c for c in cs if c["cls"] == k] for k in "CMH"}; out = []
    for i in range(max(len(v) for v in by.values())):
        for k in "MCH":
            if i < len(by[k]): out.append(by[k][i])
    return out
cells = interleave(main_c) + interleave(ext_c); assert len({(c["building"], c["weather"]) for c in cells}) == 120
with open(f"{N}/cells_h2.csv", "w", newline="") as fh: w_ = csv.DictWriter(fh, fieldnames=list(cells[0])); w_.writeheader(); w_.writerows(cells)
rec = json.load(open(f"{R21}/record.json"))["stations"]; wins = {c["weather"]: c for c in cells}
open(f"{N}/cross_h2.csv", "w").write("weather,station,month,start,convection_coefficient\n" + "".join(f"{k},{v['station']},{int(v['start'][5:7])},{v['start']},{rec[v['station'][:-2]]['convection_coefficient']}\n" for k, v in sorted(wins.items())))
ua = json.load(open(f"{N}/h0_arms.json")); arms = dict(ua); arms["MG"] = "k07unif@mgatehot"; json.dump(arms, open(f"{N}/h2_arms.json", "w"), indent=1)
jobs = [{"line": f"{c['building']} {c['weather']} {k} 8640", "pin": None} for c in cells for k in (arms["B"], arms["PS5"], arms["PD5"], arms["TRIG2"], arms["G30"], arms["MG"])]
json.dump({"prefix": "h2", "cross": "cross_h2.csv", "kscen": "sb26-h0-scenarios", "res": "results_h2", "runner": "run_episode_k07_v8_mg.py", "jobs": jobs[:600], "jobs_extension": jobs[600:], "rates": {"host-B": 1.0, "kaggle": 0.69, "laptop": 0.2}}, open(f"{N}/plan_h2.json", "w"), indent=1)
m = [c for c in cells if c["cohort"] == "H2-main"]
print("cells", len(cells), "main", len(m), {k: sum(1 for c in m if c["cls"] == k) for k in "CMH"}, "generators main", dict(collections.Counter(c["generator"] for c in m)), "stations", len({c["station"] for c in cells}), "buildings", len({c["building"] for c in cells}), "runs main", 600, "ext", len(jobs) - 600)
print("windows per station max", max(collections.Counter(c["station"] for c in cells).values()), "T19 ranges", {k: (min(c["T19_file"] for c in cells if c["cls"] == k), max(c["T19_file"] for c in cells if c["cls"] == k)) for k in "CMH"})
