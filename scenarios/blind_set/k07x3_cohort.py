"""K07-X3 fresh blind cohort (outcome-blind; no simulation result is read).  usage: <kit venv python> k07x3_cohort.py <kit_root = k07x3/kit> <k07x3_dir> <weather_bank_dir> <window_files_list>
ARCHITECTURES: composite = the first 10 seeds of the official batch 52200-52219 (seed order) that pass the OFFICIAL 20 m2 zone-area acceptance check (the scenario generator's own
  _check_generated_zone_areas on the seeded floor plan; nothing written); OOD = the 2 shapes/seeds of gen_k07x3_buildings.py (same check; a failure re-seeds the same shape with 52302+).
WEATHER (DEVIATION, see README): the kit ships 6 recorded stations (atlanta, denver, fairbanks, minneapolis, phoenix, singapore; 48 official strips), every one used by K07-X development
  or held-out, so no official station is left.  New strips are BUILT THE OFFICIAL WAY (a 30-day window of recorded hourly station data starting 06:00, the station's convection
  coefficient, official yaml preset, official scenario generator) from the project's NOAA station bank, year 2022 (the least-used year), stations other than the six.
  Candidate windows: every Sunday 06:00 start of 2022 whose 30 days end by 2022-12-31 and overlap NO window that any project design file records for that station-year
  (all csv files with 'station' and 'start' columns: cross files, scenario indexes, cell lists; each recorded window counted as 30 days, conservative).
  Day-1 class from T19 computed offline by k07x3_t19.py (validated: reproduces all 18 K07-X 229-step probe values to 0.005 K).  Margin: C in [-14, 10] C, M in [12, 23.5] C,
  H >= 25.5 C (>= 1 K from every class threshold, so no strip is borderline and none is class E).
  Pools: for each class (H first, then C, then M) candidates in the order sha256('K07X3|STRIP|<class>|<station-year>|<start>'), taken greedily with distinct stations inside a
  class and no two chosen windows overlapping on one station-year, 12 per class (one strip per architecture and class; no strip is shared).
  Assignment: architectures in the order sha256('K07X3|ORDER|<name>'); architecture i gets C[i], M[(i+a)%12], H[(i+b)%12] with the smallest (a, b) that gives every
  architecture three different stations (the fixed-offset rule of k07x_v2_cohort.py)."""
import sys, json, hashlib, subprocess, re, os, csv, random, datetime as dt, shutil, collections
from pathlib import Path
import numpy as np, yaml
KIT, D, BANK, WINLIST = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(D)); sys.path.insert(0, str(KIT)); os.chdir(KIT)
import k07x3_t19 as T19
from smart_buildings.smart_control.utils import scenario_generator as SG
H = lambda *a: hashlib.sha256("|".join(map(str, a)).encode()).hexdigest()
raw = json.load(open(D / "K07X3_BUILDINGS_RAW.json")); by = {r["name"]: r for r in raw["candidates"]}
def accept(name):
    cfg = SG.load_scenario_from_parts(str(KIT / "smart_buildings/buildings" / f"{name}.yaml"), "denver_1")
    with SG._floor_plan_generation_lock:
        random.seed(cfg.seed); np.random.seed(cfg.seed); g = SG._create_floor_plan_generator(cfg.floor_plan); g.generate()
        try: SG._check_generated_zone_areas(g, SG.CONTROL_VOLUME_CM, cfg.seed); return True, ""
        except ValueError as e: return False, str(e)[:400]
acc = {}; comp = []
for r in sorted((r for r in raw["candidates"] if r["group"] == "composite"), key=lambda r: r["seed"]):
    if len(comp) == 10: break
    ok, msg = accept(r["name"]); acc[r["name"]] = {"accepted": ok, "message": msg}
    if ok: comp.append(r["name"])
assert len(comp) == 10, acc
ood = []
for r in [r for r in raw["candidates"] if r["group"] == "ood"]:
    ok, msg = accept(r["name"]); acc[r["name"]] = {"accepted": ok, "message": msg}
    if not ok: raise SystemExit(f"OOD {r['name']} rejected: {msg} -> re-seed rule needed")
    ood.append(r["name"])
archs = comp + ood
# ---- weather pool (weather only) ----
OFFICIAL6 = {"atlanta", "denver", "fairbanks", "minneapolis", "phoenix", "singapore"}
used = collections.defaultdict(set); cc_seen = collections.defaultdict(collections.Counter); nwin = 0
for f in [l.strip() for l in open(WINLIST) if l.strip()]:
    for r in csv.DictReader(open(f)):
        st, s = (r.get("station") or "").strip(), (r.get("start") or "").strip()[:16]
        if not st or not s: continue
        try: t = dt.datetime.strptime(s, "%Y-%m-%d %H:%M")
        except ValueError:
            try: t = dt.datetime.strptime(s[:10], "%Y-%m-%d")
            except ValueError: continue
        nwin += 1
        for k in range(31): used[st].add((t + dt.timedelta(days=k)).date())     # 30 days + the partial day after a 06:00 start
        if r.get("convection_coefficient"): cc_seen[st][float(r["convection_coefficient"])] += 1
files = sorted(p for p in BANK.glob("*22.csv") if re.sub(r"\d+$", "", p.stem) not in OFFICIAL6)
starts = [dt.datetime(2022, 1, 2, 6) + dt.timedelta(days=7 * k) for k in range(60)]
starts = [s for s in starts if s + dt.timedelta(days=30) <= dt.datetime(2022, 12, 31, 0)]
cand = []
for p in files:
    sy = p.stem; base = re.sub(r"\d+$", "", sy); rows = list(csv.DictReader(open(p)))
    if len(rows) != 8760 or any(r["TempF"] in ("", "nan", "NaN") for r in rows): continue
    cc = (cc_seen[sy] or cc_seen[base] or cc_seen[base + "21"]).most_common(1)
    if not cc: continue
    for s in starts:
        days = {(s + dt.timedelta(days=k)).date() for k in range(31)}
        if days & used[sy]: continue
        t = T19.t19(KIT, str(p), s.strftime("%Y-%m-%d %H:%M")); c = T19.classify(t)
        inm = (c == "C" and -14 <= t <= 10) or (c == "M" and 12 <= t <= 23.5) or (c == "H" and t >= 25.5)
        cand.append({"station_year": sy, "station": base, "start": s.strftime("%Y-%m-%d %H:%M"), "day1_T19": round(t, 2), "day1_class": c, "in_margin": inm, "convection_coefficient": cc[0][0], "days": days})
pools, taken = {}, collections.defaultdict(set)
for cls in ("H", "C", "M"):
    pick = []
    for c in sorted((c for c in cand if c["day1_class"] == cls and c["in_margin"]), key=lambda c: H("K07X3|STRIP", cls, c["station_year"], c["start"])):
        if len(pick) == 12: break
        if c["station"] in {x["station"] for x in pick} or c["days"] & taken[c["station_year"]]: continue
        pick.append(c); taken[c["station_year"]] |= c["days"]
    assert len(pick) == 12, (cls, len(pick)); pools[cls] = pick
for cls in pools:
    for c in pools[cls]: c["weather"] = f'{c["station_year"]}_w{c["start"][5:7]}{c["start"][8:10]}'
order = sorted(archs, key=lambda n: H("K07X3|ORDER", n))
a_b = next((a, b) for a in range(12) for b in range(12) if all(len({pools["C"][i]["station"], pools["M"][(i + a) % 12]["station"], pools["H"][(i + b) % 12]["station"]}) == 3 for i in range(12)))
assign = {n: [pools["C"][i], pools["M"][(i + a_b[0]) % 12], pools["H"][(i + a_b[1]) % 12]] for i, n in enumerate(order)}
# ---- official presets + scenario generation ----
(KIT / "smart_buildings/weather/csv").mkdir(parents=True, exist_ok=True); cells = []
for cls in pools:
    for c in pools[cls]:
        dst = KIT / "smart_buildings/weather/csv" / f'{c["station_year"]}.csv'
        if not dst.exists(): shutil.copy2(BANK / f'{c["station_year"]}.csv', dst)
        y = KIT / "smart_buildings/weather" / f'{c["weather"]}.yaml'
        if not y.exists(): y.write_text(yaml.safe_dump({"name": c["weather"], "weather_csv": f'csv/{c["station_year"]}.csv', "convection_coefficient": c["convection_coefficient"], "start_timestamp": c["start"]}, sort_keys=False))
for n in order:
    ws = [c["weather"] for c in assign[n]]; need = [w for w in ws if not (KIT / "smart_buildings/training_scenarios" / n / w / "config.gin").exists()]; err = ""
    if need:
        p = subprocess.run([sys.executable, "-m", "smart_buildings.run_scenario_generator", "--buildings", n, "--weathers", *need], cwd=KIT, capture_output=True, text=True); err = (p.stderr or "")[-300:] if p.returncode else ""
    for c in assign[n]:
        d = KIT / "smart_buildings/training_scenarios" / n / c["weather"]; ok = (d / "config.gin").exists(); gin = (d / "config.gin").read_text() if ok else ""
        ah = re.findall(r"^ahu_(\d+)_rooms = \[(.*?)\]", gin, re.M); st = re.search(r"^start_timestamp = '(.*?)'", gin, re.M)
        cells.append({"architecture": n, "group": by[n]["group"], "split": "fresh", "weather": c["weather"], "station": c["station_year"], "start": c["start"], "convection_coefficient": c["convection_coefficient"],
                      "day1_class": c["day1_class"], "day1_T19": c["day1_T19"], "ok": ok and bool(st) and st.group(1) == c["start"], "error": err if not ok else "",
                      "zones": sum(len(re.findall(r"'room_\d+'", x[1])) for x in ah), "ahus": len(ah), "zones_per_ahu_max": max((len(re.findall(r"'room_\d+'", x[1])) for x in ah), default=0)})
arch = []
for n in order:
    rr = [c for c in cells if c["architecture"] == n]; r = by[n]
    arch.append({**{k: r[k] for k in ("name", "seed", "group", "shape", "style", "width_ft", "height_ft", "floors", "room_min_ft", "room_max_ft", "shape_ratio", "coverage", "ahu_floor", "occupants_per_zone", "facade_U_W_m2K", "envelope_C_MJ_m3K")},
                 "materials": r["materials"], "split": "fresh", "class": "composite (in-distribution new draw)" if r["group"] == "composite" else "OOD morphology", "weathers": [c["weather"] for c in rr],
                 "day1_classes": [c["day1_class"] for c in rr], "zones": rr[0]["zones"], "ahus": rr[0]["ahus"], "zones_per_ahu_max": rr[0]["zones_per_ahu_max"], "ok": all(c["ok"] for c in rr) and len({c["zones"] for c in rr}) == 1})
cnt = collections.Counter((c["day1_class"], c["in_margin"]) for c in cand)
json.dump({"rules": __doc__, "acceptance": acc, "rejected_by_official_acceptance": sorted(n for n, v in acc.items() if not v["accepted"]), "composite_pool": comp, "ood": ood, "architecture_order": order,
           "offsets_M_H": list(a_b), "weather_source": {"bank_dir": str(BANK), "station_year_files_considered": [p.stem for p in files], "recorded_windows_read": nwin, "window_files": [l.strip() for l in open(WINLIST) if l.strip()],
           "candidates_by_class_and_margin": {f"{k[0]}_{'in' if k[1] else 'out'}": v for k, v in sorted(cnt.items())}},
           "strip_pools": {cls: [{k: v for k, v in c.items() if k != "days"} for c in pools[cls]] for cls in pools}, "architectures": arch, "cells": cells}, open(D / "K07X3_COHORT.json", "w"), indent=1, default=str)
for a in arch: print(f'{a["name"]} {a["class"][:9]:9s} {a["shape"]:13s} {a["width_ft"]}x{a["height_ft"]} fl{a["floors"]} zones {a["zones"]:3d} AHUs {a["ahus"]:2d} ok {a["ok"]}  ' + "  ".join(f'{c["weather"]}({c["day1_class"]} {c["day1_T19"]})' for c in cells if c["architecture"] == a["name"]))
print("rejected:", [n for n, v in acc.items() if not v["accepted"]], "offsets", a_b, "candidates", dict(cnt))
