"""p273c: mean observed temperature of the days in each class by the class of the episode's start (2021 main cohort, 98 episodes; the
'unif' decision logs of the daily re-selection runs; days after day 1).  Descriptive support for the reading of p273 (mild days within
cold- or hot-start episodes lie near the class edges).  Post hoc; nothing is run.  usage: python -I p273c_class_temperatures.py <out_dir>"""
import sys, os, csv, glob, json, collections, statistics as S
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
cells = {(c["building"], c["weather"]) for c in csv.DictReader(open(f"{N}/cells_h2.csv")) if c["cohort"] == "H2-main"}
T, seen = collections.defaultdict(list), set()
for f in sorted(glob.glob(f"{N}/results_h2/**/*.summary.json", recursive=True)):
    s = json.load(open(f)); k = (s.get("building"), s.get("weather"))
    if k not in cells or s.get("policy_key") != "k07unif@dailyhot" or s.get("status") != "OK" or s.get("steps_done") != 8640 or k in seen or not s.get("unif"): continue
    seen.add(k); u = sorted(s["unif"], key=lambda e: e["day"]); first = u[0]["class"]
    for e in u[1:]:
        if e.get("class") and e.get("t_obs_c") is not None: T[f"{first}->{e['class']}"].append(float(e["t_obs_c"]))
assert len(seen) == 98, len(seen)
R = {"note": "post hoc, descriptive; mean observed temperature of the 228-step window before each daily decision, by start class -> day class",
     "episodes": len(seen), "by_direction": {k: {"days": len(v), "mean_c": S.mean(v), "median_c": S.median(v)} for k, v in sorted(T.items())}}
json.dump(R, open(f"{OUT}/P273C_CLASS_TEMPERATURES.json", "w"), indent=1)
for k, v in R["by_direction"].items(): print(f"{k:5s} days {v['days']:4d} mean {v['mean_c']:6.2f} C median {v['median_c']:6.2f}")
