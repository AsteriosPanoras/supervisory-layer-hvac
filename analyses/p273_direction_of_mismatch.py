"""p273: why the one-time gate works - the direction of a profile mismatch.  Post hoc, descriptive; 2021 cohort (H2),
98 main episodes; nothing is run.
Select-once (PS5) keeps the first profile; daily re-selection (PD5) adopts the profile of the observed class at every daily decision (step
228 + 288 (d - 1)).  Both arms log the profile active after each decision, so every decision window (from one decision to the next) is
either matched (same profile in both arms) or a mismatch in a known direction: PS5 runs the first class's profile, PD5 the current class's.
Per window, the loss difference PS5 - PD5 is the sum of the step losses (-reward) over its steps.  The windows add up exactly to the
episode difference, so the daily-minus-once gain difference of each start class splits exactly into contributions by the class of the
day (the direction), plus the matched windows after the first mismatch, which carry the effect of earlier differences in the building
state (carry-over).  Checks: (0) the class gains by the weather-file label equal those printed in Appendix E (select-once 4.96 / 1.81 /
2.71, daily 4.63 / 4.40 / 2.44 after cold / mild / hot starts); (1) before the first mismatch window the two arms' step losses are
bit-identical, so episodes without a mismatch window have identical losses; (2) the contributions add up to the class differences.
Groups by the class observed at the first decision (the controller's view; it differs from the weather-file label in three episodes);
extreme cold (E) is its own class, as the controller has an E profile.  Per direction: contribution in pp of baseline-referenced gain
of that start class, mismatch days, and rate = PS5 - PD5 loss on those days as % of B's loss on the same days.  Intervals: the 42 main
stations resampled 3,000 times (seed 1).  Version 1 of this script (never produced output) merged E into C and failed check 0, because
Appendix E's class gains use the weather-file label; this version checks them with the label.  Optional second argument H2-ext runs the
same decomposition on the 20-episode 2021 extension (check 0 and the episode count apply to the main cohort only).
usage: python -I p273_direction_of_mismatch.py <out_dir> [H2-main|H2-ext]"""
import sys, os, csv, glob, json, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
COH = sys.argv[2] if len(sys.argv) > 2 else "H2-main"; assert COH in ("H2-main", "H2-ext")
KEYS = {"B": "k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75", "PS5": "k07unif@statichot", "PD5": "k07unif@dailyhot"}
cells = [c for c in csv.DictReader(open(f"{N}/cells_h2.csv")) if c["cohort"] == COH]; info = {(c["building"], c["weather"]): c for c in cells}
RW, LOG = collections.defaultdict(dict), collections.defaultdict(dict)
for f in glob.glob(f"{N}/results_h2/**/*.summary.json", recursive=True):
    s = json.load(open(f)); k = (s.get("building"), s.get("weather"))
    if k not in info or s.get("status") != "OK" or s.get("steps_done") != 8640: continue
    a = next((a for a, v in KEYS.items() if v == s["policy_key"]), None)
    if a is None: continue
    r = -np.asarray(np.load(f[:-len(".summary.json")] + ".npz", allow_pickle=True)["rewards"], float)[:8640]
    if a in RW[k]:
        assert np.array_equal(RW[k][a], r), ("duplicate run with different rewards", k, a); continue
    RW[k][a] = r
    if a != "B": LOG[k][a] = sorted(s["unif"], key=lambda e: e["day"])
eps = sorted(k for k in RW if len(RW[k]) == 3); assert len(eps) == (98 if COH == "H2-main" else 20), len(eps)
L = {k: {a: float(RW[k][a].mean()) for a in KEYS} for k in eps}
g = lambda a, S: 100 * (1 - sum(L[k][a] for k in S) / sum(L[k]["B"] for k in S))
lab = {c: [k for k in eps if info[k]["cls"] == c] for c in "CMH"}
if COH == "H2-main":
  assert [f"{g('PS5', lab[c]):.2f}" for c in "CMH"] == ["4.96", "1.81", "2.71"], [g("PS5", lab[c]) for c in "CMH"]      # check 0
  assert [f"{g('PD5', lab[c]):.2f}" for c in "CMH"] == ["4.63", "4.40", "2.44"], [g("PD5", lab[c]) for c in "CMH"]
# windows and directions
W = {}            # episode -> list of (start, end, profile_PS5, profile_PD5)
first, n_bitcheck = {}, 0
for k in eps:
    ps, pd = LOG[k]["PS5"], LOG[k]["PD5"]
    assert [e["step"] for e in ps] == [e["step"] for e in pd] and [e.get("class") for e in ps] == [e.get("class") for e in pd], ("logs differ", k)
    steps = [e["step"] for e in ps]; bounds = steps + [8640]; aps = apd = None; win = []
    for i, (eps_, epd) in enumerate(zip(ps, pd)):
        aps = eps_.get("active_after") or aps; apd = epd.get("active_after") or apd
        win.append((bounds[i], bounds[i + 1], aps, apd))
    assert win[0][2] == win[0][3], ("first profiles differ", k); first[k] = win[0][2]
    assert all(w[2] == first[k] for w in win), ("select-once changed profile", k)
    s0 = next((w[0] for w in win if w[2] != w[3]), 8640)
    assert np.array_equal(RW[k]["PS5"][:s0], RW[k]["PD5"][:s0]), ("arms differ before the first mismatch", k); n_bitcheck += 1      # check 1
    W[k] = (win, s0)
assert all(first[k] in "CMH" for k in eps), collections.Counter(first.values())
obs = {c: [k for k in eps if first[k] == c] for c in "CMH"}
def contrib(sel):
    """per start class: {direction or 'matched after divergence': (sum PS5-PD5, sum B on those windows, days, episodes)} and the B total."""
    out = {}
    for c in "CMH":
        S = [k for k in sel if first[k] == c]; tot_b = sum(RW[k]["B"].sum() for k in S); d = collections.defaultdict(lambda: [0.0, 0.0, 0, set()])
        for k in S:
            win, s0 = W[k]
            for a, b, p1, p2 in win:
                if p1 == p2 and a < s0: continue
                key = p2 if p1 != p2 else "matched"
                x = d[key]; x[0] += RW[k]["PS5"][a:b].sum() - RW[k]["PD5"][a:b].sum(); x[1] += RW[k]["B"][a:b].sum(); x[2] += 1; x[3].add(k)
        out[c] = (tot_b, {key: (v[0], v[1], v[2], len(v[3])) for key, v in d.items()}, len(S))
    return out
main = contrib(eps)
for c in "CMH":                                                                                  # check 2
    tot_b, d, n = main[c]; S = obs[c]
    diff = 100 * sum(RW[k]["PS5"].sum() - RW[k]["PD5"].sum() for k in S) / tot_b
    assert abs(sum(100 * v[0] / tot_b for v in d.values()) - diff) < 1e-9, c
    assert abs(diff - (g("PD5", S) - g("PS5", S))) < 1e-9, c
stations = sorted({info[k]["station"] for k in eps}); by_st = {s_: [k for k in eps if info[k]["station"] == s_] for s_ in stations}
rng = np.random.default_rng(1); BC, BR = collections.defaultdict(list), collections.defaultdict(list)
for _ in range(3000):
    sel = [k for j in rng.integers(0, len(stations), len(stations)) for k in by_st[stations[j]]]
    m = contrib(sel)
    for c in "CMH":
        tot_b, d, n = m[c]
        if not n: continue
        for key in ("E", "C", "M", "H", "matched"):
            v = d.get(key)
            BC[(c, key)].append(100 * v[0] / tot_b if v else 0.0)
            if v and v[1] > 0: BR[(c, key)].append(100 * v[0] / v[1])
ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] if len(v) >= 100 else None
R = {"note": "post hoc, descriptive; 2021 main episodes (98); classes as observed by the controller at each decision; equal weights; the "
             "per-window differences after the first mismatch include carry-over from the building state, so they are not isolated per-switch costs",
     "check1_episodes_bit_identical_before_first_mismatch": n_bitcheck, "stations": len(stations),
     "class_gains_by_weather_file_label": {c: {"n": len(lab[c]), "PS5": round(g("PS5", lab[c]), 3), "PD5": round(g("PD5", lab[c]), 3)} for c in "CMH"},
     "class_gains_by_observed_class": {c: {"n": len(obs[c]), "PS5": round(g("PS5", obs[c]), 3), "PD5": round(g("PD5", obs[c]), 3)} for c in "CMH"},
     "by_start_class": {}}
for c in "CMH":
    tot_b, d, n = main[c]
    R["by_start_class"][c] = {"episodes": n, "episodes_with_a_mismatch": sum(1 for k in obs[c] if W[k][1] < 8640),
                              "daily_minus_once_pp": round(g("PD5", obs[c]) - g("PS5", obs[c]), 3), "parts": {}}
    for key in ("E", "C", "M", "H", "matched"):
        v = d.get(key)
        if not v: continue
        R["by_start_class"][c]["parts"][key] = {"windows": v[2], "episodes": v[3], "contribution_pp": round(100 * v[0] / tot_b, 3), "ci95_contribution": ci(BC[(c, key)]),
                                                "rate_pct_of_B_on_these_days": round(100 * v[0] / v[1], 3), "ci95_rate": ci(BR[(c, key)])}
R["cohort"] = COH
json.dump(R, open(f"{OUT}/P273_DIRECTION_OF_MISMATCH" + ("" if COH == "H2-main" else "_EXT") + ".json", "w"), indent=1)
print(COH, f"check 0 {'passed' if COH == 'H2-main' else 'n/a'} (Appendix E class gains by weather-file label); check 1: {n_bitcheck} episodes bit-identical before the first mismatch; check 2 passed")
print("by observed class:", R["class_gains_by_observed_class"])
for c in "CMH":
    v = R["by_start_class"][c]; print(f"start {c}: n={v['episodes']} (with mismatch {v['episodes_with_a_mismatch']}), daily - once {v['daily_minus_once_pp']:+.2f} pp")
    for key, p in v["parts"].items():
        lab_ = f"days in {key}" if key != "matched" else "matched days after divergence"
        print(f"   {lab_:32s} windows {p['windows']:4d} ep {p['episodes']:3d} | contribution {p['contribution_pp']:+.2f} pp {p['ci95_contribution']} | rate {p['rate_pct_of_B_on_these_days']:+.2f} % {p['ci95_rate']}")
