"""Reproduces the paper's main numbers from the per-episode tables of this package (no simulator needed).
usage: python reproduce_tables.py [results_dir]        (needs numpy only; default results_dir = ../results)
Every number is compared with the value printed in the paper; a mismatch is reported, not hidden.
  A. 2021 profile study (Section 5.1): the pre-specified primary contrast (one-time gate minus persistence, points of
     baseline-referenced gain, equal weights, 98 main episodes) with its 95% station-cluster interval (3,000 resamples, seed 1), and the
     gains of select-once, daily re-selection and the gate by the weather-file class (Appendix E).
  B. Blind set (Section 5.2, the learning-source table): pooled gains over the 10 composite architectures with 95% architecture-cluster intervals (3,000,
     seed 1; every drawn architecture enters with all its episodes as often as drawn) and the number of composite architectures improved.
  C. Section 5.3: Spearman correlation of the episode gain with the comfort share of the controller beneath (36 episodes).
  D. Appendix J: the unoccupied setup on the blind set."""
import csv, os, sys, collections
import numpy as np
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
bad = []
def check(name, value, printed, nd=2):
    ok = f"{value:.{nd}f}" == printed; print(f"{'ok ' if ok else 'XX '} {name:60s} {value:+.{nd}f}  (paper: {printed})")
    if not ok: bad.append(name)
# ---------------- A. 2021 profile study ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "profile_study_2021_episodes.csv"))))
cells, seen = [], set()
for r in rows:
    k = (r["building"], r["weather"])
    if r["cohort"] == "H2-main" and k not in seen: seen.add(k); cells.append((k, r["station"], r["weather_file_class"]))
arms = ["B", "PS5", "PD5", "TRIG2", "G30", "MG"]; ix = {a: i for i, a in enumerate(arms)}
L = {(r["building"], r["weather"], r["arm"]): float(r["loss"]) for r in rows}
M = np.array([[L[(*k, a)] for a in arms] for k, _, _ in cells]); st = np.array([s for _, s, _ in cells]); cl = np.array([c for _, _, c in cells])
def gains(idx): t = M[idx].sum(0); return 100 * (1 - t / t[ix["B"]])
def ci(f, groups, reps=3000, seed=1):                     # p128: rng.choice over the sorted station names, all episodes of a station kept
    rng = np.random.default_rng(seed); us = sorted(groups); b = []
    for _ in range(reps): p = rng.choice(len(us), len(us)); b.append(f(np.concatenate([groups[us[i]] for i in p])))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
groups = {u: np.where(st == u)[0] for u in sorted(set(st))}
prim = lambda idx: gains(idx)[ix["MG"]] - gains(idx)[ix["TRIG2"]]
print(f"A. 2021 profile study: {len(cells)} main episodes, {len(groups)} stations")
v = prim(np.arange(len(cells))); lo, hi = ci(prim, groups)
check("primary: one-time gate minus persistence (pp)", v, "0.504", 3); check("  95% interval, lower", lo, "0.337", 3); check("  95% interval, upper", hi, "0.704", 3)
for c, (ps, pd, mg) in (("C", ("4.96", "4.63", "5.08")), ("M", ("1.81", "4.40", "4.41")), ("H", ("2.71", "2.44", "2.89"))):
    g = gains(np.where(cl == c)[0])
    check(f"after {c} starts (weather-file class): select-once", g[ix["PS5"]], ps); check(f"after {c} starts: daily re-selection", g[ix["PD5"]], pd); check(f"after {c} starts: gate", g[ix["MG"]], mg)
# ---------------- B. blind set ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "blind_set_episodes.csv"))))
LB = {(r["controller"], r["architecture"], r["weather"], r["arm"]): float(r["loss"]) for r in rows}; CW = {(r["controller"], r["architecture"], r["weather"], r["arm"]): float(r["comfort_w"]) for r in rows}
grp = {r["architecture"]: r["group"] for r in rows}
CO = sorted({a for a in grp if grp[a] == "composite"}); OOD = sorted({a for a in grp if grp[a] == "ood"})
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]      # p254
def cells_of(ctrl, x): return sorted({(a, w) for c, a, w, p in LB if c == ctrl and p == x})
def gain(ctrl, x, y, archs):
    cs = cells_of(ctrl, x); num = sum(LB[(ctrl, a, w, x)] for aa in archs for a, w in cs if a == aa); den = sum(LB[(ctrl, a, w, y)] for aa in archs for a, w in cs if a == aa)
    return 100 * (1 - num / den)
print("\nB. blind set (pooled over the 10 composite architectures)")
for ctrl, x, y, name, pv, plo, phi, pb in (("K07", "G30", "B", "tuning: G30 vs B", "8.62", "6.01", "11.41", 10), ("K07", "SUP", "G30", "method: G30 + layer vs G30", "2.82", "1.48", "3.94", 9),
                                           ("K07", "SUP", "B", "combination: G30 + layer vs B", "11.19", "7.78", "14.52", 10), ("K07", "ROS", "ROSFIX", "learning: learned vs fixed start", "0.95", "0.11", "1.82", 8),
                                           ("K07", "SUP", "ROS", "pressure reset over learned start", "1.58", "0.58", "2.48", 9), ("Penguin", "PC", "V0", "Penguin + learned precooling vs Penguin", "1.88", "1.12", "3.07", 10),
                                           ("ZT Best", "ZS", "ZB", "ZT Best + layer vs ZT Best", "5.71", "3.64", "7.55", 10)):
    g = gain(ctrl, x, y, CO); reps = [gain(ctrl, x, y, [CO[i] for i in b]) for b in BOOT]; better = sum(1 for a in CO if gain(ctrl, x, y, [a]) > 0)
    check(name, g, pv); check("  95% interval, lower", float(np.percentile(reps, 2.5)), plo); check("  95% interval, upper", float(np.percentile(reps, 97.5)), phi)
    ok = better == pb; print(f"{'ok ' if ok else 'XX '}   composite architectures improved: {better} (paper: {pb})"); bad += [] if ok else [name + " better"]
# ---------------- C. the comfort-share relation ----------------
rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
print("\nC. gain against the comfort share of the controller beneath (36 episodes)")
for ctrl, x, y, pv in (("K07", "SUP", "G30", "0.80"), ("ZT Best", "ZS", "ZB", "0.93")):
    cs = cells_of(ctrl, x); gg = [100 * (1 - LB[(ctrl, a, w, x)] / LB[(ctrl, a, w, y)]) for a, w in cs]
    sh = [CW[(ctrl, a, w, y)] / sum(float(r[k]) for r in rows if (r["controller"], r["architecture"], r["weather"], r["arm"]) == (ctrl, a, w, y) for k in ("comfort_w", "energy_w", "carbon_w")) for a, w in cs]
    check(f"Spearman rho, {ctrl}", float(np.corrcoef(rank(gg), rank(sh))[0, 1]), pv)
# ---------------- D. unoccupied setup ----------------
print("\nD. unoccupied setup on the blind set (pooled over the composite architectures)")
LB.update({("K07U", a, w, p): v for (c, a, w, p), v in LB.items() if c == "K07"})
for x, y, pv in (("U", "G30", "-0.62"), ("UR", "U", "1.87"), ("UR", "G30", "1.27"), ("ROS", "G30", "1.25")):
    check(f"{x} vs {y}", gain("K07", x, y, CO), pv)
print("\nALL NUMBERS REPRODUCED" if not bad else f"\nMISMATCHES: {bad}")
