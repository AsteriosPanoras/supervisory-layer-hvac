"""H3 planning (PREREG_H3_v1): how often would 40 cells, drawn like H3 (M 18, C 12, H 10), give each decision if the cells behaved like
the 118 complete H2 cells (main + extension)?  Draws without replacement from H2 by design class (400 draws), each with the H3 primary
and a 1000-replicate station-cluster bootstrap (the registered analysis uses 3000).  Planning only: it uses H2 outcomes, which were read
before H3 was designed, and it chooses nothing in H3.  usage: python -I h3_power.py <profile_study_2021_episodes.csv> <out.json>"""
import sys, csv, json, collections, numpy as np
rows = list(csv.DictReader(open(sys.argv[1]))); L = collections.defaultdict(dict); meta = {}
for r in rows: k = (r["building"], r["weather"]); L[k][r["arm"]] = float(r["loss"]); meta[k] = (r["station"], r["weather_file_class"])
ks = sorted(k for k in L if all(a in L[k] for a in ("B", "TRIG2", "MG"))); assert len(ks) == 118
M = np.array([[L[k]["B"], L[k]["TRIG2"], L[k]["MG"]] for k in ks]); st = np.array([meta[k][0] for k in ks]); cl = np.array([meta[k][1] for k in ks])
prim = lambda Mi: 100 * (Mi[:, 1].sum() - Mi[:, 2].sum()) / Mi[:, 0].sum()
def interval(sel, reps=1000, seed=1):
    rng = np.random.default_rng(seed); s = st[sel]; us = sorted(set(s)); gi = {u: sel[s == u] for u in us}; b = []
    for _ in range(reps): p = rng.choice(len(us), len(us)); b.append(prim(M[np.concatenate([gi[us[i]] for i in p])]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
rng = np.random.default_rng(2026); Q = {"M": 18, "C": 12, "H": 10}; out = collections.Counter(); ests = []
for _ in range(400):
    sel = np.concatenate([rng.choice(np.where(cl == k)[0], q, replace=False) for k, q in Q.items()]); e = prim(M[sel]); lo, hi = interval(sel); ests.append(e)
    out["REPLICATED" if lo > 0 and e > 0 else "NOT REPLICATED" if hi < 0.25 else "INCONCLUSIVE"] += 1
res = {"draws": 400, "H2_pooled_118": round(float(prim(M)), 3), "share": {k: round(v / 400, 3) for k, v in out.items()},
       "estimate_quantiles_5_50_95": [round(float(np.percentile(ests, q)), 3) for q in (5, 50, 95)]}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res))
