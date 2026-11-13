"""p296: H3 analysis (PREREG_H3_v1; written and hashed BEFORE any H3 run).  usage: python -I p296_h3_analysis.py <out.json>
Loss L(cell, arm) = -mean(reward[:8640]) from results_h3 (summaries with status OK and 8640 steps); complete case over B, TRIG2, MG.
PRIMARY = the H2 primary, computed with the functions of p128_h2_analysis.py copied unchanged: MG minus TRIG2 in points of B-referenced
gain, equal-cell weights (gain(a) = 100 (1 - sum_cells L_a / sum_cells L_B)), 95 % percentile interval of 3000 station-cluster bootstrap
resamples (seed 1), over the complete H3 cells.
DECISION (PREREG_H3_v1): REPLICATED if the lower bound > 0 and the estimate > 0; NOT REPLICATED if the upper bound < +0.25 points (half of the
H2 main estimate, +0.504); INCONCLUSIVE otherwise.
SECONDARY (no multiplicity claim, reported whatever their sign):
 (1) per-cell mean of 100 (L_TRIG2 - L_MG) / L_B, same bootstrap; counts of cells with MG better / equal (|d| < 1e-9) / worse;
 (2) by design class (pooled within the class) and by generator (with intervals);
 (3) the largest single-cell share of sum L_B, and the primary without that cell (sensitivity only);
 (4) pooled with H2: the 118 complete H2 cells (complete over its six arms, as p128 'all'; the registered +0.407 is reproduced first as a
     check) plus the H3 cells, same estimator, stations as clusters across both studies;
 (5) the share of H3 cells whose MG run raised the day-1 gate flag, and design vs observed day-1 class.
The script prints only the summary written to <out.json>."""
import sys, os, json, glob, csv, collections, numpy as np
N = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]
A2 = json.load(open(f"{N}/h2_arms.json")); A3 = json.load(open(f"{N}/h3_arms.json")); assert all(A3[a] == A2[a] for a in A3) and list(A3) == ["B", "TRIG2", "MG"]
def load(resd, pat, cells, arms):
    inv = {v: k for k, v in arms.items()}; L, INFO = {}, {}
    for f in glob.glob(f"{resd}/{pat}/**/*.summary.json", recursive=True):
        s = json.load(open(f))
        if s.get("status") != "OK" or s.get("steps_done") != 8640 or s["policy_key"] not in inv: continue
        k = (s["building"], s["weather"], inv[s["policy_key"]]); L[k] = float((-np.asarray(np.load(f[:-len(".summary.json")] + ".npz", allow_pickle=True)["rewards"], float)[:8640]).mean())
        u = s.get("unif") or []; INFO[k] = [x for x in u if x.get("day") == 1]
    ok = [c for c in cells if all((c["building"], c["weather"], a) in L for a in arms)]
    return L, INFO, ok
names = ["B", "TRIG2", "MG"]; ix = {a: i for i, a in enumerate(names)}
# ---- functions of p128_h2_analysis.py, unchanged (equal-cell weights) ----
def gains_of(M):
    def gains(idx): t = M[idx].sum(0); return 100 * (1 - t / t[ix["B"]])
    return gains
def ci(f, groups, reps=3000, seed=1):
    rng = np.random.default_rng(seed); us = sorted(groups); b = []
    for _ in range(reps): p = rng.choice(len(us), len(us)); b.append(f(np.concatenate([groups[us[i]] for i in p])))
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
def block(ok, L):
    M = np.array([[L[(c["building"], c["weather"], a)] for a in names] for c in ok]); st = np.array([c["station"] for c in ok])
    groups = {u: np.where(st == u)[0] for u in sorted(set(st))}; g = gains_of(M); prim = lambda idx: g(idx)[ix["MG"]] - g(idx)[ix["TRIG2"]]
    return M, st, groups, prim
C3 = list(csv.DictReader(open(f"{N}/cells_h3.csv"))); assert len(C3) == 40
L3, I3, ok3 = load(f"{N}/results_h3", "h3*", C3, A3)
res = {"n_design": len(C3), "n_complete": len(ok3), "incomplete": [(c["building"], c["weather"]) for c in C3 if c not in ok3]}
if ok3:
    M, st, groups, prim = block(ok3, L3); idx = np.arange(len(ok3)); g = gains_of(M)(idx)
    est = round(float(prim(idx)), 3); iv = ci(prim, groups)
    res["stations"] = len(groups); res["gain_vs_B"] = {a: {"gain": round(float(g[ix[a]]), 3), "ci": ci(lambda i, a=a: gains_of(M)(i)[ix[a]], groups)} for a in ("TRIG2", "MG")}
    res["PRIMARY_MG_minus_TRIG2"] = {"points": est, "ci": iv}
    res["DECISION"] = "REPLICATED" if (iv[0] > 0 and est > 0) else "NOT REPLICATED" if iv[1] < 0.25 else "INCONCLUSIVE"
    d = 100 * (M[:, ix["TRIG2"]] - M[:, ix["MG"]]) / M[:, ix["B"]]
    res["S1_per_cell_mean"] = {"points": round(float(d.mean()), 3), "ci": ci(lambda i: float(d[i].mean()), groups),
                               "MG_better": int((d > 1e-9).sum()), "equal": int((np.abs(d) <= 1e-9).sum()), "MG_worse": int((d < -1e-9).sum())}
    cl = np.array([c["cls"] for c in ok3]); gen = np.array([c["generator"] for c in ok3])
    res["S2_by_class"] = {k: {"n": int((cl == k).sum()), "MG_minus_TRIG2": round(float(prim(np.where(cl == k)[0])), 3)} for k in "CMH" if (cl == k).any()}
    res["S2_by_generator"] = {}
    for gname in ("gen12", "gen14"):
        sel = np.where(gen == gname)[0]
        if len(sel): res["S2_by_generator"][gname] = {"n": int(len(sel)), "MG_minus_TRIG2": round(float(prim(sel)), 3), "ci": ci(prim, {u: np.where((st == u) & (gen == gname))[0] for u in sorted(set(st[sel]))})}
    share = M[:, ix["B"]] / M[:, ix["B"]].sum(); j = int(share.argmax()); rest = np.array([i for i in idx if i != j])
    res["S3_influence"] = {"largest_share_pct": round(float(100 * share[j]), 2), "cell": [ok3[j]["building"], ok3[j]["weather"], ok3[j]["cls"]],
                           "primary_without_it": round(float(prim(rest)), 3)}
    fl = [bool((I3.get((c["building"], c["weather"], "MG")) or [{}])[0].get("mgate_flag")) for c in ok3]
    obs = [(I3.get((c["building"], c["weather"], "MG")) or [{}])[0].get("class") for c in ok3]
    res["S5_gate"] = {"share_flagged_M": round(float(np.mean(fl)), 3), "design_class_M_share": round(float(np.mean(cl == "M")), 3),
                      "observed_class_differs_from_design": int(sum(1 for o, c in zip(obs, ok3) if o and o != c["cls"]))}
# (4) pooled with H2 (all 118 complete cells over its six arms, as p128 'all')
C2 = list(csv.DictReader(open(f"{N}/cells_h2.csv"))); A2six = {k: A2[k] for k in ("B", "PS5", "PD5", "TRIG2", "G30", "MG")}
L2, _, ok2 = load(f"{N}/results_h2", "h2*", C2, A2six)
M2, st2, gr2, prim2 = block(ok2, L2); h2all = round(float(prim2(np.arange(len(ok2)))), 3)
res["S4_check_H2_all_reproduced"] = {"n": len(ok2), "points": h2all, "registered": 0.407, "ok": bool(len(ok2) == 118 and abs(h2all - 0.407) < 0.0006)}
if ok3:
    okp = ok2 + ok3; Lp = {**L2, **L3}; Mp, stp, grp, primp = block(okp, Lp)
    res["S4_pooled_H2_and_H3"] = {"n": len(okp), "stations": len(grp), "points": round(float(primp(np.arange(len(okp)))), 3), "ci": ci(primp, grp)}
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res, indent=0)[:2500])
