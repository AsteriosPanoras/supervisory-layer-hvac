"""H2 analysis (PREREG_H2_v1; written and hashed BEFORE any H2 reward is read).  usage (from N10S_design): python -I p128_h2_analysis.py <main|all> <out.json>
Loss L(cell, arm) = -mean(reward[:8640]); complete case over B, PS5, PD5, TRIG2, G30, MG.  PRIMARY: MG minus TRIG2 in points of B-referenced gain, equal-cell weights, 95 % station-cluster bootstrap (3000, seed 1) over the 'main' cells (cohort H2-main); the direct loss ratio 1 - sum(L_MG)/sum(L_TRIG2) is reported next to it.  SECONDARY: MG-PD5, MG-PS5, MG-G30, PD5-PS5, class-weighted version (C .3066 / M .4348 / H .2495 renormalised), by class, by generator (GEN12 weather-shift subset vs GEN14 variant subset), harmed >1 % share and worst cell, realised switches, exactness QA (MG must equal PD5 when its day-1 observed class is M and PS5 otherwise), observed vs design class at step 228.  'all' adds the pre-frozen extension cells (pooled, reported as extension)."""
import sys, json, glob, csv, numpy as np, collections
import os
N = "."; RESD = os.environ.get("P128_RES", f"{N}/results_h2"); mode = sys.argv[1]; cells = [c for c in csv.DictReader(open(os.environ.get("P128_CELLS", f"{N}/cells_h2.csv"))) if mode == "all" or c["cohort"] == "H2-main"]; arms = json.load(open(f"{N}/h2_arms.json")); arms = {k: arms[k] for k in ("B", "PS5", "PD5", "TRIG2", "G30", "MG")}; inv = {v: k for k, v in arms.items()}
L, INFO, SW = {}, {}, {}
for f in glob.glob(f"{RESD}/h2*/**/*.summary.json", recursive=True):
    s = json.load(open(f))
    if s.get("status") != "OK" or s.get("steps_done") != 8640 or s["policy_key"] not in inv: continue
    k = (s["building"], s["weather"], inv[s["policy_key"]]); L[k] = float((-np.asarray(np.load(f[:-len(".summary.json")] + ".npz", allow_pickle=True)["rewards"], float)[:8640]).mean())
    u = s.get("unif") or []; SW[k] = sum(1 for x in u if x.get("switched")); INFO[k] = [x for x in u if x.get("day") == 1]
names = list(arms); ok = [c for c in cells if all((c["building"], c["weather"], a) in L for a in names)]; miss = [(c["building"], c["weather"]) for c in cells if c not in ok]
cl = np.array([c["cls"] for c in ok]); st = np.array([c["station"] for c in ok]); gen = np.array([c["generator"] for c in ok]); M = np.array([[L[(c["building"], c["weather"], a)] for a in names] for c in ok]); ix = {a: i for i, a in enumerate(names)}; wts = json.load(open(f"{N}/unif_weights.json"))
def W(idx, kind):
    if kind == "equal": return np.ones(len(idx))
    w = {k: wts[k] for k in "CMH"}; tw = sum(w.values()); return np.array([w[cl[i]] / tw / max(1, (cl[idx] == cl[i]).sum()) for i in idx])
def gains(idx, kind): w = W(idx, kind); t = (w[:, None] * M[idx]).sum(0); return 100 * (1 - t / t[ix["B"]])
def ci(f, groups, reps=3000, seed=1):
    rng = np.random.default_rng(seed); us = sorted(groups); b = []
    for _ in range(reps): p = rng.choice(len(us), len(us)); b.append(f(np.concatenate([groups[us[i]] for i in p])))
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
groups = {u: np.where(st == u)[0] for u in sorted(set(st))}; res = {"mode": mode, "n_design": len(cells), "n_complete": len(ok), "incomplete": miss, "stations": len(groups), "by_class_n": dict(collections.Counter(cl)), "by_generator_n": dict(collections.Counter(gen))}
pairs = (("MG", "TRIG2"), ("MG", "PD5"), ("MG", "PS5"), ("MG", "G30"), ("PD5", "PS5"), ("PD5", "TRIG2"), ("TRIG2", "PS5"), ("PD5", "G30"))
if len(ok):
    for kind in ("equal", "classw"):
        g = gains(np.arange(len(ok)), kind); d = {}
        for a, b in pairs: d[f"{a}_minus_{b}"] = {"points": round(float(g[ix[a]] - g[ix[b]]), 3), "ci": ci(lambda idx, a=a, b=b: gains(idx, kind)[ix[a]] - gains(idx, kind)[ix[b]], groups)}
        res[kind] = {"gain_vs_B": {a: {"gain": round(float(g[ix[a]]), 3), "ci": ci(lambda idx, a=a: gains(idx, kind)[ix[a]], groups)} for a in names if a != "B"}, "differences": d}
    res["PRIMARY_MG_minus_TRIG2_equal"] = res["equal"]["differences"]["MG_minus_TRIG2"]; res["direct_ratio_MG_vs_TRIG2_pct"] = round(float(100 * (1 - M[:, ix["MG"]].sum() / M[:, ix["TRIG2"]].sum())), 3)
    r = M / M[:, [ix["B"]]]; res["harmed_gt1pct_share"] = {a: round(float((r[:, ix[a]] > 1.01).mean()), 3) for a in names}; res["worst_cell_ratio"] = {a: round(float(r[:, ix[a]].max()), 3) for a in names}
    res["by_class_equal_gain"] = {k: {a: round(float(100 * (1 - M[cl == k][:, ix[a]].sum() / M[cl == k][:, ix["B"]].sum())), 2) for a in names} for k in "CMH" if (cl == k).any()}
    res["by_generator"] = {}
    for gname in ("gen12", "gen14"):
        sel = np.where(gen == gname)[0]
        if len(sel):
            gg = {u: np.where((st == u) & (gen == gname))[0] for u in sorted(set(st[sel]))}; f_ = lambda idx: gains(idx, "equal")[ix["MG"]] - gains(idx, "equal")[ix["TRIG2"]]
            res["by_generator"][gname] = {"n": int(len(sel)), "MG_minus_TRIG2": round(float(f_(sel)), 3), "ci": ci(f_, gg)}
    ex = []; flags = []; mism = 0
    for c in ok:
        b, w = c["building"], c["weather"]; inf = INFO.get((b, w, "MG")) or []; fl = bool(inf and inf[0].get("mgate_flag")); cobs = inf[0].get("class") if inf else None; flags.append(fl)
        ref = L[(b, w, "PD5")] if fl else L[(b, w, "PS5")]; ex.append(abs(L[(b, w, "MG")] - ref) / ref)
        if cobs and cobs != c["cls"]: mism += 1
    res["exactness_QA"] = {"max_rel_diff_MG_vs_selected_rule": float(max(ex)), "n_cells_not_exact_1e-9": int(sum(1 for e in ex if e > 1e-9)), "share_M_flag": round(float(np.mean(flags)), 3), "observed_class_differs_from_design_class": mism}
    res["realised_switches_mean"] = {a: round(float(np.mean([SW.get((c["building"], c["weather"], a), 0) for c in ok])), 2) for a in ("PS5", "PD5", "TRIG2", "MG")}
    # three SEPARATE estimates of the M-gate: (1) retrospective reconstruction with the DESIGN class, (2) retrospective reconstruction with the OBSERVED class at step 228 (taken from the day-1 decision of the PD5 run: PD5 if M else PS5), (3) the ACTUALLY EXECUTED MG runs (= the PRIMARY above).  (1) and (2) are diagnostics, never the primary; none is chosen afterwards.
    def recon(getcls):
        col = np.array([L[(c["building"], c["weather"], "PD5" if getcls(c) == "M" else "PS5")] for c in ok]); n_m = sum(1 for c in ok if getcls(c) == "M")
        def f(idx): w = np.ones(len(idx)); b = (w * M[idx, ix["B"]]).sum(); return 100 * ((w * M[idx, ix["TRIG2"]]).sum() - (w * col[idx]).sum()) / b
        g = lambda idx: 100 * (1 - col[idx].sum() / M[idx, ix["B"]].sum())
        return {"MG_gain_vs_B_equal": round(float(g(np.arange(len(ok)))), 3), "MG_minus_TRIG2_points": round(float(f(np.arange(len(ok)))), 3), "ci": ci(f, groups), "n_cells_classed_M": int(n_m)}
    obs_cls = lambda c: ((INFO.get((c["building"], c["weather"], "PD5")) or [{}])[0].get("class") or c["cls"])
    res["three_estimates_of_MG"] = {"1_retrospective_design_class": recon(lambda c: c["cls"]), "2_retrospective_observed_class": recon(obs_cls), "3_actual_executed_MG_PRIMARY": {"MG_gain_vs_B_equal": res["equal"]["gain_vs_B"]["MG"]["gain"], "MG_minus_TRIG2_points": res["PRIMARY_MG_minus_TRIG2_equal"]["points"], "ci": res["PRIMARY_MG_minus_TRIG2_equal"]["ci"]}, "cells_where_observed_class_differs_from_design": [[c["building"], c["weather"], c["cls"], obs_cls(c)] for c in ok if obs_cls(c) != c["cls"]]}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps({k: res.get(k) for k in ("n_complete", "stations", "PRIMARY_MG_minus_TRIG2_equal", "direct_ratio_MG_vs_TRIG2_pct", "exactness_QA", "by_generator")}, indent=0)[:1500])
