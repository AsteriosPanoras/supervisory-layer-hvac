"""Results analyses for paper v48_6 on FROZEN, already-read results (nothing new is run; descriptive / post hoc, labelled as such).  usage: python -I p254_results_analyses.py <out_dir>
A. Blind set, tests per contrast: architecture-cluster bootstrap p-value (two-sided, from the 3000 replicates of p251's scheme, seed 1), Wilcoxon signed-rank and sign test
   on the 10 composite per-architecture gains; Holm over the secondary contrasts.  Primary = layer vs G30 (SUP_vs_G30).
B. Robustness of the primary: leave-one-architecture-out range of the pooled gain; mean of per-architecture gains; per-episode better/worse/equal counts.
C. Per-architecture gains of the main contrasts (table for the appendix and the per-architecture figure).
D. Learning dynamics: per blind episode, the learned tau after each informative morning (ROS update events), mornings until tau stays within 10 % of its final value.
E. Worst cases: blind episodes where the layer lost more than 0.1 % vs G30 (activity and component change); held-out composite architectures that lost (activity and
   component change per episode), to tell inactivity from a small loss where the layer acted.
F. Where the gain comes from, occupied vs unoccupied steps: weighted components for layer vs G30 and Penguin precooling vs Penguin.
G. Profile study (v47 Table 3, a co-author's request): the four secondary contrasts on the 98 main 2021 episodes with the same station-cluster bootstrap as p128 (3000, seed 1),
   p-values from the replicates, and Holm."""
import sys, os, csv, glob, json, math, collections, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}
def comps(npz):
    ck = [str(x) for x in npz["comp_keys"]]; c = {k: npz["comps"][:, ck.index(k)].astype(float) for k in ck}; occ = c["total_occupancy"] > 0
    return {nm: (float(np.abs(wt * c[k])[occ].sum()), float(np.abs(wt * c[k])[~occ].sum())) for nm, (k, wt) in W.items()}
# ---------------- blind set ----------------
MAN = [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv", f"{N}/k07x3h/K07X3H_RUN_MANIFEST.csv", f"{N}/k07supw2/K07SUPW2_FRESH_MANIFEST.csv", f"{N}/pgf/PGF_RUN_MANIFEST.csv"]
man = [r for f in MAN for r in csv.DictReader(open(f))]; key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
grp = {r["architecture"]: r["group"] for r in man}; cls = {(r["architecture"], r["weather"]): r["day1_class"] for r in man}
L, S, F = {}, {}, {}
for d in ("results_k07x3", "results_k07x3h", "results_k07supw2_fresh", "results_pgf"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
        if not r or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0): continue
        k = (r["architecture"], r["weather"], r["policy"]); L[k] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); S[k] = s; F[k] = f
cells = sorted({(a, w) for a, w, p in L}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
def Ssum(p, archs): return sum(L[(a, w, p)] for a, w in cells if a in archs)
def gain(x, y, archs): return 100 * (1 - Ssum(x, archs) / Ssum(y, archs))
def wilcoxon_p(d):
    d = [x for x in d if x != 0]; n = len(d)
    if n == 0: return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ranks for ties
    ab = np.abs(d);
    for v in set(ab):
        idx = [i for i in range(n) if ab[i] == v]; r[idx] = np.mean(r[idx])
    wpos = float(sum(r[i] for i in range(n) if d[i] > 0))
    # exact two-sided p by enumeration of sign patterns (n <= 12)
    tot = 0; cnt = 0; obs = min(wpos, n * (n + 1) / 2 - wpos)
    for m in range(2 ** n):
        w = sum(r[i] for i in range(n) if (m >> i) & 1); tot += 1; cnt += min(w, n * (n + 1) / 2 - w) <= obs + 1e-12
    return cnt / tot
def sign_p(d):
    d = [x for x in d if x != 0]; n = len(d); k = sum(1 for x in d if x > 0); m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0
PAIRS = [("SUP", "G30"), ("G30", "B"), ("SUP", "B"), ("ROS", "G30"), ("ROSFIX", "G30"), ("ROS", "ROSFIX"), ("SUP", "ROS"), ("SUPW2", "SUP"), ("PC", "V0")]
A = {}
for x, y in PAIRS:
    reps = np.array([gain(x, y, [CO[i] for i in b]) for b in BOOT]); g = gain(x, y, CO); per = [gain(x, y, [a]) for a in CO]
    A[f"{x}_vs_{y}"] = {"gain_pct": g, "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "p_boot": float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))),
                        "p_wilcoxon": wilcoxon_p(per), "p_sign": sign_p(per), "better_of_10": sum(1 for v in per if v > 0), "worse_of_10": sum(1 for v in per if v < 0)}
SEC = ["ROS_vs_G30", "ROSFIX_vs_G30", "ROS_vs_ROSFIX", "SUP_vs_ROS", "SUPW2_vs_SUP", "PC_vs_V0"]
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
for name, key in (("p_holm_boot", "p_boot"), ("p_holm_wilcoxon", "p_wilcoxon")):
    adj = holm([A[k][key] for k in SEC])
    for k, v in zip(SEC, adj): A[k][name] = v
# B. robustness of the primary
loo = [gain("SUP", "G30", [a for a in CO if a != b]) for b in CO]
per_ep = [100 * (1 - L[(a, w, "SUP")] / L[(a, w, "G30")]) for a, w in cells]
B = {"pooled_gain_pct": gain("SUP", "G30", CO), "leave_one_architecture_out_min_max": [min(loo), max(loo)], "mean_of_architecture_gains_pct": float(np.mean([gain("SUP", "G30", [a]) for a in CO])),
     "episodes_better_worse_equal": [sum(1 for v in per_ep if v > 0), sum(1 for v in per_ep if v < 0), sum(1 for v in per_ep if v == 0)], "episodes_worse_than_0p1pct": sum(1 for v in per_ep if v < -0.1)}
# C. per-architecture gains
C = {a: {"group": grp[a], **{f"{x}_vs_{y}": gain(x, y, [a]) for x, y in PAIRS}} for a in CO + OOD}
# D. learning dynamics
D = {}
for a, w in cells:
    lg = (S[(a, w, "SUP")].get("ros") or {}).get("log", []); taus = [e["tau_after_h"] for e in lg if e["event"] == "update" and e.get("n_inf") is not None]
    inf = [e for e in lg if e["event"] == "update"]; tr = [e["tau_after_h"] for e in inf]; ninf = [e["n_inf"] for e in inf]
    seq = [t for t, n0, n1 in zip(tr, [0] + ninf[:-1], ninf) if n1 > n0]   # tau after each INFORMATIVE morning
    if seq:
        fin = seq[-1]; k = next(i for i in range(len(seq)) if all(abs(math.log(t / fin)) < math.log(1.1) for t in seq[i:]))
        D[f"{a}|{w}"] = {"informative_mornings": len(seq), "tau_seq_h": [round(t, 3) for t in seq], "mornings_to_within_10pct": k + 1}
mk = [v["mornings_to_within_10pct"] for v in D.values()]
D_sum = {"episodes": len(D), "informative_mornings_median": float(np.median([v["informative_mornings"] for v in D.values()])), "mornings_to_within_10pct_median": float(np.median(mk)),
         "mornings_to_within_10pct_p75": float(np.percentile(mk, 75)), "share_within_10pct_by_morning_5": float(np.mean([m <= 5 for m in mk]))}
# E. worst cases (blind) and held-out composite losses
def activity(s):
    lg = (s.get("ros") or {}).get("log", []); tr = s.get("tr") or {}; ps = sum(tr.get("pi_steps") or [0])
    return {"los_extensions": sum(1 for e in lg if e["event"] == "start" and e.get("why") == "need"), "tr_override_share": (sum(tr.get("override_steps") or [0]) / ps) if ps else 0.0, "frozen": (s.get("ros") or {}).get("frozen")}
def comp_delta(fx, fy):
    cx, cy = comps(np.load(fx.replace(".summary.json", ".npz"), allow_pickle=True)), comps(np.load(fy.replace(".summary.json", ".npz"), allow_pickle=True)); tot = sum(sum(v) for v in cy.values())
    return {nm: {"occupied": 100 * (cx[nm][0] - cy[nm][0]) / tot, "unoccupied": 100 * (cx[nm][1] - cy[nm][1]) / tot} for nm in W}
E = {"blind_episodes_worse_than_0p1pct": []}
for a, w in cells:
    g = 100 * (1 - L[(a, w, "SUP")] / L[(a, w, "G30")])
    if g < -0.1: E["blind_episodes_worse_than_0p1pct"].append({"episode": f"{a}|{w}", "class": cls[(a, w)], "gain_pct": g, **activity(S[(a, w, "SUP")]), "component_change_pct": comp_delta(F[(a, w, "SUP")], F[(a, w, "G30")])})
coh = json.load(open(f"{N}/k07x/K07X_V2_COHORT.json"))["cells"]; HO = {(c["architecture"], c["weather"]): c for c in coh if c["split"] == "heldout"}
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
KH = {f"k07@{G}": "G30", f"k07sup@{G};ROS=1,TR=1": "SUP"}; LH, SH, FH = {}, {}, {}
for d in ("results_k07x2", "results_k07sup"):
    for f in glob.glob(f"{N}/{d}/**/*.summary.json", recursive=True):
        s = json.load(open(f)); p = KH.get(s["policy_key"]); c = (s["building"], s["weather"])
        if p is None or c not in HO or s["status"] != "OK" or s["steps_done"] != 8640: continue
        LH[(*c, p)] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean()); SH[(*c, p)] = s; FH[(*c, p)] = f
E["heldout_episodes"] = []
for c in sorted(HO):
    if (*c, "SUP") in LH and (*c, "G30") in LH:
        E["heldout_episodes"].append({"episode": f"{c[0]}|{c[1]}", "group": HO[c]["group"], "gain_pct": 100 * (1 - LH[(*c, "SUP")] / LH[(*c, "G30")]), **activity(SH[(*c, "SUP")]),
                                      "component_change_pct": comp_delta(FH[(*c, "SUP")], FH[(*c, "G30")])})
# F. occupied vs unoccupied
def occ_split(x, y):
    acc = collections.defaultdict(float); tot = 0.0
    for a, w in cells:
        if a not in CO: continue
        cx = comps(np.load(F[(a, w, x)].replace(".summary.json", ".npz"), allow_pickle=True)); cy = comps(np.load(F[(a, w, y)].replace(".summary.json", ".npz"), allow_pickle=True)); tot += sum(sum(v) for v in cy.values())
        for nm in W: acc[f"{nm}_occupied"] += cx[nm][0] - cy[nm][0]; acc[f"{nm}_unoccupied"] += cx[nm][1] - cy[nm][1]
    return {k: 100 * v / tot for k, v in acc.items()}
Fo = {"SUP_vs_G30": occ_split("SUP", "G30"), "PC_vs_V0": occ_split("PC", "V0")}
# G. profile study (v47 Table 3) p-values from replicates
os.environ.setdefault("P128_RES", f"{N}/results_h2"); cells_h2 = [c for c in csv.DictReader(open(f"{N}/cells_h2.csv")) if c["cohort"] == "H2-main"]
arms = json.load(open(f"{N}/h2_arms.json")); arms = {k: arms[k] for k in ("B", "PS5", "PD5", "TRIG2", "G30", "MG")}; inv = {v: k for k, v in arms.items()}   # exactly as p128
LL = {}
for f in glob.glob(f"{N}/results_h2/h2*/**/*.summary.json", recursive=True):
    s = json.load(open(f))
    if s.get("status") != "OK" or s.get("steps_done") != 8640 or s["policy_key"] not in inv: continue
    LL[(s["building"], s["weather"], inv[s["policy_key"]])] = float((-np.asarray(np.load(f[:-len(".summary.json")] + ".npz", allow_pickle=True)["rewards"], float)[:8640]).mean())
names = list(arms); ok = [c for c in cells_h2 if all((c["building"], c["weather"], a) in LL for a in names)]
st = np.array([c["station"] for c in ok]); M = np.array([[LL[(c["building"], c["weather"], a)] for a in names] for c in ok]); ix = {a: i for i, a in enumerate(names)}
groups = {u: np.where(st == u)[0] for u in sorted(set(st))}; us = sorted(groups)
def delta(idx, a, b): t = M[idx].sum(0); return 100 * (t[ix[b]] - t[ix[a]]) / t[ix["B"]]
Gres = {"n_complete": len(ok), "stations": len(us)}
pv = {}
for a, b in (("MG", "PD5"), ("MG", "PS5"), ("MG", "G30"), ("PD5", "PS5")):
    r = np.random.default_rng(1); reps = []
    for _ in range(3000):
        p = r.choice(len(us), len(us)); reps.append(delta(np.concatenate([groups[us[i]] for i in p]), a, b))
    reps = np.array(reps); est = delta(np.arange(len(ok)), a, b)
    pv[f"{a}-{b}"] = {"delta_pp": est, "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "p_boot": float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()))),
                      "reps_le_0": int((reps <= 0).sum())}
adj = holm([v["p_boot"] for v in pv.values()])
for (k, v), q in zip(pv.items(), adj): v["p_holm_boot"] = q
Gres["secondary"] = pv
res = {"A_tests_blind": A, "B_robustness_primary": B, "C_per_architecture_gain_pct": C, "D_learning_dynamics": {"summary": D_sum, "per_episode": D}, "E_worst_cases": E,
       "F_components_occupied_unoccupied_pct_of_ref_loss": Fo, "G_profile_study_secondary_bootstrap_p": Gres,
       "note": "post hoc analyses of frozen, already-read results; the pre-registered rules and verdicts are unchanged"}
json.dump(res, open(f"{OUT}/PAPER_ANALYSES_3.json", "w"), indent=1)
f2 = lambda v: f"{v:+.2f}"; fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
md = ["## A. Blind set: tests (10 composite architectures)", "| Contrast | Gain | 95% CI | p (bootstrap) | p (Wilcoxon) | p (sign) | better/worse of 10 | Holm (boot) |", "|---|---|---|---|---|---|---|---|"]
for k, v in A.items(): md.append(f"| {k} | {f2(v['gain_pct'])} | [{f2(v['ci95'][0])}, {f2(v['ci95'][1])}] | {fp(v['p_boot'])} | {fp(v['p_wilcoxon'])} | {fp(v['p_sign'])} | {v['better_of_10']}/{v['worse_of_10']} | {fp(v['p_holm_boot']) if 'p_holm_boot' in v else '(primary)' if k == 'SUP_vs_G30' else '-'} |")
md += ["", "## B. Robustness of the primary (layer vs G30)", f"- pooled {f2(B['pooled_gain_pct'])} %; leave-one-architecture-out [{f2(B['leave_one_architecture_out_min_max'][0])}, {f2(B['leave_one_architecture_out_min_max'][1])}]; mean of architecture gains {f2(B['mean_of_architecture_gains_pct'])} %",
       f"- episodes better/worse/equal: {B['episodes_better_worse_equal']}; worse than -0.1 %: {B['episodes_worse_than_0p1pct']}", "",
       "## D. Learning dynamics", f"- {D_sum}", "", "## E. Worst cases"]
for e in E["blind_episodes_worse_than_0p1pct"]: md.append(f"- blind {e['episode']} ({e['class']}): gain {f2(e['gain_pct'])} %, LOS extensions {e['los_extensions']}, T&R share {100 * e['tr_override_share']:.1f} %, components {json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in e['component_change_pct'].items()})}")
for e in E["heldout_episodes"]: md.append(f"- held-out {e['episode']} ({e['group']}): gain {f2(e['gain_pct'])} %, LOS extensions {e['los_extensions']}, T&R share {100 * e['tr_override_share']:.2f} %, frozen {e['frozen']}")
md += ["", "## F. Occupied vs unoccupied (% of reference loss)", f"- {json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in Fo.items()})}", "", "## G. Profile study secondary contrasts (station bootstrap, replicate p-values)",
       "| Contrast | Δ (pp) | 95% CI | p (bootstrap) | Holm |", "|---|---|---|---|---|"] + [f"| {k} | {v['delta_pp']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {fp(v['p_boot'])} | {fp(v['p_holm_boot'])} |" for k, v in pv.items()]
open(f"{OUT}/analyses_3.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
