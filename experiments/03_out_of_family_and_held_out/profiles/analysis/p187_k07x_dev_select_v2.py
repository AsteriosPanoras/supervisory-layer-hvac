"""K07-X v2 development selection (to be frozen in PREREG_K07X_DEV_v2 before any development run).  usage: python -I p187_k07x_dev_select_v2.py <results_dir> <dev_manifest.csv> out.json
manifest columns: architecture, group (composite|ood), weather, policy, policy_key.  Loss per cell = mean |reward| over 8640 steps; r[a,P] = sum over a's strips L_P / sum L_B - 1.
Target = in-distribution composite: R_P = mean over composite a of r[a,P]; J_P = R_P + 0.5 * max(0, worst composite r[a,P]); OOD guard (eligibility): every OOD r[a,P] <= +0.02.
S = argmin J over eligible policies (B always eligible, J_B = 0); ties within 0.0005 go to the earlier of B, G30, MG, MGG30.
Grouped CV: leave one COMPOSITE architecture out (select on the other composite ones with the same guard), score on the left-out one.
GO: S != B, CV mean < 0, >= 4 of 6 folds negative, CV worst <= +0.02, S chosen in >= 4 of 6 folds, OOD guard met by S."""
import json, glob, csv, sys, numpy as np
R_DIR, MAN, OUT = sys.argv[1:4]; ORDER = ["B", "G30", "MG", "MGG30"]
man = list(csv.DictReader(open(MAN))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}; grp = {r["architecture"]: r["group"] for r in man}
L = {}
for f in glob.glob(f"{R_DIR}/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if r and s["status"] == "OK" and s["steps_done"] == 8640 and s.get("illegal_actions", 0) == 0: L[(r["architecture"], r["weather"], r["policy"])] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
cells = sorted({(r["architecture"], r["weather"]) for r in man}); ok = [c for c in cells if all((c[0], c[1], p) in L for p in ORDER)]
archs = sorted({c[0] for c in ok}); comp = [a for a in archs if grp[a] == "composite"]; ood = [a for a in archs if grp[a] == "ood"]
r = {(a, p): sum(L[(c[0], c[1], p)] for c in ok if c[0] == a) / sum(L[(c[0], c[1], "B")] for c in ok if c[0] == a) - 1 for a in archs for p in ORDER}
def select(cset):
    tab = {}
    for p in ORDER:
        R = float(np.mean([r[(a, p)] for a in cset])); Aw = max(r[(a, p)] for a in cset); guard = all(r[(a, p)] <= 0.02 for a in ood) or p == "B"
        tab[p] = {"R_composite": R, "worst_composite": Aw, "J": R + 0.5 * max(0.0, Aw) if p != "B" else 0.0, "ood_guard_met": guard, "ood_r": {a: r[(a, p)] for a in ood}}
    el = [p for p in ORDER if tab[p]["ood_guard_met"]]; best = min(tab[p]["J"] for p in el); return next(p for p in el if tab[p]["J"] <= best + 0.0005), tab
S, tab = select(comp); folds = {}
for a in comp:
    s_a, _ = select([x for x in comp if x != a]); folds[a] = {"selected": s_a, "e_vs_B": r[(a, s_a)]}
e = np.array([v["e_vs_B"] for v in folds.values()]) if folds else np.array([0.0]); stab = sum(1 for v in folds.values() if v["selected"] == S)
reasons = {"S_not_B": S != "B", "CV_mean_lt_0": bool(e.mean() < 0), "folds_negative_ge_4": int((e < 0).sum()) >= 4, "CV_worst_le_2pct": bool(e.max() <= 0.02), "stability_ge_4": stab >= 4, "ood_guard": tab[S]["ood_guard_met"]}
res = {"n_cells": len(ok), "missing_cells": [c for c in cells if c not in ok], "composite": comp, "ood": ood, "table": tab, "S": S, "folds": folds, "CV_mean": float(e.mean()), "CV_worst": float(e.max()), "folds_negative": int((e < 0).sum()), "stability": stab,
       "GO": all(reasons.values()), "GO_conditions": reasons, "r_per_architecture": {a: {p: r[(a, p)] for p in ORDER} for a in archs}}
json.dump(res, open(OUT, "w"), indent=1); print("S =", S, "| GO =", res["GO"], reasons); print({p: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in tab[p].items() if k != "ood_r"} for p in ORDER})
