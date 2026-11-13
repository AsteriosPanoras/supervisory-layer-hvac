"""K07-X development selection (frozen in PREREG_K07X_DEV_v1 before any development run).  usage: python -I p185_k07x_dev_select.py <results_dir> <manifest.csv> out.json
manifest columns: architecture, weather, policy, policy_key.  Loss per cell = mean |reward| over 8640 steps.
J_P = R_P + 0.5 * max(0, A_P); R_P = sum L_P / sum L_B - 1 over all development cells; A_P = max over architectures of (sum_arch L_P / sum_arch L_B - 1); J_B = 0.
S = argmin J; ties within 0.0005 (0.05 percentage points) go to the earlier policy in ORDER.  Grouped CV: leave one architecture out, select on the other 8, score the choice on the left-out one.
GO (claim that a candidate beats the operational baseline): S != B and CV_mean < 0 and (folds with e_a < 0) >= 6 and CV_worst <= +0.02 and (folds selecting S) >= 6."""
import json, glob, csv, sys, numpy as np
R_DIR, MAN, OUT = sys.argv[1:4]; ORDER = ["B", "G30", "MG", "MGG30"]
man = list(csv.DictReader(open(MAN))); key2 = {(r["architecture"], r["weather"], r["policy_key"]): r for r in man}
L = {}
for f in glob.glob(f"{R_DIR}/**/*.summary.json", recursive=True):
    s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"]))
    if r and s["status"] == "OK" and s["steps_done"] == 8640 and s.get("illegal_actions", 0) == 0: L[(r["architecture"], r["weather"], r["policy"])] = float(np.abs(np.load(f.replace(".summary.json", ".npz"), allow_pickle=True)["rewards"]).mean())
cells = sorted({(r["architecture"], r["weather"]) for r in man}); complete = [c for c in cells if all((c[0], c[1], p) in L for p in ORDER)]
archs = sorted({c[0] for c in complete})
def J(sel_archs):
    cs = [c for c in complete if c[0] in sel_archs]; out = {}
    for p in ORDER:
        Rp = sum(L[(c[0], c[1], p)] for c in cs) / sum(L[(c[0], c[1], "B")] for c in cs) - 1
        Ap = max(sum(L[(c[0], c[1], p)] for c in cs if c[0] == a) / sum(L[(c[0], c[1], "B")] for c in cs if c[0] == a) - 1 for a in sel_archs)
        out[p] = {"R": Rp, "A": Ap, "J": Rp + 0.5 * max(0.0, Ap)}
    best = min(out[p]["J"] for p in ORDER); S = next(p for p in ORDER if out[p]["J"] <= best + 0.0005); return S, out
S, full = J(archs); folds = {}
for a in archs:
    s_a, _ = J([x for x in archs if x != a]); e = sum(L[(c[0], c[1], s_a)] for c in complete if c[0] == a) / sum(L[(c[0], c[1], "B")] for c in complete if c[0] == a) - 1; folds[a] = {"selected": s_a, "e_vs_B": e}
e = np.array([v["e_vs_B"] for v in folds.values()]); stab = sum(1 for v in folds.values() if v["selected"] == S)
go = bool(S != "B" and e.mean() < 0 and (e < 0).sum() >= 6 and e.max() <= 0.02 and stab >= 6)
res = {"n_cells": len(complete), "missing_cells": [c for c in cells if c not in complete], "architectures": archs, "full": full, "S": S, "folds": folds, "CV_mean": float(e.mean()), "CV_worst": float(e.max()), "folds_negative": int((e < 0).sum()), "stability": stab, "GO": go,
       "per_architecture_rel_vs_B": {a: {p: sum(L[(c[0], c[1], p)] for c in complete if c[0] == a) / sum(L[(c[0], c[1], "B")] for c in complete if c[0] == a) - 1 for p in ORDER} for a in archs}}
json.dump(res, open(OUT, "w"), indent=1); print("S =", S, "| GO =", go, "| CV_mean %.4f CV_worst %.4f folds<0 %d stability %d" % (res["CV_mean"], res["CV_worst"], res["folds_negative"], stab)); print({p: {k: round(v, 5) for k, v in full[p].items()} for p in ORDER})
