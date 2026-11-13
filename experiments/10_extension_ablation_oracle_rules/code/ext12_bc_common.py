"""Shared loader for p290 (B) and p291 (C) of PREREG_EXT12_v1.  It reads three sources, told apart by folder:
 - the K07 blind runs (G30, ROS, SUP): results_k07x3;
 - the unoccupied-setup blind runs (U, UR): results_k07uns_fresh;
 - the new Kaggle runs (SUPR, OR, URT): results_ext12_bc.
Episodes must be OK, 8640 steps, 0 illegal actions.  Loss L = mean |reward|.  Gains are pooled over the 10 composite architectures, with
a 95 % architecture-cluster bootstrap (3000 replicates, seed 1, an architecture drawn twice counted twice: the p277b logic).
Reproducibility check F-B: the SUPR re-runs (4 scenes) must equal their SUP originals bit for bit (rewards and continuous actions).
Then the reference is SUP; otherwise it is SUPR if all 36 SUPR runs exist, and otherwise there is no reference (not run)."""
import os, csv, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
SRC = {"k07": [f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"], "uns": [f"{N}/k07uns/K07UNS_FRESH_MANIFEST.csv"],
       "ext": [f"{H}/EXT12_BC_MANIFEST.csv"]}
PATS = ((f"{N}/results_k07x3/**/*.summary.json", "k07"), (f"{N}/results_k07uns_fresh/**/*.summary.json", "uns"), (f"{N}/results_ext12_bc/**/*.summary.json", "ext"))
W = {"energy": ("normalized_energy_cost", 0.2), "carbon": ("normalized_carbon_emission", 0.2), "comfort": ("normalized_productivity_regret", 0.6)}


def load():
    key2, grp = {}, {}
    for src, fs in SRC.items():
        for f in fs:
            for r in csv.DictReader(open(f, newline="")):
                key2[(r["architecture"], r["weather"], r["policy_key"], src)] = r; grp[r["architecture"]] = r["group"]
    L, RW, C, RAW = {}, {}, {}, {}
    for pat, src in PATS:
        for f in glob.glob(pat, recursive=True):
            s = json.load(open(f)); r = key2.get((s["building"], s["weather"], s["policy_key"], src))
            if not r or s["status"] != "OK" or s["steps_done"] != 8640 or s.get("illegal_actions", 0):
                continue
            k = (r["architecture"], r["weather"], r["policy"])
            if k in L: raise SystemExit(f"duplicate {k}: {f}")
            z = np.load(f.replace(".summary.json", ".npz"), allow_pickle=True); rw = np.abs(z["rewards"].astype(float))
            L[k] = float(rw.mean()); RW[k] = rw; ck = [str(x) for x in z["comp_keys"]]
            C[k] = {nm: float(np.abs(wt * z["comps"][:, ck.index(c)].astype(float)).mean()) for nm, (c, wt) in W.items()}
            if r["policy"] in ("SUP", "SUPR"): RAW[k] = (z["rewards"], z["actions_cont"])
    return L, RW, C, RAW, grp


def reference(L, RAW):
    rep = sorted({(a, w) for a, w, p in L if p == "SUPR"})
    same = {f"{a}|{w}": bool((a, w, "SUP") in RAW and np.array_equal(RAW[(a, w, "SUPR")][0], RAW[(a, w, "SUP")][0])
                              and np.array_equal(RAW[(a, w, "SUPR")][1], RAW[(a, w, "SUP")][1])) for a, w in rep}
    ok = len(rep) >= 4 and all(same.values())
    n_supr = sum(1 for k in L if k[2] == "SUPR")
    ref = "SUP" if ok else ("SUPR" if n_supr >= 36 else None)
    return ref, {"repro_scenes": same, "REPRO_OK": ok, "reference": ref}


def tools(L, grp, x0):
    cells = sorted({(a, w) for a, w, p in L if p == x0}); CO = sorted({a for a, w in cells if grp[a] == "composite"}); OOD = sorted({a for a, w in cells if grp[a] == "ood"})
    rng = np.random.default_rng(1); BOOT = [rng.integers(0, len(CO), len(CO)) for _ in range(3000)]
    def Ssum(p, archs): return sum(L[(a, w, p)] for aa in archs for a, w in cells if a == aa)
    def gain(x, y, archs=None): archs = CO if archs is None else archs; return 100 * (1 - Ssum(x, archs) / Ssum(y, archs))
    def contrast(x, y):
        ga = {a: gain(x, y, [a]) for a in CO + OOD}; reps = [gain(x, y, [CO[i] for i in b]) for b in BOOT]; ch = [a for a in CO if ga[a] != 0.0]
        return {"gain_pct": gain(x, y), "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "composite_changed": len(ch),
                "composite_changed_better": sum(1 for a in ch if ga[a] > 0), "worst_architecture_gain_pct": min(ga.values()), "ood_gain_pct": gain(x, y, OOD) if OOD else None,
                "per_architecture_gain_pct": ga}
    return cells, CO, OOD, gain, contrast
