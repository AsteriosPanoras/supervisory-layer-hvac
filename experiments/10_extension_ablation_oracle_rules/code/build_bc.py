"""Builds experiments B and C of PREREG_EXT12_v1 for Kaggle: the manifest, the private experiments dataset folder ext12/exp_ds
(a copy of k07uns/exp_ds: the same runner v9supe e1b7921f, kernel runner, K07 classes and KRE-v2 overlay) and the job files.
 - e0: 4 SUPR jobs, which re-run the exact SUP key on 4 scenes (reproducibility check F-B).
 - e19..e26 (fallback, only if F-B fails): SUPR on the other 32 scenes.
 - e1..e18: 36 OR jobs (SUP key + TAU0 = that episode's final tau from its SUP run) and 36 URT jobs (SUP key + UNS_COOL_K=305.4),
   4 per kernel.
The scenes are the 36 of the K07 blind test (Kaggle dataset sb26-k07x3-scenarios), 8640 steps.  usage: python3 build_bc.py"""
import csv, glob, json, os, shutil
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
man = []
for f in (f"{N}/k07x3run/K07X3_RUN_MANIFEST.csv", f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"):
    man += list(csv.DictReader(open(f, newline="")))
sup = [r for r in man if r["policy"] == "SUP"]; assert len(sup) == 36 and all(r["policy_key"].endswith(";ROS=1,TR=1") for r in sup)
key = {(r["architecture"], r["weather"]): r["policy_key"] for r in sup}
tau = {}
for f in glob.glob(f"{N}/results_k07x3/**/*.summary.json", recursive=True):
    s = json.load(open(f)); k = (s["building"], s["weather"])
    if k in key and s["policy_key"] == key[k] and s["status"] == "OK" and s["steps_done"] == 8640:
        assert k not in tau, k; tau[k] = float(s["ros"]["tau_final_h"])
assert len(tau) == 36 and all(0.3 < v < 3.0 for v in tau.values())
rows, jobs = [], {"e0": []}
repro = [r for r in sup if r["group"] == "composite"][:4]
for r in repro:
    rows.append({**r, "policy": "SUPR", "kernel": "e0"}); jobs["e0"].append(f'{r["architecture"]} {r["weather"]} {r["policy_key"]} 8640')
flat = []
for r in sup:
    k = (r["architecture"], r["weather"])
    flat.append(({**r, "policy": "OR", "policy_key": r["policy_key"] + f",TAU0={tau[k]:.6f}"}))
    flat.append(({**r, "policy": "URT", "policy_key": r["policy_key"] + ",UNS_COOL_K=305.4"}))
for i, r in enumerate(flat):
    st = f"e{1 + i // 4}"; r["kernel"] = st; rows.append(r); jobs.setdefault(st, []).append(f'{r["architecture"]} {r["weather"]} {r["policy_key"]} 8640')
# fallback (pushed only if the reproducibility check of e0 fails, as PREREG_EXT12_v1 says): SUPR for the other 32 scenes, e19..e26
rest = [r for r in sup if r not in repro]
for i, r in enumerate(rest):
    st = f"e{19 + i // 4}"; rows.append({**r, "policy": "SUPR", "kernel": st}); jobs.setdefault(st, []).append(f'{r["architecture"]} {r["weather"]} {r["policy_key"]} 8640')
assert len(rows) == 108 and len(jobs) == 27 and all(len(v) == 4 for v in jobs.values())
with open(f"{H}/EXT12_BC_MANIFEST.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sup[0].keys())); w.writeheader(); w.writerows(rows)
D = f"{H}/exp_ds"; shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
for e in ("runner", "k07", "kre2_overlay"):
    shutil.copytree(f"{N}/k07uns/exp_ds/{e}", f"{D}/{e}")
for e in ("SB26_EXPERIMENTS_DATASET",):
    shutil.copy(f"{N}/k07uns/exp_ds/{e}", f"{D}/{e}")
os.makedirs(f"{D}/EXT12_BC/jobs"); shutil.copy(f"{N}/k07uns/exp_ds/K07UNS_FRESH/cross.csv", f"{D}/EXT12_BC/cross.csv")
for st, js in jobs.items():
    open(f"{D}/EXT12_BC/jobs/{st}.jobs", "w").write("\n".join(js) + "\n")
open(f"{D}/README.md", "w").write("Private study dataset: SB26 extension experiments B (oracle prior) and C (three rules), K07 blind scenes.\n")
json.dump({"title": "SB26 EXT12 Exp", "id": "asteriospanoras/sb26-ext12-exp", "subtitle": "SB26 extension experiments B and C (private)",
           "licenses": [{"name": "Apache 2.0"}], "description": "Private study dataset.", "keywords": [], "isPrivate": True}, open(f"{D}/dataset-metadata.json", "w"), indent=2)
print("built:", len(rows), "manifest rows,", len(jobs), "kernels; tau range", round(min(tau.values()), 3), round(max(tau.values()), 3))
