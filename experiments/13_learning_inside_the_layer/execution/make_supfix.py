"""K07SUPFIX setup (PREREG_K07SUPFIX_v1; written 2026-11-12 before any SUPFIX run): the layer with the start's time constant FIXED
(ALPHA = 0: tau stays at its prior 1.0 h) and the pressure reset ON, on the 36 blind scenes of the K07 test, to separate what learning
adds inside the full layer (SUP - SUPFIX) from what the feedback rule adds.  No new code: the arm is the frozen SUP key with ',ALPHA=0'
appended, on the frozen runner run_episode_k07_v9sup.py (sha256 e9f12b1c...), which accepts any start parameter in the key.
Laptop (5 workers, cores 0-11), the KRE-v2 kernel runner v3 and the competition-tracker kit of the graph-network laptop stages.
Two SUP jobs (the first two rows of the SUP manifest, fixed before any run) are re-run first on the laptop to measure laptop-vs-Kaggle
agreement (descriptive), because the SUP arm itself ran on Kaggle.  Writes exp_lap/, K07SUPFIX_MANIFEST.csv and the job list.
usage: python -I make_supfix.py"""
import os, csv, shutil, hashlib
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); S2 = f"{N}/k07x3run/exp_ds_s2"
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
E = f"{H}/exp_lap"; assert not os.path.exists(E), "exp_lap exists: the setup is written once"
os.makedirs(f"{E}/runner"); os.makedirs(f"{E}/K07SUPFIX_L/jobs")
for f in ("kernel_runner_v3.py", "run_episode_k07_v9sup.py", "cross_config.py"): shutil.copy2(f"{S2}/runner/{f}", f"{E}/runner/{f}")
shutil.copytree(f"{S2}/k07", f"{E}/k07"); shutil.copytree(f"{S2}/kre2_overlay", f"{E}/kre2_overlay"); shutil.copy2(f"{S2}/SB26_EXPERIMENTS_DATASET", f"{E}/SB26_EXPERIMENTS_DATASET")
shutil.copy2(f"{S2}/K07X3/cross.csv", f"{E}/K07SUPFIX_L/cross.csv")
assert sha(f"{E}/runner/run_episode_k07_v9sup.py").startswith("e9f12b1c3f07") and sha(f"{E}/runner/kernel_runner_v3.py").startswith("f01f234c433f")
man = list(csv.DictReader(open(f"{N}/k07x3run/K07X3_RUN_MANIFEST_S2.csv"))); assert len(man) == 36 and all(r["policy"] == "SUP" for r in man)
SUPKEY = man[0]["policy_key"]; assert all(r["policy_key"] == SUPKEY for r in man) and SUPKEY.endswith(";ROS=1,TR=1")
FIXKEY = SUPKEY + ",ALPHA=0"
jobs = [f"{r['architecture']} {r['weather']} {SUPKEY} 8640" for r in man[:2]] + [f"{r['architecture']} {r['weather']} {FIXKEY} 8640" for r in man]
open(f"{E}/K07SUPFIX_L/jobs/stage1.jobs", "w").write("".join(j + "\n" for j in jobs))
with open(f"{H}/K07SUPFIX_MANIFEST.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["architecture", "group", "weather", "day1_class", "policy", "policy_key", "kernel"]); w.writeheader()
    for r in man: w.writerow(dict(r, policy="SUPFIX", policy_key=FIXKEY, kernel="laptop"))
print("jobs", len(jobs), "(2 SUP repro + 36 SUPFIX); SUP key ends", SUPKEY[-14:], "| SUPFIX key ends", FIXKEY[-22:])
