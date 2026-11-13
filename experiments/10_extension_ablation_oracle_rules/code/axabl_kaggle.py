"""Experiment A on Kaggle (PREREG_EXT12_v1, execution note 3: host-B is unreachable): ONE push per call, then the caller checks the
status.  The kernel is the KRE-v2 kernel runner (kre2/runner/kernel_runner_v3.py) with EXPERIMENT=AXABL_K, the stage a0..a17 (the 72
registered jobs of AXABL_stage1.jobs, 4 per stage, unchanged), one shard and the episode runner run_episode_k07_v9supe_ax.py (557d4aab).
AX_DIR points into the private dataset sb26-axabl-exp (the Residual PPO copy in ax/).  Datasets: sb26-k07x3-scenarios, sb26-axabl-exp
(both private); the competition data.  Collected outputs go to results_axabl_blind/<stage>, where p289 reads them.  Nothing here reads or
prints a reward: collect only counts summaries.
usage: python -I axabl_kaggle.py kernel <stage> | collect <stage>     (stage a0..a17)"""
import os, sys, json, glob, shutil, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
mode, st = sys.argv[1], sys.argv[2]; assert st[0] == "a" and 0 <= int(st[1:]) <= 17; name = f"sb26-axabl-{st}"
EXP, SCEN = "AXABL_K", "sb26-k07x3-scenarios"
ENV = '''
# --- AXABL: the Residual PPO copy comes from the private experiments dataset (EXT12 execution note 3) ---
_gm = sorted(glob.glob("/kaggle/input/**/SB26_EXPERIMENTS_DATASET", recursive=True))
if _gm:
    _gr = os.path.dirname(_gm[0])
    os.environ["AX_DIR"] = os.path.join(_gr, "ax")
    os.environ.pop("AX_TORCH", None)
'''
if mode == "kernel":
    jobs = open(f"{H}/kaggle_axabl_ds/{EXP}/jobs/{st}.jobs").read().split("\n"); assert len([j for j in jobs if j.strip()]) == 4, st
    base = f"{H}/kaggle_a/{st}"; shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    src = open(f"{N}/kre2/runner/kernel_runner_v3.py").read()
    t = src.replace('"__EXPERIMENT__"', f'"{EXP}"').replace('"__STAGE__"', f'"{st}"').replace('"__SHARD__"', '"0"').replace('"__NSHARDS__"', '"1"').replace('"run_episode_k07_v2.py"', '"run_episode_k07_v9supe_ax.py"')
    anchor = 'LOCAL_ROOT_ENV = "RUNNER_LOCAL_ROOTS"'
    assert t.count(f'"{EXP}"') == 1 and '"run_episode_k07_v9supe_ax.py"' in t and t.count(anchor) == 1
    t = t.replace(anchor, anchor + "\n" + ENV, 1); open(base + "/runner.py", "w").write(t)
    json.dump({"id": f"{U}/{name}", "title": name, "code_file": "runner.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
               "enable_tpu": False, "enable_internet": False, "dataset_sources": [f"{U}/{SCEN}", f"{U}/sb26-axabl-exp"], "competition_sources": ["neurips-2026-smart-buildings"],
               "kernel_sources": [], "model_sources": []}, open(base + "/kernel-metadata.json", "w"), indent=2)
    if os.environ.get("AXABL_DRY"): sys.exit(print(name, "built (dry run, not pushed)"))
    r = subprocess.run([KG, "kernels", "push", "-p", base], capture_output=True, text=True); print(name, (r.stdout + r.stderr).strip().replace("\n", " ")[-150:])
else:
    raw = f"{N}/results_axabl_kaggle_raw/{st}"; shutil.rmtree(raw, ignore_errors=True); os.makedirs(raw)
    subprocess.run([KG, "kernels", "output", f"{U}/{name}", "-p", raw], capture_output=True, text=True)
    out = f"{N}/results_axabl_blind/{st}"; os.makedirs(out, exist_ok=True); n = 0
    for f in glob.glob(f"{raw}/**/*.summary.json", recursive=True) + glob.glob(f"{raw}/**/*.npz", recursive=True):
        shutil.copy2(f, out); n += f.endswith(".summary.json")
    print(name, "collected", n, "summaries")
