"""GNNSUP blind rerun on Kaggle (execution note 3): ONE push per call, then the caller checks the status.
The kernel is the KRE-v2 kernel runner (kre2/runner/kernel_runner_v3.py) with EXPERIMENT=GNNSUP_BLIND_PK, the stage k0..k11, one shard and
the episode runner run_episode_k07_v9supe_gnn.py. AX_DIR, GNN_DIR and GNN_WEIGHTS point into the private dataset sb26-gnnsup-exp (the
Residual PPO copy in ax/, the graph network and its registered weights in gnn/). Datasets: sb26-k07x3-scenarios, sb26-gnnsup-exp (both
private); the competition data. Collected outputs go to results_gnnsup_blind/<stage>; the reproducibility pair (building_52300
pittsburgh22_w0918, already run on the laptop) goes to results_gnnsup_kaggle_repro/ so that it is never counted twice.
usage: python -I gnn_kaggle.py kernel <stage> | collect <stage>     (stage k0..k11)"""
import os, sys, json, glob, shutil, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
mode, st = sys.argv[1], sys.argv[2]; assert st[0] == "k" and 0 <= int(st[1:]) <= 11; name = f"sb26-gnnsup-{st}"
EXP, SCEN = "GNNSUP_BLIND_PK", "sb26-k07x3-scenarios"
ENV = '''
# --- GNNSUP: the graph network and the Residual PPO copy come from the private experiments dataset (execution note 3) ---
_gm = sorted(glob.glob("/kaggle/input/**/SB26_EXPERIMENTS_DATASET", recursive=True))
if _gm:
    _gr = os.path.dirname(_gm[0])
    os.environ["AX_DIR"] = os.path.join(_gr, "ax"); os.environ["GNN_DIR"] = os.path.join(_gr, "gnn")
    os.environ["GNN_WEIGHTS"] = os.path.join(_gr, "gnn", "gnn_weights_v1.pt")
    os.environ.pop("GNN_REC_DIR", None); os.environ.pop("AX_TORCH", None)
'''
if mode == "kernel":
    base = f"{H}/kaggle/{st}"; shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    src = open(f"{N}/kre2/runner/kernel_runner_v3.py").read()
    t = src.replace('"__EXPERIMENT__"', f'"{EXP}"').replace('"__STAGE__"', f'"{st}"').replace('"__SHARD__"', '"0"').replace('"__NSHARDS__"', '"1"').replace('"run_episode_k07_v2.py"', '"run_episode_k07_v9supe_gnn.py"')
    anchor = 'LOCAL_ROOT_ENV = "RUNNER_LOCAL_ROOTS"'
    assert t.count(f'"{EXP}"') == 1 and '"run_episode_k07_v9supe_gnn.py"' in t and t.count(anchor) == 1
    t = t.replace(anchor, anchor + "\n" + ENV, 1); open(base + "/runner.py", "w").write(t)
    json.dump({"id": f"{U}/{name}", "title": name, "code_file": "runner.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
               "enable_tpu": False, "enable_internet": False, "dataset_sources": [f"{U}/{SCEN}", f"{U}/sb26-gnnsup-exp"], "competition_sources": ["neurips-2026-smart-buildings"],
               "kernel_sources": [], "model_sources": []}, open(base + "/kernel-metadata.json", "w"), indent=2)
    r = subprocess.run([KG, "kernels", "push", "-p", base], capture_output=True, text=True); print(name, (r.stdout + r.stderr).strip().replace("\n", " ")[-150:])
else:
    raw = f"{N}/results_gnnsup_kaggle_raw/{st}"; shutil.rmtree(raw, ignore_errors=True); os.makedirs(raw)
    subprocess.run([KG, "kernels", "output", f"{U}/{name}", "-p", raw], capture_output=True, text=True)
    out, rep = f"{N}/results_gnnsup_blind/{st}", f"{N}/results_gnnsup_kaggle_repro/{st}"; os.makedirs(out, exist_ok=True); os.makedirs(rep, exist_ok=True)
    n = r_ = 0
    for f in glob.glob(f"{raw}/**/*.summary.json", recursive=True) + glob.glob(f"{raw}/**/*.npz", recursive=True):
        dst = rep if os.path.basename(f).startswith("building_52300_pittsburgh22_w0918_") else out
        shutil.copy2(f, dst); n += f.endswith(".summary.json") and dst == out; r_ += f.endswith(".summary.json") and dst == rep
    print(name, "collected", n, "blind summaries", f"+ {r_} reproducibility" if r_ else "")
