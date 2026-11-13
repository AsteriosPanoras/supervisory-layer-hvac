"""EXT12 B/C kernel launcher (PREREG_EXT12_v1), adapted from k07uns/k07uns_kaggle.py: ONE push per call, then the caller checks the
status.  The kernel is the KRE-v2 kernel runner (kre2/runner/kernel_runner_v3.py) with EXPERIMENT=EXT12_BC, the stage, one shard and the
episode runner run_episode_k07_v9supe.py; datasets sb26-k07x3-scenarios and sb26-ext12-exp (private); the competition data.
usage: python -I ext12_kaggle.py kernel <stage> | collect <stage>     (stage e0..e26)"""
import os, sys, json, shutil, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
mode, st = sys.argv[1], sys.argv[2]; assert st[0] == "e" and 0 <= int(st[1:]) <= 26; name = f"sb26-ext12-{st}-s0"
EXP, SCEN, RES = "EXT12_BC", "sb26-k07x3-scenarios", "results_ext12_bc"
if mode == "kernel":
    base = f"{H}/kaggle/{st}"; shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    src = open(f"{N}/kre2/runner/kernel_runner_v3.py").read()
    t = src.replace('"__EXPERIMENT__"', f'"{EXP}"').replace('"__STAGE__"', f'"{st}"').replace('"__SHARD__"', '"0"').replace('"__NSHARDS__"', '"1"').replace('"run_episode_k07_v2.py"', '"run_episode_k07_v9supe.py"')
    assert t.count(f'"{EXP}"') == 1 and '"run_episode_k07_v9supe.py"' in t; open(base + "/runner.py", "w").write(t)
    json.dump({"id": f"{U}/{name}", "title": name, "code_file": "runner.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
               "enable_tpu": False, "enable_internet": False, "dataset_sources": [f"{U}/{SCEN}", f"{U}/sb26-ext12-exp"], "competition_sources": ["neurips-2026-smart-buildings"],
               "kernel_sources": [], "model_sources": []}, open(base + "/kernel-metadata.json", "w"), indent=2)
    r = subprocess.run([KG, "kernels", "push", "-p", base], capture_output=True, text=True); print(name, (r.stdout + r.stderr).strip().replace("\n", " ")[-150:])
else:
    out = f"{N}/{RES}/{st}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    r = subprocess.run([KG, "kernels", "output", f"{U}/{name}", "-p", out], capture_output=True, text=True)
    print(name, "collected", sum(1 for _ in os.popen(f"find {out} -name '*.summary.json'")), "summaries")
