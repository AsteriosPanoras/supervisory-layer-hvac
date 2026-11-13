"""ZTSUP kernel launcher (KRE-v2; PREREG_ZTSUP_v1): ONE push per call, then the caller checks the status.  usage: python -I ztsup_kaggle.py kernel <z0..z11> | collect <z0..z11>
Datasets: sb26-k07x3-scenarios (fresh scenes) + sb26-ztsup-exp (runner v9supe_zt, ZT classes, overlay, jobs; private)."""
import os, sys, json, shutil, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
mode, st = sys.argv[1], sys.argv[2]; assert st.startswith("z"); name = f"sb26-ztsup-{st}-s0"
if mode == "kernel":
    base = f"{H}/kaggle/{st}"; shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    src = open(f"{N}/kre2/runner/kernel_runner_v3.py").read()
    t = src.replace('"__EXPERIMENT__"', '"ZTSUP"').replace('"__STAGE__"', f'"{st}"').replace('"__SHARD__"', '"0"').replace('"__NSHARDS__"', '"1"').replace('"run_episode_k07_v2.py"', '"run_episode_k07_v9supe_zt.py"')
    assert t.count('"ZTSUP"') == 1 and '"run_episode_k07_v9supe_zt.py"' in t; open(base + "/runner.py", "w").write(t)
    json.dump({"id": f"{U}/{name}", "title": name, "code_file": "runner.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "dataset_sources": [f"{U}/sb26-k07x3-scenarios", f"{U}/sb26-ztsup-exp"], "competition_sources": ["neurips-2026-smart-buildings"], "kernel_sources": [], "model_sources": []}, open(base + "/kernel-metadata.json", "w"), indent=2)
    r = subprocess.run([KG, "kernels", "push", "-p", base], capture_output=True, text=True); print(name, (r.stdout + r.stderr).strip().replace("\n", " ")[-150:])
else:
    out = f"{N}/results_ztsup_fresh/{st}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out); r = subprocess.run([KG, "kernels", "output", f"{U}/{name}", "-p", out], capture_output=True, text=True); print(name, "collected", sum(1 for _ in os.popen(f"find {out} -name '*.summary.json'")), "summaries")
