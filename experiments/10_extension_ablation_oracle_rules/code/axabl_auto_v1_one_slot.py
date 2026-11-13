"""Experiment A Kaggle sequencer (PREREG_EXT12_v1, execution note 3), the logic of gnnsup/gnn_auto.py.
- The account runs at most 5 Kaggle CPU sessions.  While the graph-network rerun still has kernels on Kaggle (fewer than its 12 kernels
  logged COMPLETE in gnnsup/GNN_AUTO.log, and no 'weights check' line yet), A uses at most 1 slot (the rerun keeps at most 4); afterwards
  at most min(5, ext12/MAX_ACTIVE_A) (default 5), read every cycle.
- Kernels a0..a17 are pushed in order; a kernel that exists is never pushed again (state rebuilt from Kaggle on start); a failed kernel
  is retried once.  A push that Kaggle refuses (session limit) is put back in the queue, not counted as pushed.
- When every kernel is collected: p289 (RESULT_P289_AXABL.json), then 'ALL DONE'.  The script prints no reward itself; it logs p289's
  own output.
log: ext12/AXABL_AUTO.log"""
import os, re, sys, json, glob, time, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
PY = "<repo>/.venv/bin/python"; KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
def log(m): open(f"{H}/AXABL_AUTO.log", "a").write(time.strftime("%F %T ") + m + "\n")
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)
st = lambda k: sh([KG, "kernels", "status", f"{U}/{k}"]).stdout
isdone = lambda s: any(x in s for x in ("COMPLETE", "ERROR", "CANCEL")); isact = lambda s: any(x in s for x in ("RUNNING", "QUEUED"))
kname = lambda s_: f"sb26-axabl-{s_}"
def maxact():
    try: g = open(f"{N}/gnnsup/GNN_AUTO.log").read()
    except Exception: g = ""
    if len(set(re.findall(r"(sb26-gnnsup-k\d+) KernelWorkerStatus\.COMPLETE", g))) < 12 and "weights check" not in g: return 1
    try: return max(0, min(5, int(open(f"{H}/MAX_ACTIVE_A").read().strip())))
    except Exception: return 5
queue = [f"a{i}" for i in range(18)]; pushed, final, retried = [], {}, set()
log(f"start (pid {os.getpid()})")
for s_ in list(queue):
    s = st(kname(s_)); time.sleep(2)
    if isact(s) or isdone(s): queue.remove(s_); pushed.append(s_); log(f"recovered {s_}: {s.strip()[-30:]}")
while True:
    act = 0
    for s_ in list(pushed):
        k = kname(s_)
        if k in final: continue
        s = st(k); time.sleep(2)
        if isdone(s):
            final[k] = s; r = sh([PY, "-I", f"{H}/axabl_kaggle.py", "collect", s_]); log(f"{k} {s.strip()[-28:]} | {r.stdout.strip()}")
            if ("ERROR" in s or "CANCEL" in s) and s_ not in retried: retried.add(s_); pushed.remove(s_); final.pop(k); queue.insert(0, s_); log(f"{k} failed -> one retry queued")
        else: act += isact(s)
    while queue and act < maxact():
        s_ = queue.pop(0); r = sh([PY, "-I", f"{H}/axabl_kaggle.py", "kernel", s_]); out = r.stdout.strip()
        if "successfully pushed" not in out: queue.insert(0, s_); log(f"push {s_} refused, back in the queue: {out[-120:]}"); break
        pushed.append(s_); act += 1; log(f"push {s_}: {out[-120:]}"); time.sleep(5)
    if not queue and len(final) == len(pushed): break
    time.sleep(120)
r = sh([PY, "-I", f"{H}/p289_axabl.py", f"{H}/RESULT_P289_AXABL.json"]); log(f"p289_axabl.py: {(r.stdout + r.stderr).strip()[-900:]}")
log("ALL DONE")
