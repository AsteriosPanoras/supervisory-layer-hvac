"""Experiment A Kaggle sequencer (PREREG_EXT12_v1, execution note 3), the logic of gnnsup/gnn_auto.py.
- The account runs at most 5 Kaggle CPU sessions.  A uses min(ext12/MAX_ACTIVE_A (default 5), 5 - the graph-network kernels still active
  according to gnnsup/GNN_AUTO.log), read every cycle (execution note 4; the first version, which kept A at 1 slot until all 12 rerun
  kernels were complete, is kept as axabl_auto_v1_one_slot.py).
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
def gnn_active():
    """kernels of the graph-network rerun still active on Kaggle, from its log: the last event of each kernel is a push or a final status
    (execution note 4: A takes the slots the rerun does not use; a push refused by Kaggle is re-queued by both sequencers)"""
    try: lines = open(f"{N}/gnnsup/GNN_AUTO.log").read().splitlines()
    except Exception: return 4
    if any("weights check" in l for l in lines): return 0
    last = {}
    for l in lines:
        m = re.search(r"push (k\d+): .*successfully pushed", l) or re.search(r"recovered (k\d+): .*(RUNNING|QUEUED)", l)
        if m: last[m.group(1)] = "act"; continue
        m = re.search(r"sb26-gnnsup-(k\d+) KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)", l)
        if m: last[m.group(1)] = "done"
    return sum(1 for v in last.values() if v == "act")
def maxact():
    try: cap = max(0, min(5, int(open(f"{H}/MAX_ACTIVE_A").read().strip())))
    except Exception: cap = 5
    return max(0, min(cap, 5 - gnn_active()))
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
