"""K07UNS sequencer (PREREG_K07UNS_v1).  Kaggle: development kernels d0..d4 (UR; d0 starts with the 288-step parity run) then d5..d9 (U), whenever fewer than 5 kernels of
ours are active; collected into results_k07uns_dev/<st>.  d0 parity vs the laptop fidelity UR run (prefix) -> PARITY_D0.txt.  When d0..d4 are collected and the 12 official
G30+UR laptop runs are in: p247 -> RESULT_K07UNS_GATE.json (once).  Only if GATE_PASS and parity PASS: fresh kernels f0..f17, collected into results_k07uns_fresh/<st>;
when all are in: p248 -> RESULT_K07UNS_FRESH.json.  When d5..d9 are collected and the laptop official stage has ended: p247 again -> RESULT_K07UNS_GATE_ABLATION.json.
This script prints no reward itself; it logs the analysis scripts' own output.  log: k07uns/K07UNS_AUTO.log
v2 (11 Nov ~10:50, before any outcome): on start it rebuilds its state from Kaggle (a kernel that exists is never pushed again; a finished one is collected
unless already collected), and the number of active kernels it may use is read every cycle from k07uns/MAX_ACTIVE (default 5), so a slot can be left free for the
official submissions.
v3 (11 Nov ~11:10, before any outcome): once the gate has passed, the fresh kernels go to the FRONT of the queue (before the remaining development-U kernels, the least
important part); on start it also recovers fresh kernels that already exist."""
import os, json, glob, time, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
PY = "<repo>/.venv/bin/python"; KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
G = "RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
GATE_KEYS = {f"k07@{G}", f"k07sup@{G};ROS=1,TR=0,UNS_COOL_K=305.4"}
def log(m): open(f"{H}/K07UNS_AUTO.log", "a").write(time.strftime("%F %T ") + m + "\n")
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)
st = lambda k: sh([KG, "kernels", "status", f"{U}/{k}"]).stdout
isdone = lambda s: any(x in s for x in ("COMPLETE", "ERROR", "CANCEL")); isact = lambda s: any(x in s for x in ("RUNNING", "QUEUED"))
kname = lambda s_: f"sb26-k07uns-{s_}-s0"
def official_gate_runs():
    n = 0
    for f in glob.glob(f"{H}/laptop/work_off*/out/**/*.summary.json", recursive=True):
        s = json.load(open(f)); n += int(s["policy_key"] in GATE_KEYS and s["status"] == "OK" and s["steps_done"] == 8640)
    return n
def parity():
    pj = json.load(open(f"{H}/PARITY_JOB_D0.json"))["parity_job"]
    pick = lambda d, n: [f for f in glob.glob(f"{d}/**/*.summary.json", recursive=True) if json.load(open(f))["building"] == pj["building"] and json.load(open(f))["weather"] == pj["weather"]
                         and json.load(open(f))["policy_key"] == pj["policy_key"] and json.load(open(f))["steps_done"] == n]
    kag, loc = pick(f"{N}/results_k07uns_dev/d0", pj["steps"]), pick(pj["laptop_reference"], pj["reference_steps"])
    if len(kag) != 1 or len(loc) != 1: return f"MISSING kaggle={len(kag)} laptop={len(loc)}"
    a, b = np.load(kag[0].replace(".summary.json", ".npz"), allow_pickle=True), np.load(loc[0].replace(".summary.json", ".npz"), allow_pickle=True); n = pj["steps"]; bad = []
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files: bad.append(f"{k}:missing"); continue
        x, y = a[k], b[k]
        if getattr(x, "ndim", 0) >= 1 and getattr(y, "ndim", 0) >= 1 and x.shape[0] == n and y.shape[0] == pj["reference_steps"]: y = y[:n]
        if not (x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))): bad.append(k)
    la = [e for e in json.load(open(kag[0]))["ros"]["log"] if e.get("step", 0) < n]; lb = [e for e in json.load(open(loc[0]))["ros"]["log"] if e.get("step", 0) < n]
    return "PASS-BITWISE (prefix)" if not bad and la == lb else f"FAIL {bad[:6]} log_equal={la == lb}"
log(f"start v3 (pid {os.getpid()})")
dev_q = [f"d{i}" for i in range(10)]; fresh_q = [f"f{i}" for i in range(18)]; queue = list(dev_q); pushed, final, retried = [], {}, set()
def maxact():
    try: return max(0, min(5, int(open(f"{H}/MAX_ACTIVE").read().strip())))
    except Exception: return 5
for s_ in dev_q + fresh_q:                                         # v2/v3: state from Kaggle; never push a kernel that exists
    s = st(kname(s_)); time.sleep(2)
    if (isact(s) or isdone(s)) and s_ in queue: queue.remove(s_)
    if isact(s) or isdone(s): pushed.append(s_); log(f"recovered {s_}: {s.strip()[-30:]}")
    if isdone(s) and os.path.isdir(f"{N}/{'results_k07uns_dev' if s_[0] == 'd' else 'results_k07uns_fresh'}/{s_}") and any(True for _ in glob.iglob(f"{N}/{'results_k07uns_dev' if s_[0] == 'd' else 'results_k07uns_fresh'}/{s_}/**/*.summary.json", recursive=True)): final[kname(s_)] = s; log(f"{s_} already collected")
if kname("d0") in final: par0 = parity(); open(f"{H}/PARITY_D0.txt", "w").write(par0 + "\n"); log(f"parity d0 (recovered): {par0}")
gate_done = os.path.exists(f"{H}/RESULT_K07UNS_GATE.json"); gate_pass = False; par = (open(f"{H}/PARITY_D0.txt").read().strip() if os.path.exists(f"{H}/PARITY_D0.txt") else None); fresh_open = False; fresh_expected = False; abl_done = False; fresh_done = False
while True:
    act = 0
    for s_ in list(pushed):
        k = kname(s_)
        if k in final: continue
        s = st(k); time.sleep(2)
        if isdone(s):
            final[k] = s; r = sh([PY, "-I", f"{H}/k07uns_kaggle.py", "collect", s_]); log(f"{k} {s.strip()[-28:]} | {r.stdout.strip()}")
            if ("ERROR" in s or "CANCEL" in s) and s_ not in retried: retried.add(s_); pushed.remove(s_); final.pop(k); queue.insert(0, s_); log(f"{k} failed -> one retry queued")
            if s_ == "d0" and k in final: par = parity(); open(f"{H}/PARITY_D0.txt", "w").write(par + "\n"); log(f"parity d0: {par}")
        else: act += isact(s)
    while queue and act < maxact():
        s_ = queue.pop(0); r = sh([PY, "-I", f"{H}/k07uns_kaggle.py", "kernel", s_]); time.sleep(25); s2 = st(kname(s_)); log(f"push {s_}: {r.stdout.strip()[-60:]} | {s2.strip()[-30:]}")
        if not (isact(s2) or isdone(s2)): queue.insert(0, s_); break
        pushed.append(s_); act += 1
    dev_ur_in = all(kname(s_) in final for s_ in dev_q[:5]); dev_all_in = all(kname(s_) in final for s_ in dev_q)
    if not gate_done and dev_ur_in and (official_gate_runs() >= 12 or os.path.exists(f"{H}/laptop/work_off/END")):
        r = sh([PY, "-I", f"{H}/p247_uns_gate.py", f"{H}/RESULT_K07UNS_GATE.json"]); gate_done = True; log("GATE p247: " + r.stdout.strip().replace("\n", " || ")[:1500])
    if gate_done and not fresh_open:
        g = json.load(open(f"{H}/RESULT_K07UNS_GATE.json")); gate_pass = bool(g.get("GATE_PASS"))
        if gate_pass and par is not None and par.startswith("PASS"): fresh_open = True; fresh_expected = True; queue[:0] = [s_ for s_ in fresh_q if s_ not in pushed]; log("GATE_PASS and parity PASS -> fresh kernels queued")
        elif not gate_pass: fresh_open = True; log("GATE not passed -> fresh kernels NOT pushed (PREREG_K07UNS_v1)")
        elif par is not None: fresh_open = True; log(f"GATE_PASS but Kaggle parity {par} -> fresh kernels NOT pushed; experimenter must look")
    if not abl_done and dev_all_in and os.path.exists(f"{H}/laptop/work_off/END"):
        r = sh([PY, "-I", f"{H}/p247_uns_gate.py", f"{H}/RESULT_K07UNS_GATE_ABLATION.json"]); abl_done = True; log("ABLATION p247: " + r.stdout.strip().replace("\n", " || ")[:2500])
    if fresh_expected and not fresh_done and all(kname(s_) in final for s_ in fresh_q):
        r = sh([PY, "-I", f"{H}/p248_uns_fresh.py", f"{H}/RESULT_K07UNS_FRESH.json"]); fresh_done = True; log("FRESH p248: " + r.stdout.strip().replace("\n", " || ")[:2500])
    if abl_done and fresh_open and (fresh_done or not fresh_expected) and not queue: log("ALL DONE"); break
    time.sleep(300)
