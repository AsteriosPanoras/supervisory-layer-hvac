"""EXT12 B/C sequencer (PREREG_EXT12_v1), the logic of k07uns_auto_v3 reduced to this experiment.
- Kernels e0..e18 are pushed whenever fewer than MAX_ACTIVE kernels of ours are active (ext12/MAX_ACTIVE, default 4, read every cycle). They
  are collected into results_ext12_bc/<stage>.  A failed kernel is retried once.
- On start the state is rebuilt from Kaggle: a kernel that exists is never pushed again.
- When e0 is collected, F-B (SUPR = SUP bit for bit) is computed by ext12_bc_common.reference. If it fails, the fallback kernels
  e19..e26 are queued.
- When everything queued is collected: p290 and p291.  The script prints no reward itself; it logs the analyses' own output.
log: ext12/EXT12_AUTO.log"""
import os, sys, json, glob, time, subprocess
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); sys.path.insert(0, H)
PY = "<repo>/.venv/bin/python"; KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
def log(m): open(f"{H}/EXT12_AUTO.log", "a").write(time.strftime("%F %T ") + m + "\n")
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)
st = lambda k: sh([KG, "kernels", "status", f"{U}/{k}"]).stdout
isdone = lambda s: any(x in s for x in ("COMPLETE", "ERROR", "CANCEL")); isact = lambda s: any(x in s for x in ("RUNNING", "QUEUED"))
kname = lambda s_: f"sb26-ext12-{s_}-s0"
def maxact():
    try: return max(0, min(5, int(open(f"{H}/MAX_ACTIVE").read().strip())))
    except Exception: return 4
main_q = [f"e{i}" for i in range(19)]; fb_q = [f"e{i}" for i in range(19, 27)]; queue = list(main_q); pushed, final, retried = [], {}, set(); expected = set(main_q)
log(f"start (pid {os.getpid()})")
for s_ in main_q + fb_q:
    s = st(kname(s_)); time.sleep(2)
    if isact(s) or isdone(s):
        if s_ in queue: queue.remove(s_)
        pushed.append(s_); expected.add(s_); log(f"recovered {s_}: {s.strip()[-30:]}")
repro_done = False
while True:
    act = 0
    for s_ in list(pushed):
        k = kname(s_)
        if k in final: continue
        s = st(k); time.sleep(2)
        if isdone(s):
            final[k] = s; r = sh([PY, "-I", f"{H}/ext12_kaggle.py", "collect", s_]); log(f"{k} {s.strip()[-28:]} | {r.stdout.strip()}")
            if ("ERROR" in s or "CANCEL" in s) and s_ not in retried: retried.add(s_); pushed.remove(s_); final.pop(k); queue.insert(0, s_); log(f"{k} failed -> one retry queued")
        else: act += isact(s)
    if not repro_done and kname("e0") in final:
        from ext12_bc_common import load, reference
        L, RW, C, RAW, grp = load(); ref, rep = reference(L, RAW); repro_done = True; log(f"F-B reproducibility: {json.dumps(rep)}")
        if not rep["REPRO_OK"]: queue += [s_ for s_ in fb_q if s_ not in pushed and s_ not in queue]; expected |= set(fb_q); log("F-B FAILED -> fallback SUPR kernels e19..e26 queued")
    while queue and act < maxact():
        s_ = queue.pop(0); r = sh([PY, "-I", f"{H}/ext12_kaggle.py", "kernel", s_]); time.sleep(25); s2 = st(kname(s_)); log(f"push {s_}: {r.stdout.strip()[-60:]} | {s2.strip()[-30:]}")
        if not (isact(s2) or isdone(s2)): queue.insert(0, s_); break
        pushed.append(s_); act += 1
    if repro_done and not queue and all(kname(s_) in final for s_ in expected):
        for p in ("p290_oracle_prior", "p291_three_rules"):
            r = sh([PY, "-I", f"{H}/{p}.py", f"{H}/RESULT_{p.upper()}.json"]); log(f"{p}: " + (r.stdout + r.stderr).strip().replace("\n", " || ")[:2500])
        log("ALL DONE"); break
    time.sleep(300)
