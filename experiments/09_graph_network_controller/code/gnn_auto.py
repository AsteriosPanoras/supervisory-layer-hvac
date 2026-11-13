"""GNNSUP Kaggle sequencer (execution note 3), the logic of ext12_auto.py reduced to this rerun.
- Kernels k0..k11 are pushed whenever fewer than MAX_ACTIVE kernels of ours are active (gnnsup/MAX_ACTIVE, default 4, read every cycle).
  A kernel that exists is never pushed again (state rebuilt from Kaggle on start). A failed kernel is retried once.
- k0 also reruns the laptop's building_52300 pittsburgh22_w0918 pair; whether Kaggle reproduces it bitwise is logged (descriptive).
- When every kernel is collected AND the laptop stage gnb_pl has ended: the weights check, then p285 and p286. The script prints no reward
  itself; it logs the analyses' own output.
log: gnnsup/GNN_AUTO.log"""
import os, sys, json, glob, time, subprocess
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
PY = "<repo>/.venv/bin/python"; KG = "<repo>/.venv/bin/kaggle"; U = "asteriospanoras"
WSHA = "e5483721a48a99d50e1684d28c9dff48da00fe5894908a214b0ceaf70c7e4c51"
def log(m): open(f"{H}/GNN_AUTO.log", "a").write(time.strftime("%F %T ") + m + "\n")
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True)
st = lambda k: sh([KG, "kernels", "status", f"{U}/{k}"]).stdout
isdone = lambda s: any(x in s for x in ("COMPLETE", "ERROR", "CANCEL")); isact = lambda s: any(x in s for x in ("RUNNING", "QUEUED"))
kname = lambda s_: f"sb26-gnnsup-{s_}"
def maxact():
    try: return max(0, min(5, int(open(f"{H}/MAX_ACTIVE").read().strip())))
    except Exception: return 4
queue = [f"k{i}" for i in range(12)]; pushed, final, retried = [], {}, set(); repro_done = False
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
            final[k] = s; r = sh([PY, "-I", f"{H}/gnn_kaggle.py", "collect", s_]); log(f"{k} {s.strip()[-28:]} | {r.stdout.strip()}")
            if ("ERROR" in s or "CANCEL" in s) and s_ not in retried: retried.add(s_); pushed.remove(s_); final.pop(k); queue.insert(0, s_); log(f"{k} failed -> one retry queued")
        else: act += isact(s)
    if not repro_done and kname("k0") in final:
        rows = []
        for f in glob.glob(f"{N}/results_gnnsup_kaggle_repro/k0/*.npz"):
            g = glob.glob(f"{H}/laptop/work_gnb_l/out/**/{os.path.basename(f)}", recursive=True)
            if g:
                a, b = np.load(f, allow_pickle=True), np.load(g[0], allow_pickle=True)
                rows.append({"file": os.path.basename(f), "rewards_bitwise": bool(np.array_equal(a["rewards"], b["rewards"])),
                             "max_abs_diff": float(np.abs(a["rewards"] - b["rewards"]).max()), "mean_abs_kaggle": float(np.abs(a["rewards"]).mean()), "mean_abs_laptop": float(np.abs(b["rewards"]).mean())})
        json.dump(rows, open(f"{H}/RESULT_GNNSUP_KAGGLE_REPRO.json", "w"), indent=1); repro_done = True; log(f"Kaggle reproducibility (descriptive): {json.dumps(rows)}")
    while queue and act < maxact():
        s_ = queue.pop(0); r = sh([PY, "-I", f"{H}/gnn_kaggle.py", "kernel", s_]); out = r.stdout.strip()
        if "successfully pushed" not in out: queue.insert(0, s_); log(f"push {s_} refused, back in the queue: {out[-120:]}"); break      # execution note 4
        pushed.append(s_); act += 1; log(f"push {s_}: {out[-120:]}"); time.sleep(5)
    if not queue and len(final) == len(pushed) and os.path.exists(f"{H}/laptop/work_gnb_pl/END"):
        break
    time.sleep(120)
fs = glob.glob(f"{N}/results_gnnsup_blind/**/*.summary.json", recursive=True) + glob.glob(f"{H}/laptop/work_gnb*/out/**/*.summary.json", recursive=True)
bad = [f for f in fs if (json.load(open(f)).get("gnn") or {}).get("weights_sha256") != WSHA]
json.dump({"n": len(fs), "ok": sum(json.load(open(f))["status"] == "OK" for f in fs), "bad_weights": bad}, open(f"{H}/RESULT_GNNSUP_WEIGHTS_CHECK.json", "w"), indent=1)
log(f"weights check: {len(fs)} blind summaries, {len(bad)} with a wrong or missing weights sha")
if not bad:
    for p, o in (("p285_gnnsup_blind.py", "RESULT_GNNSUP_BLIND.json"), ("p286_headroom_gnn.py", "RESULT_GNNSUP_HEADROOM.json")):
        r = sh([PY, "-I", f"{H}/{p}", f"{H}/{o}"]); log(f"{p}: {(r.stdout + r.stderr).strip()[-700:]}")
log("ALL DONE")
