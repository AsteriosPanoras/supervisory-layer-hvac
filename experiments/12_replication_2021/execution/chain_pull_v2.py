"""chain_pull_v2 (H3, 12 Nov) = chain_pull.py with three infrastructure changes and nothing else: (1) the plan may set the worker count of a machine, (2) "wait_for" tolerates a start file that does not exist yet and accepts any of the listed strings, (3) the ledger records this file.
N10-S PULL scheduler: ONE queue of jobs, the three machines PULL chunks when they are free, so that no machine waits for another and the three finish together.
usage (from N10S_design): python -I chain_pull.py plan.json
plan.json: {"prefix": "par", "cross": "cross_r.csv", "kscen": "sb26-n18-scenarios", "res": "results_h0",
            "jobs": [{"line": "<building> <weather> <policy_key> 8640", "pin": "kaggle"|"laptop"|"host-B"|null}, ...],
            "rates": {"host-B": 1.0, "kaggle": 0.69, "laptop": 0.2}, "chunk_min": {...}, "chunk_max": {...}}
A machine takes first the jobs pinned to it, then a chunk of the shared pool: guided self-scheduling, chunk = share(m) * remaining / 2, bounded by chunk_min/chunk_max (the end of the queue is therefore handed out in small pieces to whoever is free).
A chunk that fails is put back (twice); a third failure writes FAILED_pull.txt.  Jobs already complete in the result directory are skipped (restart-safe).  Log: chain_pull_<prefix>.log, last word DONE."""
import re, sys, os, json, time, glob, csv, subprocess, hashlib, shutil, threading, traceback, math
N = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(N); PY = "<repo>/.venv/bin/python"; KG = "<repo>/.venv/bin/kaggle"
plan = json.load(open(sys.argv[1])); P = plan["prefix"]; machine = "pull"; RES = f"{N}/{plan['res']}"; os.makedirs(RES, exist_ok=True); LOG = f"{N}/chain_pull_{P}.log"; LH = f"{D}/N9_design/laptop_run"
SSHK = os.path.expanduser("~/.ssh/<key>"); PH = "<host-b>"; RR = "<host-b>/n10s"; SSH = ["ssh", "-i", SSHK, "-o", "BatchMode=yes", "-o", "ConnectTimeout=25"]
def say(*a): open(LOG, "a").write(time.strftime("%F %T ") + " ".join(str(x) for x in a) + "\n")
def sh(cmd, **kw): return subprocess.run(cmd, capture_output=True, text=True, **kw)
def ledger(*files):
    with open(f"{N}/N10S_FROZEN_SHA256_v1_1.txt", "a") as fh:
        for f in files: fh.write(hashlib.sha256(open(f, "rb").read()).hexdigest() + "  " + os.path.basename(f) + "\n")
src = open(f"{N}/chain_r.py").read(); AD = src[src.index("# ---------------- machine adapters"):src.index("RUN = {")].replace("N19_", "N21_").replace("sb26-n19-exp-", "sb26-n21-exp-").replace('f"sb26-n19-{t}"', 'f"sb26-n21-{t}"').replace('kid = f"sb26-n19-{t}-s{i}"', 'kid = f"sb26-n21-{t}-s{i}"')
assert "n19" not in AD, "adapter rename incomplete"
RUNNER = plan.get("runner", "run_episode_k07_v7_abl.py"); CROSS = f"{N}/{plan['cross']}"; SCEN = plan.get("kscen"); exec(AD)
RATES = plan["rates"]; CMIN = plan.get("chunk_min", {"host-B": 4, "kaggle": 5, "laptop": 2}); CMAX = plan.get("chunk_max", {"host-B": 28, "kaggle": 30, "laptop": 8}); WORKERS = {"host-B": 14, "kaggle": None, "laptop": 12}; WORKERS.update(plan.get("workers", {}))
RUNFN = {"host-B": run_host-B, "kaggle": run_kaggle, "laptop": run_laptop}
START_T = time.time(); DEADLINE = float(plan["no_new_chunk_after_epoch"]) if plan.get("no_new_chunk_after_epoch") else (START_T + float(plan["no_new_chunk_after_hours"]) * 3600 if plan.get("no_new_chunk_after_hours") else None)      # H2: no new chunk after this time; unstarted bundles (extension first, by the fixed order) are simply not run
LOCK = threading.Lock(); QUEUE = []; DONE_N = [0]; FAILS = {}; ERR = []
def key(line): p = line.split(); return (p[0], p[1], p[2])
def grp(j): p = j["line"].split(); return j.get("grp") or (p[0], p[1])
def have_keys(pat="*"):
    h = set()
    for f_ in glob.glob(RES + f"/{pat}/**/*.summary.json", recursive=True):
        try: j_ = json.load(open(f_))
        except Exception: continue
        if j_.get("status") == "OK" and j_.get("steps_done") == 8640: h.add((j_["building"], j_["weather"], j_["policy_key"]))
    return h
def claim(m):
    with LOCK:
        if DEADLINE and time.time() > DEADLINE: return []
        pinned = [j for j in QUEUE if j["pin"] == m]
        if pinned: take = pinned[:CMAX[m]]
        else:
            pool = [j for j in QUEUE if not j["pin"]]
            if not pool: return []
            share = RATES[m] / sum(RATES.values()); n = max(CMIN[m], min(CMAX[m], math.ceil(len(pool) * share / 2))); take = pool[:n]
            while len(take) < len(pool) and grp(pool[len(take)]) == grp(take[-1]): take.append(pool[len(take)])      # MATCHED BUNDLES: all arms of one cell go to the same machine (no hardware confounding inside a cell)
        for j in take: QUEUE.remove(j)
        return take
PERM = []
def collect_partial(m, tag):
    """a runner that returned a failure code for SOME episodes (e.g. a building the simulator cannot initialise: 'Wasn't able to determine which corner'): the episodes that finished are still valid results and are collected."""
    out = f"{RES}/{tag}"; os.makedirs(out, exist_ok=True)
    if m == "host-B": sh(["rsync", "-a", "-e", " ".join(SSH), "--include=*/", "--include=*.summary.json", "--include=*.npz", "--exclude=*", f"{PH}:{RR}/work_{tag}/out/", out + "/"]); return out
    return f"{LH}/work_{tag}/out"
def post_chunk(m, tag, chunk):
    """Amendment 1 (8 Nov 04:05, infrastructure only): only the jobs WITHOUT a result are re-queued, once; a job that has no result after its second attempt is a PERMANENT FAILURE (listed in PERMANENT_FAILURES_<prefix>.txt; per the pre-registration its bundle is incomplete and is dropped from all comparisons).  Jobs with results are never run twice."""
    hk = have_keys(tag); missing = [j for j in chunk if key(j["line"]) not in hk]
    with LOCK:
        DONE_N[0] += len(chunk) - len(missing)
        for j in missing:
            FAILS[j["line"]] = FAILS.get(j["line"], 0) + 1
            if FAILS[j["line"]] >= 2: PERM.append(j["line"])
            else: QUEUE.insert(0, dict(j, pin=None))
    say(m, "finished", tag, "| ok", len(chunk) - len(missing), "| without result", len(missing), "| done", DONE_N[0], "| permanent failures", len(PERM))
def adopt_chunk(m, a):
    """collect a chunk that was started by the previous scheduler version and is still running (no job is run twice)"""
    tag = a["tag"]; chunk = [{"line": l.strip(), "pin": None} for l in open(f"{N}/pull_{tag}.jobs") if l.strip()]; say(m, "adopts", tag)
    if m == "laptop":
        end = f"{LH}/work_{tag}/END"
        while not (os.path.exists(end) and open(end).read().strip()): time.sleep(30)
    else:
        while not subprocess.run(SSH + [PH, f"cat {RR}/work_{tag}/END 2>/dev/null"], capture_output=True, text=True).stdout.strip(): time.sleep(30)
    out = collect_partial(m, tag)
    if m == "laptop": shutil.copytree(out, f"{RES}/{tag}", dirs_exist_ok=True)
    post_chunk(m, tag, chunk)
def work(m):
    n = 0
    for a_ in plan.get("adopt", []):
        if a_["machine"] == m: adopt_chunk(m, a_)
    wi = plan.get("wait_idle")      # H0-S joins a machine the moment the first queue (log) is empty AND the machine has no simulator running: no gap between the two cohorts
    def _idle():
        t = open(f"{N}/{wi}").read(); q = re.findall(r"queue left (\d+)", t)
        if not q or int(q[-1]) != 0: return False
        if m == "kaggle": return t.count("kaggle claims") == t.count("kaggle finished") + t.count("kaggle FAILED chunk")
        cmd = "ps -eo args | grep -c '[r]un_episode_k07'"
        r = subprocess.run(SSH + [PH, cmd] if m == "host-B" else ["bash", "-c", cmd], capture_output=True, text=True); return int(r.stdout.strip or 1) == 0
    while wi and not (_idle() and (time.sleep(25) or _idle())): time.sleep(20)
    wf = plan.get("wait_for", {}).get(m)      # a machine that is still busy with the smoke test joins as soon as its smoke chunk is finished (no idle, no overload)
    while wf and not (os.path.exists(f"{N}/{wf[0]}") and any(open(f"{N}/{wf[0]}").read().count(x) for x in wf[1:])): time.sleep(30)
    while True:
        chunk = claim(m)
        if not chunk: break
        n += 1; tag = f"{P}{m[0]}{n:02d}"; jf = f"{N}/pull_{tag}.jobs"; open(jf, "w").write("".join(j["line"] + "\n" for j in chunk)); say(m, "claims", len(chunk), "jobs as", tag, "| queue left", len(QUEUE))
        try:
            try:
                out = RUNFN[m](tag, jf, WORKERS[m]) if WORKERS[m] else RUNFN[m](tag, jf)
            except RuntimeError as e_:
                if "round failed" not in str(e_) or m == "kaggle": raise
                say(m, "runner reported failures in", tag, "- collecting the partial results:", str(e_)[:120]); out = collect_partial(m, tag)
            if m == "laptop": shutil.copytree(out, f"{RES}/{tag}", dirs_exist_ok=True)
            elif os.path.dirname(out) != RES: shutil.copytree(out, f"{RES}/{tag}", dirs_exist_ok=True)
            post_chunk(m, tag, chunk)
        except Exception as e:
            say(m, "FAILED chunk", tag, repr(e)[:300])
            with LOCK:
                for j in chunk:
                    FAILS[j["line"]] = FAILS.get(j["line"], 0) + 1
                    if FAILS[j["line"]] >= 3: ERR.append(j["line"])
                    else: QUEUE.insert(0, dict(j, pin=None if j["pin"] == m and FAILS[j["line"]] >= 2 else j["pin"]))   # a pinned job that failed twice goes to the pool
            time.sleep(60)
            if ERR: return
try:
    say("pull scheduler start", P, "| jobs", len(plan["jobs"]))
    HK = {m_: have_keys(f"{P}{m_[0]}*") for m_ in ("host-B", "kaggle", "laptop")}; HA = have_keys("*") if plan.get("skip_in_all_dirs") else set.union(*HK.values); QUEUE.extend(j for j in plan["jobs"] if key(j["line"]) not in (HK[j["pin"]] if j["pin"] else HA)); say("to run", len(QUEUE), "of", len(plan["jobs"]), "(rest already complete)")
    for l_ in plan.get("prior_failed", []): FAILS[l_] = 1      # already failed once under the previous scheduler version
    ledger(sys.argv[1], os.path.abspath(__file__))
    ths = [threading.Thread(target=work, args=(m,)) for m in plan.get("machines", ["host-B", "kaggle", "laptop"])]
    for t in ths: t.start(); time.sleep(2)
    for t in ths: t.join()
    if ERR or (QUEUE and not (DEADLINE and time.time() > DEADLINE)): raise RuntimeError(f"unfinished: queue {len(QUEUE)}, failed {len(ERR)}")
    if PERM: open(f"{N}/PERMANENT_FAILURES_{P}.txt", "w").write("\n".join(PERM) + "\n")
    say("permanent failures:", len(PERM)); say("DONE")
except Exception as e:
    say("FAILED", repr(e), traceback.format_exc()[-500:].replace("\n", " | ")); open(f"{N}/FAILED_pull_{P}.txt", "w").write(repr(e)); sys.exit(1)
