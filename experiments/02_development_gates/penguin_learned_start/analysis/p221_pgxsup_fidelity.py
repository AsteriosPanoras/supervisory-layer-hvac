"""p221: fidelity checks for run_episode_penguin_v3sup.py (Penguin ROS-P supervisor).  INTERNAL.  No reward, loss or score is printed: only equality flags, step indices and counts.

usage (after the experimenter ran the jobs in pgxsup/JOBS_FID_OLD.jobs with run_episode_penguin_v3.py and pgxsup/JOBS_FID_NEW.jobs with run_episode_penguin_v3sup.py):
  python -I p221_pgxsup_fidelity.py --old <out dir of the OLD stage> --new <out dir of the NEW stage> [--ref8640 <native 8640-step npz of the same scene>] [--report out.json]
  python -I p221_pgxsup_fidelity.py --selftest        (no simulator: drives the runner's own ROS classes with a synthetic building and checks the F4 replay against them)
F1  existing keys ('pgx@static@', 'pgx@ctx@TEST_FORCE_SHORT') through the new file are bit-identical to run_episode_penguin_v3.py (all npz arrays, switch log, final policy-state hash);
    the old runner's lines that the new file changed are listed (expected: 3 neutral lines); optional: the new native run equals the prefix of an existing 8640-step native run.
F2  the degenerate key (decision window empty: LMAX = LMAX_ID = 1.0) is bit-identical to 'pgx@static@'; its learning code ran (update events are counted).
F3  the active key equals native until its first knob change; the first differing action is in Penguin's own pre-conditioning hour (soon and not now) with a changed knob;
    only precondition_hours is ever moved, inside [LMIN, LMAX], set inside the decision window on a weekday, back to native at noon; Penguin's clock agrees with the env schedule.
F4  offline recomputation: D from the logged zone temperatures; an independent replay of the rule (decisions, leads, tau updates, model checks, resets) equals the logged arrays and events."""
import argparse, ast, glob, hashlib, json, math, os, sys
import numpy as np

B, W, STEPS = "building_52103", "fairbanks_3", 864
K_STATIC, K_CTX = "pgx@static@", "pgx@ctx@TEST_FORCE_SHORT"
K_DEGEN, K_ACTIVE = "pgx@sup@ROS=1,TR=0,LMAX=1.0,LMAX_ID=1.0", "pgx@sup@ROS=1,TR=0"
COMMON = ("rewards", "comp_keys", "comps", "occupancy", "ts", "actions_cont", "oat_c")
HERE = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(HERE)
RESULTS = []


def rec(check, ok, **info):
    RESULTS.append({"check": check, "ok": bool(ok), **info}); print(f"{'PASS' if ok else 'FAIL'}  {check}  " + "  ".join(f"{k}={v}" for k, v in info.items()), flush=True)


def tag(b, w, key, steps):
    sk = key.replace("@", "__").replace(",", "_").replace("=", "-")
    if len(sk) > 100: sk = sk[:50] + "_h" + hashlib.sha1(key.encode()).hexdigest()[:16]
    return f"{b}_{w}_{sk}_{steps}"


def load(d, key, b=B, w=W, steps=STEPS):
    fs = sorted(glob.glob(os.path.join(d, "**", tag(b, w, key, steps) + ".npz"), recursive=True))
    if len(fs) != 1: raise SystemExit(f"expected exactly one npz for {key} under {d}, found {len(fs)}")
    z = np.load(fs[0], allow_pickle=False); s = json.load(open(fs[0][:-4] + ".summary.json"))
    return {k: z[k] for k in z.files}, s


def same(a, b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def first_diff_row(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if np.asarray(a[i]).tobytes() != np.asarray(b[i]).tobytes(): return i
    return None if len(a) == len(b) else n


# ---------------- independent replay of ROS-P (written from the design note, not imported from the runner) ----------------
def D_from_zt(zt, lo, hi):
    dev = np.maximum(np.maximum(lo - zt, 0.0), zt - hi); D = dev.mean(axis=1)
    D[~np.isfinite(zt).all(axis=1)] = np.nan; return D


def replay(z, ros):
    P, ms, base = ros["params"], float(ros["opening_hour"]), float(ros["base_precondition_hours"])
    D, tod, dow, soon, now = z["ros_D"], z["ros_tod"], z["ros_dow"], z["ros_soon"], z["ros_now"]
    lt, n_inf, started, s_step, D_s, ext_bad, frozen, pred, last_start, cond_prev = math.log(P["TAU0"]), 0, False, None, None, 0, False, None, None, False
    knob = base; ph, tau, ev, sw = np.zeros(len(D)), np.zeros(len(D)), [], []
    for i in range(len(D)):
        wk = 0 <= int(dow[i]) < 5; t = float(tod[i]); d = float(D[i]); tau[i] = math.exp(lt)
        if math.isfinite(t):
            if wk and math.isfinite(d) and not started and not frozen:
                lmax = P["LMAX"] if n_inf >= 2 else P["LMAX_ID"]
                if ms - lmax <= t < ms - P["LMIN"]:
                    need = math.exp(lt) * math.log(max(d, P["DSTAR"]) / P["DSTAR"]); left = ms - t
                    if left <= need:
                        capped = (t - (ms - lmax)) < 0.5 / 12 and need > left
                        started = True; pred = None if capped else d * math.exp(-left / math.exp(lt)); knob = left; sw.append((i, left))
                        ev.append({"step": i, "event": "start", "lead_h": left, "need_h": need, "capped": capped})
            if t >= 12.0: started = False
            if (t >= 12.0 or frozen) and knob != base: knob = base; sw.append((i, base))
        ph[i] = knob
        # learning (after the controller acted)
        cond = bool(soon[i]) and not bool(now[i])
        if math.isfinite(t):
            if cond and not cond_prev and wk and t < ms: s_step, D_s, last_start = i, d, i
            if s_step is not None and i == s_step + 12:
                if math.isfinite(d) and D_s >= P["INF_D0"] and 0.0 < d < D_s:
                    lt = min(max((1 - P["ALPHA"]) * lt + P["ALPHA"] * math.log(1.0 / math.log(D_s / d)), math.log(P["TAU_LO"])), math.log(P["TAU_HI"])); n_inf += 1
                ev.append({"step": i, "event": "update", "tau_after_h": math.exp(lt)}); s_step = None
            if wk and ms <= t < ms + 1.0 and pred is not None:
                if last_start is None or i - last_start > 24: ev.append({"step": i, "event": "check_skipped"})
                else:
                    ext_bad = ext_bad + 1 if (math.isfinite(d) and d > pred + P["CHECK_MARGIN"]) else 0; ev.append({"step": i, "event": "check", "ext_bad": ext_bad})
                    if ext_bad >= 2: frozen = True; ev.append({"step": i, "event": "frozen"})
                pred = None
            cond_prev = cond
    return ph, tau, ev, sw, {"n_inf": n_inf, "frozen": frozen, "tau_final_h": math.exp(lt)}


def check_f4(z, s, label="F4"):
    ros = s["ros"]; lo, hi = ros["comfort_band_k"]
    Dr = D_from_zt(z["ros_zt"], lo, hi); okD = bool(np.array_equal(np.isnan(Dr), np.isnan(z["ros_D"]))) and float(np.nanmax(np.abs(Dr - z["ros_D"]), initial=0.0)) <= 1e-12
    rec(f"{label}a D recomputed from the logged zone temperatures", okD, max_abs_diff_k=float(np.nanmax(np.abs(Dr - z["ros_D"]), initial=0.0)))
    ph, tau, ev, sw, fin = replay(z, ros)
    rec(f"{label}b per-step precondition_hours replay", np.array_equal(ph, z["ros_ph"]), first_mismatch=first_diff_row(ph, z["ros_ph"]))
    rec(f"{label}c per-step tau replay", float(np.max(np.abs(tau - z["ros_tau"]), initial=0.0)) <= 1e-12, max_abs_diff_h=float(np.max(np.abs(tau - z["ros_tau"]), initial=0.0)))
    lg = ros["log"]; pairs = [(e["step"], e["event"]) for e in lg]; rpairs = [(e["step"], e["event"]) for e in ev]
    okv = pairs == rpairs
    for a, b in zip(lg, ev):
        if a["event"] == "start": okv &= abs(a["lead_h"] - b["lead_h"]) <= 1e-12 and abs(a["need_h"] - b["need_h"]) <= 1e-12 and bool(a["capped"]) == bool(b["capped"])
        if a["event"] == "update": okv &= abs(a["tau_after_h"] - b["tau_after_h"]) <= 1e-12
        if a["event"] == "check": okv &= a["ext_bad"] == b["ext_bad"]
    rec(f"{label}d decision / update / check events replay", okv, n_events=len(lg), n_starts=sum(e == "start" for _, e in pairs), n_updates=sum(e == "update" for _, e in pairs),
        n_checks=sum(e == "check" for _, e in pairs), n_check_skipped=sum(e == "check_skipped" for _, e in pairs))
    swl = [(int(e["step"]), float(e["new"])) for e in s["switch_log"] if e["knob"] == "precondition_hours" and int(e["step"]) >= 0]
    rec(f"{label}e switch log replay (step, value)", swl == [(int(a), float(b)) for a, b in sw], n_switches=len(swl))
    rec(f"{label}f final state replay", fin["n_inf"] == ros["n_informative"] and fin["frozen"] == ros["frozen"] and abs(fin["tau_final_h"] - ros["tau_final_h"]) <= 1e-12,
        n_informative=ros["n_informative"], frozen=ros["frozen"])


def check_f3(zs, ss, za, sa):
    ros = sa["ros"]; P, ms, base = ros["params"], float(ros["opening_hour"]), float(ros["base_precondition_hours"])
    sw = [e for e in sa["switch_log"] if int(e["step"]) >= 0]
    rec("F3a only precondition_hours is moved, no static shift at step -1", all(e["knob"] == "precondition_hours" for e in sa["switch_log"]) and not [e for e in sa["switch_log"] if int(e["step"]) < 0])
    if not sw:
        rec("F3b the rule acted in this window (otherwise F3 is vacuous)", False, note="no knob change: F3 vacuous, F2-like"); return
    s0 = int(sw[0]["step"]); d0 = first_diff_row(za["actions_cont"], zs["actions_cont"])
    pre = all(same(za[k][:s0], zs[k][:s0]) for k in ("rewards", "comps", "actions_cont", "oat_c", "occupancy"))
    rec("F3b identical to native before the first change", pre, first_change_step=s0, first_action_diff_step=d0)
    if d0 is not None:
        cond = bool(za["ros_soon"][d0]) and not bool(za["ros_now"][d0])
        rec("F3c first differing action: at/after the change, inside Penguin's pre-conditioning hour, with the knob changed", d0 >= s0 and cond and za["ros_ph"][d0] != base,
            step=d0, tod_h=float(za["ros_tod"][d0]), ph=float(za["ros_ph"][d0]))
    else: rec("F3c first differing action exists", False, note="actions identical over the whole run although the knob moved")
    okb = True
    for e in sw:
        i, v = int(e["step"]), float(e["new"]); t = float(za["ros_tod"][i]); wk = 0 <= int(za["ros_dow"][i]) < 5
        if v != base: okb &= (P["LMIN"] - 1e-12 <= v <= P["LMAX"] + 1e-12) and (ms - P["LMAX"] <= t < ms - P["LMIN"]) and wk and abs(v - (ms - t)) <= 1e-12
        else: okb &= (t >= 12.0) or bool(ros["frozen"])
    rec("F3d every change: lead in [LMIN, LMAX] set inside the weekday window as (opening - tod); every reset at noon or after a freeze", okb, n_switches=len(sw))
    okc, n_m = True, 0
    for i in range(1, len(za["ros_soon"])):
        c, cp = za["ros_soon"][i] and not za["ros_now"][i], za["ros_soon"][i - 1] and not za["ros_now"][i - 1]
        if c and not cp: n_m += 1; okc &= abs(float(za["ros_tod"][i]) - (ms - 1.0)) < 1e-9
    rec("F3e Penguin's clock (hod features) agrees with the env schedule: every pre-conditioning hour starts at opening - 1 h", okc and n_m > 0, n_mornings=n_m)


def code_diff(old, new):
    import difflib
    a, b = open(old).read().splitlines(), open(new).read().splitlines()
    removed = [l for l in difflib.ndiff(a, b) if l.startswith("- ")]
    rec("F1c old-runner lines changed by the new file (expected 3: the key assert, else->elif ctx, savez with optional ros arrays)", len(removed) == 3, n_removed=len(removed))
    for l in removed: print("      changed old line:", l[2:][:150])


def main_runs(a):
    zo, so = load(a.old, K_STATIC); zn, sn = load(a.new, K_STATIC)
    for key in (K_STATIC, K_CTX):
        zo, so = load(a.old, key); zn, sn = load(a.new, key)
        ok = all(same(zo[k], zn[k]) for k in COMMON) and not [k for k in zn if k.startswith("ros_")]
        ok &= so["status"] == sn["status"] == "OK" and so["steps_done"] == sn["steps_done"] == STEPS and so["switch_log"] == sn["switch_log"] and so["state_sha256"] == sn["state_sha256"]
        ok &= json.dumps(so["decisions"], sort_keys=True) == json.dumps(sn["decisions"], sort_keys=True)
        rec(f"F1a {key}: new file == v3 (arrays bit-identical, switch log, decisions, final policy-state hash)", ok, first_action_diff=first_diff_row(zo["actions_cont"], zn["actions_cont"]))
    zs, ss = load(a.new, K_STATIC)
    if a.ref8640:
        zr = np.load(a.ref8640, allow_pickle=False); n = len(zs["rewards"])
        ok = all(same(zs[k], zr[k][:n]) for k in ("rewards", "comps", "occupancy", "ts", "actions_cont", "oat_c")) and same(zs["comp_keys"], zr["comp_keys"])
        rec("F1b new native run == prefix of the existing 8640-step native run", ok, n=n)
    code_diff(os.path.join(N, "run_episode_penguin_v3.py"), os.path.join(N, "run_episode_penguin_v3sup.py"))
    zd, sd = load(a.new, K_DEGEN)
    ok = all(same(zd[k], zs[k]) for k in COMMON) and sd["state_sha256"] == ss["state_sha256"] and sd["switch_log"] == [] and sd["status"] == "OK"
    ok &= sd["ros"]["n_starts"] == 0 and bool(np.all(zd["ros_ph"] == sd["ros"]["native_precondition_hours"]))
    rec("F2 degenerate key (window empty) == pgx@static@", ok, n_updates=sum(e["event"] == "update" for e in sd["ros"]["log"]), n_informative=sd["ros"]["n_informative"])
    check_f4(zd, sd, "F4[degenerate]")
    za, sa = load(a.new, K_ACTIVE)
    rec("F3- active run status", sa["status"] == "OK" and sa["steps_done"] == STEPS, status=sa["status"], error=sa["error"])
    check_f3(zs, ss, za, sa); check_f4(za, sa, "F4[active]")


# ---------------- self-test without the simulator ----------------
def selftest():
    src = open(os.path.join(N, "run_episode_penguin_v3sup.py")).read(); tree = ast.parse(src); g = {"np": np, "math": math, "json": json, "hashlib": hashlib}
    keep = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name in ("LegalRangeError", "PenguinAdapter", "RecoveryStartCore", "PenguinROS"))
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("INTERFACE_TXT", "TR_NOT_IMPLEMENTED") for t in n.targets))]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "v3sup_classes", "exec"), g)
    INTERFACE = json.loads(g["INTERFACE_TXT"]); import datetime as dt

    class FakePenguin:                       # the attributes PenguinROS reads, with Penguin's semantics; a toy two-zone plant responds to the band (so the knob matters)
        def __init__(self):
            self.field_names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000", "comfort_mode_now", "comfort_mode_soon", "z0", "z1"]
            self.field_index = {n: i for i, n in enumerate(self.field_names)}; self.comfort_now_index, self.comfort_soon_index = 4, 5
            self.comfort_band, self.eco_band, self.start_hour, self.n_zones = (294.0, 297.0), (289.0, 298.0), 6.0, 2
            self.parameters = {"hot_water_target": 345.0, "precondition_hours": 1.0, "comfort_margin": 0.08, "gas_cost_ratio": 0.3, "comfort_cost": 65000.0}; self.HOLD_STEPS = 864
        def _zone_temperatures(self, o): return o[6:8].astype(float)
        def _hour(self, o): return (math.atan2(float(o[0]), float(o[1])) % (2.0 * math.pi)) * 12.0 / math.pi

    rep = {}
    for label, params, tau_b, holiday in (("cold_slow", {}, 2.5, None), ("degenerate", {"LMAX": 1.0, "LMAX_ID": 1.0}, 2.5, None), ("fast_building", {}, 0.4, None), ("holiday", {}, 2.5, 1)):
        p = FakePenguin(); ad = g["PenguinAdapter"](p, INTERFACE); ros = g["PenguinROS"](p, ad, params); T = np.array([292.0, 291.0]); t0 = dt.datetime(2023, 3, 26, 6, 0)
        for i in range(8640):
            ts = t0 + dt.timedelta(minutes=5 * i); h = ts.hour + ts.minute / 60.0; wd = ts.weekday(); day = (ts.date() - t0.date()).days
            comfort = lambda x: (wd < 5 and 6 <= x.hour < 19 and not (holiday is not None and (x.date() - t0.date()).days == holiday))
            now, soon = comfort(ts), comfort(ts + dt.timedelta(minutes=60))
            o = np.array([math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), math.sin(2 * math.pi * wd / 7), math.cos(2 * math.pi * wd / 7), float(now), float(soon), T[0], T[1]], dtype=np.float32)
            ros.before(i, o)
            # toy plant: the band Penguin would aim at (TransferPolicy._band semantics) and a first-order response with time constant tau_b
            if now: target = 294.0
            elif soon:
                prog = min(max(1.0 - max(0.0, (6.0 - h) % 24.0) / max(0.1, p.parameters["precondition_hours"]), 0.0), 1.0); target = 289.0 + prog * 5.0
            else: target = None
            out_t = 270.0 + 3.0 * math.sin(2 * math.pi * (day % 9) / 9.0)
            T = T + (5 / 60.0) * ((target - T) / tau_b if target is not None and (T < target).any() else (out_t - T) / 12.0)
            ros.after(i)
        z = {k: np.asarray(v) for k, v in ros.arrays().items()}; s = {"ros": json.loads(json.dumps(ros.summary())), "switch_log": ad.switch_log}
        print(f"--- selftest scene '{label}': starts={s['ros']['n_starts']} informative={s['ros']['n_informative']} frozen={s['ros']['frozen']} tau_final={s['ros']['tau_final_h']:.3f} h")
        check_f4(z, s, f"SELFTEST[{label}] F4")
        rep[label] = s["ros"]["n_starts"]
    rec("SELFTEST degenerate window never acts", rep["degenerate"] == 0)
    rec("SELFTEST cold slow building extends mornings", rep["cold_slow"] > 0, n_starts=rep["cold_slow"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--old"); ap.add_argument("--new"); ap.add_argument("--ref8640"); ap.add_argument("--report"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest()
    else: main_runs(a)
    n_fail = sum(not r["ok"] for r in RESULTS); print(f"\n{len(RESULTS) - n_fail} PASS, {n_fail} FAIL")
    if a.report: json.dump(RESULTS, open(a.report, "w"), indent=1)
    sys.exit(1 if n_fail else 0)
