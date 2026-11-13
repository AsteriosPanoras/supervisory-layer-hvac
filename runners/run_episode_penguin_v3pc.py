# v3pc: a NEW file = run_episode_penguin_v3sup.py + the key option PC=1
# (pgx@sup@ROS=0,TR=0,PC=1[,PC_<param>=v]).  Public principle: night pre-cooling of the building mass with cool outside air (economizer only).  Weather decides WHEN (cooling-season
# cluster: mean outside air observed over the last 24 h >= PC_GATE_C; and per step outside air <= mean zone temperature - PC_ECON_K); the building decides WHETHER/HOW MUCH (learned
# online: the zone-mean excess over the comfort maximum at the start of Penguin's own soon hour on mornings without pre-cooling; pre-cool only if its EWMA > PC_NEED_K).  Actuator:
# Penguin's own unoccupied upper band (parameters['standby_maximum']) := max(hi - PC_BELOW_HI_K, max zone T - PC_STEP_K, lo + 0.5) during the night window before weekday openings;
# Penguin's own model-predictive search chooses the actions (it records what it applies, so its online identification stays consistent).  Native value otherwise.
# Safety: two consecutive pre-cooled mornings with the zone mean below the comfort minimum at the soon-hour start -> pre-cooling off for the episode.  PC=0 or absent: as v3sup.
# PENGUIN OBSERVE-LEARN-ADAPT SUPERVISOR run_episode_penguin_v3sup.py (10 Nov 2026; INTERNAL design pgxsup/DESIGN_NOTE.md): a NEW file built from run_episode_penguin_v3.py (unchanged).
# One addition only: the key family 'pgx@sup@ROS=1,TR=0[,PARAM=value,...]', on top of the NATIVE configuration (= 'pgx@static@').  ROS-P = the K07 recovery-learning optimal start
# (run_episode_k07_v9ros.py, k07fb_design/DESIGN_PROPOSAL.md), same rule and same frozen defaults; the controller-specific part is only the adapter: zone temperatures, comfort band,
# opening hour, time of day / weekday and Penguin's own pre-conditioning flag are read from the policy object and the observation (read-only), and the ONE actuator is Penguin's
# 'precondition_hours' (interface range [0.25, 2.0] h; LMIN = the native value 1.0 h, LMAX = 2.0 h), set before Penguin acts, inside [opening - LMAX, opening - LMIN), back to native at noon.
# Penguin reads precondition_hours only inside its comfort_mode_soon hour (TransferPolicy._band); see the design note for what this knob can and cannot do.
# TR-P (G36 static-pressure trim & respond) is NOT implemented: Penguin already resets the duct static pressure by need every step (critical-zone sizing in _choose_ahu); TR=1 is refused.
# Public principle: optimal start / recovery-time prediction (Seem, Armstrong & Hancock 1989; ASHRAE Guideline 36).  Every existing key ('static', 'ctx') behaves exactly as in v3.
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Penguin-V2 runner v3 (N14).  Built by pgv2/build_runner_v3.py from: supervisor_core.py (generic) + adapter_penguin.py (controller-specific) + interface_penguin.json + ctx_specs.json (frozen).
# usage (same CLI as the other runners): python run_episode_penguin_v3.py <kit> <their_dir> <out> <building> <weather> <key> <steps>
# keys:  pgx@static@            reference (no change)            pgx@static@knob=value,knob=value   static shifts from step 0 (legal ranges enforced)            pgx@ctx@SPEC_ID   contextual supervisor with a frozen spec
# Decision at step i uses ONLY the scalar observed at steps < i (for i = 228: steps 0..227), exactly as the K07 gate (checked: identical prefix temperature on 18 cells).
# ---------------------------------------------------------------------------------------------------------------------------------------------------
import hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

kit, their_dir, out, building, weather, key, steps = sys.argv[1:8]
steps = int(steps); out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True); their_dir = str(Path(their_dir).resolve()); here = str(Path(__file__).resolve())
try: kit_head = subprocess.run(["git", "-C", kit, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception: kit_head = "nogit:evaluate.py-sha256=" + hashlib.sha256(Path(kit, "smart_buildings", "evaluate.py").read_bytes()).hexdigest()
os.chdir(kit); sys.path.insert(0, kit); sys.path.insert(0, their_dir)
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
import their_k07_policy as T


# ===== supervisor_core.py (generic) =====
"""GENERIC contextual supervisor core (controller-agnostic): no controller knob names, no fixed classes.
It needs an ADAPTER with  set(knob, value, step)  and  get(knob)  (legal ranges are enforced by the adapter) and a stream of one observed scalar per completed step.
A SPEC (data, not code) says: which scalar signal over which prefix window, when to decide (select-once at the end of the first window, or periodically), which knob to move around which reference by which step,
and the selector: sparse (one threshold, two levels) or continuous (clipped linear law) or force/none (tests).  A level L in [-1, 1] means value = ref + delta * L."""
import math


class LegalRangeError(ValueError):
    pass


class ContextSupervisor:
    def __init__(self, spec, adapter):
        self.spec, self.ad = spec, adapter
        sig = spec["signal"]; self.window = int(sig["window"]); self.agg = sig.get("agg", "mean"); assert self.agg == "mean", "only the mean is implemented"
        self.every = spec.get("every")                       # None = select-once at step == window; otherwise a decision every `every` steps from step == window
        self.sel = spec["selector"]; self.knob = spec.get("knob"); self.ref = spec.get("ref"); self.delta = spec.get("delta")
        self.history, self.decisions = [], []
        for k, v in spec.get("base", {}).items():           # the static base is part of the configuration, applied from step 0
            adapter.set(k, v, step=-1)

    def due(self, i):
        if i < self.window: return False
        return i == self.window if self.every is None else (i - self.window) % int(self.every) == 0

    def level(self, z):
        s = self.sel; t = s["type"]
        if t == "none": return None
        if t == "force": return float(s["level"])
        if t == "sparse": return float(s["below"]) if z < float(s["tau"]) else float(s["above"])
        if t == "continuous": return max(-1.0, min(1.0, float(s["a"]) + float(s["b"]) * (z - float(s["z0"]))))
        raise ValueError(t)

    def before_step(self, i):
        """called BEFORE the controller acts at step i; uses only the values observed at steps < i"""
        if not self.due(i): return None
        assert len(self.history) == i, f"history has {len(self.history)} values at step {i}"
        w = self.history[i - self.window:i]; assert len(w) == self.window
        rec = {"step": i, "n_values_used": len(w), "signal": None, "level": None, "value": None}
        if all(isinstance(x, float) and math.isfinite(x) for x in w):
            z = sum(w) / len(w); rec["signal"] = z; L = self.level(z)
            if L is not None and self.knob is not None:
                v = float(self.ref) + float(self.delta) * L; rec.update(level=L, value=v); self.ad.set(self.knob, v, step=i)
        self.decisions.append(rec); return rec

    def after_step(self, source_value):
        """called AFTER the controller acted: the scalar observed at this step becomes available for later decisions"""
        self.history.append(float(source_value))


def resolve_spec(spec, group):
    """a spec with 'by_fold' is resolved for the cell's group (here: its station) with the parameters fitted WITHOUT that fold"""
    if "by_fold" not in spec: return dict(spec), None
    f = spec["fold_of_station"][group]; r = dict(spec); r["selector"] = spec["by_fold"][f]["selector"]; return r, f

# ===== adapter_penguin.py (controller-specific) =====
"""CONTROLLER ADAPTER for the public Penguin 'Deep Fine Search Lean Margin' controller (policy source = cell 1 of the notebook, unchanged).  Everything controller-specific lives here and in interface_penguin.json."""
import hashlib
import numpy as np


class PenguinAdapter:
    def __init__(self, policy, interface):
        self.policy, self.iface = policy, interface["knobs"]; self.switch_log = []; self.ref = {k: self.get(k) for k in self.iface}

    def get(self, knob):
        e = self.iface[knob]; return self.policy.parameters[knob] if e["where"] == "parameters" else getattr(self.policy, e["attr"])

    def set(self, knob, value, step):
        e = self.iface[knob]; v = float(value)
        if not (float(e["lo"]) <= v <= float(e["hi"])): raise LegalRangeError(f"{knob}={v} outside the legal range [{e['lo']}, {e['hi']}] ({e['source']})")
        old = self.get(knob)
        if e["where"] == "parameters":
            assert knob in self.policy.parameters, knob; self.policy.parameters[knob] = v
        else: setattr(self.policy, e["attr"], int(round(v)) if e.get("integer") else v)
        self.switch_log.append({"step": int(step), "knob": knob, "old": old, "new": self.get(knob)})

    def source_value(self, obs):
        """the observed outdoor air temperature (C) of the first AHU sensor: the same observation as the K07 prefix (checked: identical on 18 cells)"""
        try: return float(self.policy._temperature(np.asarray(obs, dtype=float).reshape(-1), self.policy.outside_obs[0], "outside_air_temperature_sensor", 290.0)) - 273.15
        except Exception: return float("nan")

    def act(self, obs): return self.policy.get_action(obs)

    def fingerprint(self):
        p = self.policy; h = hashlib.sha256()
        for a in ("theta", "slow_temperature"):
            v = getattr(p, a, None)
            if v is not None: h.update(np.asarray(v, dtype=float).tobytes())
        h.update(repr((getattr(p, "step_count", None), getattr(p, "last_heating_step", None), len(getattr(p, "history", [])))).encode()); return h.hexdigest()

# ===== ros_core (generic, controller-agnostic: the K07 ROS rule of run_episode_k07_v9ros.py, 'ROS decision' / 'ROS learning', unchanged in substance) =====
class RecoveryStartCore:
    """Recovery-learning optimal start.  Per step it receives, from the observation the controller is about to use: the time of day tod (h, 5-minute grid), a weekday flag and
    D = mean over zones of max(lo - T_z, 0, T_z - hi) (K); after the controller acted, whether the controller's OWN morning conditioning started at this step.
    decide() returns the lead (h) to apply from this step, or None.  learn() updates tau (first-order recovery D(t_s + 1 h) = D(t_s) exp(-1 h / tau), log-EWMA) and runs the model check."""
    DEFAULTS = {"DSTAR": 0.3, "LMIN": None, "LMAX": 2.0, "LMAX_ID": None, "TAU0": 1.0, "ALPHA": 0.5, "TAU_LO": 0.3, "TAU_HI": 3.0, "INF_D0": 0.3, "CHECK_MARGIN": 0.5}

    def __init__(self, P, ms):
        self.P, self.ms = dict(P), float(ms)
        self.lt = math.log(self.P["TAU0"]); self.n_inf = 0; self.started = False; self.s_step = None; self.D_s = None
        self.ext_bad = 0; self.frozen = False; self.pred = None; self.last_start = None; self.cond_prev = False; self.log = []

    def tau(self): return math.exp(self.lt)

    def decide(self, i, tod, weekday, D):
        P, ms = self.P, self.ms
        if not (weekday and math.isfinite(D) and not self.started and not self.frozen): return None
        lmax = P["LMAX"] if self.n_inf >= 2 else P["LMAX_ID"]                         # identification window: at most LMAX_ID until 2 informative mornings
        if not (ms - lmax <= tod < ms - P["LMIN"]): return None
        need = self.tau() * math.log(max(D, P["DSTAR"]) / P["DSTAR"]); left = ms - tod   # hours of conditioning needed to reach D* at the opening; hours left
        if left > need: return None
        capped = (tod - (ms - lmax)) < 0.5 / 12 and need > left                      # first window step and the rule wanted more lead than the bound: no model check (K07 fix 1)
        self.started = True; self.pred = None if capped else D * math.exp(-left / self.tau())
        self.log.append({"step": i, "event": "start", "why": "need", "tod_h": tod, "D_k": D, "tau_h": self.tau(), "need_h": need, "lead_h": left, "lmax_h": lmax, "capped": bool(capped)})
        return left

    def learn(self, i, tod, weekday, D, cond):
        P, ms = self.P, self.ms
        if cond and not self.cond_prev and weekday and tod < ms:                        # the controller's own morning conditioning started at this step
            self.s_step, self.D_s, self.last_start = i, D, i
        if self.s_step is not None and i == self.s_step + 12:                          # one hour later (D of this step's observation)
            if math.isfinite(D) and self.D_s >= P["INF_D0"] and 0.0 < D < self.D_s:
                td = 1.0 / math.log(self.D_s / D)
                self.lt = min(max((1 - P["ALPHA"]) * self.lt + P["ALPHA"] * math.log(td), math.log(P["TAU_LO"])), math.log(P["TAU_HI"])); self.n_inf += 1
            self.log.append({"step": i, "event": "update", "D_s": self.D_s, "D_1h": D, "tau_after_h": self.tau(), "n_inf": self.n_inf}); self.s_step = None
        if weekday and ms <= tod < ms + 1.0 and self.pred is not None:                 # model check at the first step of the opening hour
            if self.last_start is None or i - self.last_start > 24:                    # the controller did not pre-condition this morning (e.g. a holiday): no evidence, no check
                self.log.append({"step": i, "event": "check_skipped", "D_k": D, "pred_k": self.pred})
            else:
                self.ext_bad = self.ext_bad + 1 if (math.isfinite(D) and D > self.pred + P["CHECK_MARGIN"]) else 0
                self.log.append({"step": i, "event": "check", "D_k": D, "pred_k": self.pred, "ext_bad": self.ext_bad})
                if self.ext_bad >= 2: self.frozen = True; self.log.append({"step": i, "event": "frozen"})
            self.pred = None
        self.cond_prev = cond

# ===== ros_adapter_penguin (controller-specific: what is read and which knob is moved) =====
class PenguinROS:
    """Reads (never writes) Penguin's state: _zone_temperatures(obs) (the same observed K values Penguin uses), comfort_band (metadata comfort_temp_window), start_hour (metadata
    morning_start_hour), _hour(obs) (hod_sin_000 / hod_cos_000), dow_sin_000 / dow_cos_000 and comfort_mode_soon / comfort_mode_now (Penguin pre-conditions iff soon and not now).
    The one actuator is the knob 'precondition_hours' through the adapter (legal range enforced): lead L -> precondition_hours = L."""
    KNOB = "precondition_hours"

    def __init__(self, policy, ad, params):
        p = policy; self.p, self.ad = p, ad
        self.lo, self.hi = (float(x) for x in p.comfort_band); self.ms = float(p.start_hour)
        self.i_ds, self.i_dc = p.field_index.get("dow_sin_000"), p.field_index.get("dow_cos_000")
        self.native = float(ad.ref[self.KNOB]); P = dict(RecoveryStartCore.DEFAULTS); P.update(params)
        if P["LMIN"] is None: P["LMIN"] = self.native                                   # never shorter than native
        if P["LMAX_ID"] is None: P["LMAX_ID"] = min(P["LMIN"] + 1.0, P["LMAX"])          # K07: base + 1 h until 2 informative mornings
        assert 0.0 < P["LMIN"] <= P["LMAX_ID"] <= P["LMAX"] <= 6.0 and P["DSTAR"] > 0 and 0 < P["TAU_LO"] < P["TAU0"] < P["TAU_HI"] and 0 < P["ALPHA"] <= 1, P
        self.base = float(P["LMIN"])
        if self.base != float(ad.get(self.KNOB)): ad.set(self.KNOB, self.base, step=-1)  # only if LMIN differs from native (not the default)
        self.core = RecoveryStartCore(P, self.ms); self.P = self.core.P; self.cur = None
        self.h = {k: [] for k in ("D", "tod", "dow", "soon", "now", "ph", "tau", "zt")}

    def observe(self, obs):
        p, o = self.p, np.asarray(obs, dtype=float).reshape(-1)
        try: zt = np.asarray(p._zone_temperatures(o), dtype=float)
        except Exception: zt = np.full(int(p.n_zones), np.nan)
        D = float(np.mean(np.maximum(np.maximum(self.lo - zt, 0.0), zt - self.hi))) if (zt.size and np.isfinite(zt).all()) else float("nan")
        try: tod = (round(float(p._hour(o)) * 12.0) / 12.0) % 24.0                     # Penguin's own clock, on the 5-minute grid
        except Exception: tod = float("nan")
        dow = -1
        if self.i_ds is not None and self.i_dc is not None and np.isfinite(o[self.i_ds]) and np.isfinite(o[self.i_dc]):
            dow = int(round(7.0 * (math.atan2(float(o[self.i_ds]), float(o[self.i_dc])) % (2.0 * math.pi)) / (2.0 * math.pi))) % 7   # 0 = Monday (pandas weekday)
        soon = bool(o[p.comfort_soon_index] > 0.5) if p.comfort_soon_index is not None else False
        now = bool(o[p.comfort_now_index] > 0.5) if p.comfort_now_index is not None else False
        return zt, D, tod, dow, soon, now

    def before(self, i, obs):
        """ROS decision: BEFORE Penguin acts at step i, from this step's observation only"""
        zt, D, tod, dow, soon, now = self.observe(obs); wk = 0 <= dow < 5; c = self.core; tau_now = c.tau()
        self.cur = (D, tod, wk, bool(soon and not now))
        if math.isfinite(tod):
            lead = c.decide(i, tod, wk, D)
            if lead is not None: self.ad.set(self.KNOB, lead, step=i)
            if tod >= 12.0: c.started = False
            if (tod >= 12.0 or c.frozen) and float(self.ad.get(self.KNOB)) != self.base: self.ad.set(self.KNOB, self.base, step=i)   # after the morning, or frozen: native
        for k, v in (("D", D), ("tod", tod), ("dow", dow), ("soon", int(soon)), ("now", int(now)), ("ph", float(self.ad.get(self.KNOB))), ("tau", tau_now), ("zt", zt)): self.h[k].append(v)

    def after(self, i):
        """ROS learning: AFTER Penguin acted; uses only what was observed up to this step"""
        D, tod, wk, cond = self.cur
        if math.isfinite(tod): self.core.learn(i, tod, wk, D, cond)

    def arrays(self):
        nz = max((len(z) for z in self.h["zt"]), default=0)
        zt = np.asarray([z if len(z) == nz else np.full(nz, np.nan) for z in self.h["zt"]], dtype=float).reshape(len(self.h["zt"]), nz)
        return {"ros_D": np.asarray(self.h["D"], dtype=float), "ros_tod": np.asarray(self.h["tod"], dtype=float), "ros_dow": np.asarray(self.h["dow"], dtype=np.int8),
                "ros_soon": np.asarray(self.h["soon"], dtype=np.int8), "ros_now": np.asarray(self.h["now"], dtype=np.int8), "ros_ph": np.asarray(self.h["ph"], dtype=float),
                "ros_tau": np.asarray(self.h["tau"], dtype=float), "ros_zt": zt}

    def summary(self):
        c = self.core
        return {"params": c.P, "base_precondition_hours": self.base, "native_precondition_hours": self.native, "comfort_band_k": [self.lo, self.hi], "opening_hour": self.ms,
                "actuator": "precondition_hours = lead (h); Penguin reads it only while comfort_mode_soon and not comfort_mode_now (TransferPolicy._band)",
                "n_informative": c.n_inf, "frozen": c.frozen, "tau_final_h": c.tau(), "n_starts": sum(1 for e in c.log if e["event"] == "start"), "log": c.log}

TR_NOT_IMPLEMENTED = ("Penguin resets the duct static pressure by need at every step: per AHU, P = clip(2000 Pa * max_z(needed_flow_z / Vmax_z)^2, 200, 2000) and per-zone dampers deliver "
                      "the needed flow at that P (their_k07_policy.py FineSupplySearchPolicy._choose_ahu, lines 750-755; TransferPolicy._choose_ahu lines 387-392; command lines 479-480). "
                      "K07's airflow limit (pressure held at the 200 Pa floor) does not exist here, so a G36 static-pressure trim & respond would duplicate Penguin's own critical-zone reset.")

# ===== ros_adapter_penguin_standby (experimenter, 10 Nov: the start-time actuator; ACT=standby) =====
class RecoveryStartCoreWide(RecoveryStartCore):
    """As RecoveryStartCore; the model check accepts a conditioning start up to LMAX before the opening (the standby actuator can start earlier than Penguin's own soon hour)."""
    def learn(self, i, tod, weekday, D, cond):
        P, ms = self.P, self.ms
        if cond and not self.cond_prev and weekday and tod < ms: self.s_step, self.D_s, self.last_start = i, D, i
        if self.s_step is not None and i == self.s_step + 12:
            if math.isfinite(D) and self.D_s >= P["INF_D0"] and 0.0 < D < self.D_s:
                td = 1.0 / math.log(self.D_s / D)
                self.lt = min(max((1 - P["ALPHA"]) * self.lt + P["ALPHA"] * math.log(td), math.log(P["TAU_LO"])), math.log(P["TAU_HI"])); self.n_inf += 1
            self.log.append({"step": i, "event": "update", "D_s": self.D_s, "D_1h": D, "tau_after_h": self.tau(), "n_inf": self.n_inf}); self.s_step = None
        if weekday and ms <= tod < ms + 1.0 and self.pred is not None:
            if self.last_start is None or i - self.last_start > int(round(12 * P["LMAX"])) + 2:
                self.log.append({"step": i, "event": "check_skipped", "D_k": D, "pred_k": self.pred})
            else:
                self.ext_bad = self.ext_bad + 1 if (math.isfinite(D) and D > self.pred + P["CHECK_MARGIN"]) else 0
                self.log.append({"step": i, "event": "check", "D_k": D, "pred_k": self.pred, "ext_bad": self.ext_bad})
                if self.ext_bad >= 2: self.frozen = True; self.log.append({"step": i, "event": "frozen"})
            self.pred = None
        self.cond_prev = cond

class PenguinROSStandby(PenguinROS):
    """Same rule; actuator = Penguin's standby band.  Penguin conditions toward its occupied band only inside the 60-min comfort_mode_soon hour; before it the band is
    [standby_minimum, standby_maximum].  At the step the rule fires, the standby band is set to the comfort band [lo, hi] (Penguin starts conditioning from that step) and
    precondition_hours to PH_HOLD (24 h) so that the soon-hour ramp stays near the comfort band; at noon (or when frozen) the native values return.  LMIN = 1 h = Penguin's own
    start (the soon hour); LMAX = 4 h as for K07.  Learning: conditioning starts at the fire step, or at the soon hour when the rule did not fire."""
    PH_HOLD = 24.0
    def __init__(self, policy, ad, params):
        P0 = dict(params); P0.setdefault("LMIN", 1.0); P0.setdefault("LMAX", 4.0)
        p = policy; self.p, self.ad = p, ad
        self.lo, self.hi = (float(x) for x in p.comfort_band); self.ms = float(p.start_hour)
        self.i_ds, self.i_dc = p.field_index.get("dow_sin_000"), p.field_index.get("dow_cos_000")
        self.native = float(ad.ref[self.KNOB]); P = dict(RecoveryStartCore.DEFAULTS); P.update(P0)
        if P["LMAX_ID"] is None: P["LMAX_ID"] = min(P["LMIN"] + 1.0, P["LMAX"])
        assert 0.0 < P["LMIN"] <= P["LMAX_ID"] <= P["LMAX"] <= 6.0 and P["DSTAR"] > 0 and 0 < P["TAU_LO"] < P["TAU0"] < P["TAU_HI"] and 0 < P["ALPHA"] <= 1, P
        self.base = self.native; self.core = RecoveryStartCoreWide(P, self.ms); self.P = self.core.P; self.cur = None; self.fired = False
        self.nat = {k: float(p.parameters[k]) for k in ("standby_minimum", "standby_maximum", "precondition_hours")}
        self.h = {k: [] for k in ("D", "tod", "dow", "soon", "now", "ph", "tau", "zt")}
    def _setp(self, k, v, i):
        old = float(self.p.parameters[k]); v = float(v)
        if old != v: self.p.parameters[k] = v; self.ad.switch_log.append({"step": int(i), "knob": k, "old": old, "new": v})
    def _restore(self, i):
        for k, v in self.nat.items(): self._setp(k, v, i)
    def before(self, i, obs):
        zt, D, tod, dow, soon, now = self.observe(obs); wk = 0 <= dow < 5; c = self.core; tau_now = c.tau()
        if math.isfinite(tod):
            lead = c.decide(i, tod, wk, D)
            if lead is not None:
                self._setp("standby_minimum", self.lo, i); self._setp("standby_maximum", self.hi, i); self._setp("precondition_hours", self.PH_HOLD, i); self.fired = True
            if tod >= 12.0: c.started = False; self.fired = False
            if tod >= 12.0 or c.frozen: self._restore(i)
        self.cur = (D, tod, wk, bool((soon and not now) or (self.fired and math.isfinite(tod) and tod < self.ms)))
        for k, v in (("D", D), ("tod", tod), ("dow", dow), ("soon", int(soon)), ("now", int(now)), ("ph", float(self.p.parameters["precondition_hours"])), ("tau", tau_now), ("zt", zt)): self.h[k].append(v)
    def summary(self):
        s = PenguinROS.summary(self); s.update({"actuator": "standby band := comfort band and precondition_hours := 24 h at the fired step; native values at noon / when frozen", "native_values": self.nat, "LMIN_h": self.P["LMIN"]}); return s


# ===== precool_adapter_penguin (experimenter, 11 Nov: night pre-cooling, PC=1) =====
class PenguinPrecool:
    DEFAULTS = {"PC_GATE_C": 11.0, "PC_ECON_K": 2.0, "PC_BELOW_HI_K": 0.5, "PC_STEP_K": 0.25, "PC_ALPHA": 0.5, "PC_NEED_K": 0.0, "PC_WINDOW_H": 5.0}
    def __init__(self, policy, params):
        p = policy; self.p = p; self.P = dict(self.DEFAULTS); self.P.update(params)
        self.lo, self.hi = (float(x) for x in p.comfort_band); self.ms = float(p.start_hour)
        self.i_ds, self.i_dc = p.field_index.get("dow_sin_000"), p.field_index.get("dow_cos_000")
        self.nat_max = float(p.parameters["standby_maximum"]); self.oat_hist = []
        self.need = None; self.gate_cls = None; self.night_pc = False; self.soon_prev = False; self.bad = 0; self.frozen = False; self.log = []; self.h = {"pc_on": [], "pc_max": [], "pc_zt": []}
    def _setmax(self, v, i):
        v = float(v)
        if float(self.p.parameters["standby_maximum"]) != v: self.p.parameters["standby_maximum"] = v
    def observe(self, obs):
        p, o = self.p, np.asarray(obs, dtype=float).reshape(-1)
        try: zt = np.asarray(p._zone_temperatures(o), dtype=float)
        except Exception: zt = np.full(int(p.n_zones), np.nan)
        try: tod = (round(float(p._hour(o)) * 12.0) / 12.0) % 24.0
        except Exception: tod = float("nan")
        dow = -1
        if self.i_ds is not None and self.i_dc is not None and np.isfinite(o[self.i_ds]) and np.isfinite(o[self.i_dc]):
            dow = int(round(7.0 * (math.atan2(float(o[self.i_ds]), float(o[self.i_dc])) % (2.0 * math.pi)) / (2.0 * math.pi))) % 7
        soon = bool(o[p.comfort_soon_index] > 0.5) if p.comfort_soon_index is not None else False
        now = bool(o[p.comfort_now_index] > 0.5) if p.comfort_now_index is not None else False
        return zt, tod, dow, soon, now
    def before(self, i, obs):
        """decision BEFORE Penguin acts at step i, from this step's observation and the outside air observed before it"""
        zt, tod, dow, soon, now = self.observe(obs); P = self.P; on, val = False, self.nat_max
        ok = zt.size and np.isfinite(zt).all() and math.isfinite(tod)
        w0, w1 = self.ms - 1.0 - P["PC_WINDOW_H"], self.ms - 1.0
        inwin = ok and 0 <= dow < 5 and w0 <= tod < w1
        if ok and tod >= 12.0: self.gate_cls = None; self.night_pc = False
        if inwin and not self.frozen:
            if self.gate_cls is None:
                w = [x for x in self.oat_hist[-288:] if math.isfinite(x)]; Tm = float(np.mean(w)) if w else float("nan")
                self.gate_cls = "nan" if not math.isfinite(Tm) else ("warm" if Tm >= P["PC_GATE_C"] else "cold")
                self.log.append({"step": i, "event": "gate", "tod_h": tod, "oat_mean_c": Tm, "n_obs": len(w), "cls": self.gate_cls, "need_k": self.need})
            if self.gate_cls == "warm" and self.need is not None and self.need > P["PC_NEED_K"]:
                oat = self.oat_hist[-1] + 273.15 if self.oat_hist and math.isfinite(self.oat_hist[-1]) else float("nan")
                zm = float(np.mean(zt))
                if math.isfinite(oat) and oat <= zm - P["PC_ECON_K"]:
                    on = True; val = max(self.hi - P["PC_BELOW_HI_K"], float(np.max(zt)) - P["PC_STEP_K"], self.lo + 0.5); self.night_pc = True
        self._setmax(val if on else self.nat_max, i)
        self.h["pc_on"].append(int(on)); self.h["pc_max"].append(float(self.p.parameters["standby_maximum"])); self.h["pc_zt"].append(zt)
        self.cur = (zt, tod, dow, soon, now, ok)
    def after(self, i):
        """learning AFTER Penguin acted: at the first step of Penguin's own soon hour on a weekday"""
        zt, tod, dow, soon, now, ok = self.cur
        first_soon = ok and soon and not now and not self.soon_prev and 0 <= dow < 5
        if first_soon:
            x = float(np.mean(zt)) - self.hi
            if self.night_pc:
                self.bad = self.bad + 1 if float(np.mean(zt)) < self.lo else 0
                self.log.append({"step": i, "event": "check", "zone_mean_k": float(np.mean(zt)), "lo_k": self.lo, "bad": self.bad})
                if self.bad >= 2: self.frozen = True; self._setmax(self.nat_max, i); self.log.append({"step": i, "event": "frozen"})
            elif self.gate_cls == "warm":                                          # learn the building's natural morning excess only on cooling-season mornings without pre-cooling
                self.need = x if self.need is None else (1 - self.P["PC_ALPHA"]) * self.need + self.P["PC_ALPHA"] * x
                self.log.append({"step": i, "event": "need", "excess_k": x, "need_k": self.need})
        self.soon_prev = bool(ok and soon)
    def arrays(self):
        nz = max((len(z) for z in self.h["pc_zt"]), default=0)
        return {"pc_on": np.asarray(self.h["pc_on"], dtype=np.int8), "pc_max": np.asarray(self.h["pc_max"], dtype=float),
                "pc_zt": np.asarray([z if len(z) == nz else np.full(nz, np.nan) for z in self.h["pc_zt"]], dtype=float).reshape(len(self.h["pc_zt"]), nz)}
    def summary(self):
        return {"params": self.P, "native_standby_maximum": self.nat_max, "comfort_band_k": [self.lo, self.hi], "opening_hour": self.ms, "need_final_k": self.need, "frozen": self.frozen,
                "precool_steps": int(sum(self.h["pc_on"])), "log": self.log}

# ===== main =====
INTERFACE_TXT = r'''{"controller": "Penguin Deep Fine Search Lean Margin (public notebook, cell 1)",
 "knobs": {
  "hot_water_target": {"where": "parameters", "lo": 310.0, "hi": 350.0, "source": "code: _native_action('supply_water_temperature_setpoint', ..., 310.0, 350.0)"},
  "precondition_hours": {"where": "parameters", "lo": 0.25, "hi": 2.0, "source": "code lower bound max(0.1, x); engineering-declared range 0.25-2.0 h"},
  "comfort_margin": {"where": "parameters", "lo": 0.0, "hi": 0.5, "source": "engineering-declared"},
  "gas_cost_ratio": {"where": "parameters", "lo": 0.0, "hi": 1.0, "source": "engineering-declared"},
  "comfort_cost": {"where": "parameters", "lo": 1000.0, "hi": 1000000.0, "source": "engineering-declared"},
  "HOLD_STEPS": {"where": "attribute", "attr": "HOLD_STEPS", "integer": true, "lo": 0, "hi": 999, "source": "declared: last_heating_step starts at -1000, a hold >= 1000 is active at step 0 (initialisation artefact found in the screening)"}
 }}
'''
SPECS_TXT = r'''{"CLUSTER_ALL18_FOR_EVALUATION_NOT_USED_IN_DEV": {"base": {"hot_water_target": 340.0}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "selector": {"above": 1, "below": -1, "tau": 12.889691055030173, "train_mean_ratio": 0.9982713017063118, "type": "sparse"}, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}, "CLUSTER_DEV": {"base": {"hot_water_target": 340.0}, "by_fold": {"A": {"fitted_on_cells": 7, "fitted_on_stations": ["anchorage22", "chicago22", "fargo22", "miami22", "saltlake22", "sanfrancisco22"], "selector": {"above": 1, "below": -1, "tau": 21.15997707969266, "train_mean_ratio": 0.999039930402798, "type": "sparse"}}, "B": {"fitted_on_cells": 11, "fitted_on_stations": ["albuquerque22", "boise22", "elpaso22", "kansascity22", "newyork22", "sandiego22", "tampa22"], "selector": {"above": 1, "below": -1, "tau": 12.065349230849975, "train_mean_ratio": 0.9975221032559193, "type": "sparse"}}}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}, "CONT_ALL18_FOR_EVALUATION_NOT_USED_IN_DEV": {"base": {"hot_water_target": 340.0}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "selector": {"a": 1.0, "b": 0.15000000000000002, "train_mean_ratio": 0.998323655362066, "type": "continuous", "z0": 17.167299781487024}, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}, "CONT_DEV": {"base": {"hot_water_target": 340.0}, "by_fold": {"A": {"fitted_on_cells": 7, "fitted_on_stations": ["anchorage22", "chicago22", "fargo22", "miami22", "saltlake22", "sanfrancisco22"], "selector": {"a": -1.0, "b": 0.18, "train_mean_ratio": 0.9990674678942326, "type": "continuous", "z0": 16.44201120553461}}, "B": {"fitted_on_cells": 11, "fitted_on_stations": ["albuquerque22", "boise22", "elpaso22", "kansascity22", "newyork22", "sandiego22", "tampa22"], "selector": {"a": 1.0, "b": 0.14, "train_mean_ratio": 0.9977748169049931, "type": "continuous", "z0": 17.628847057093108}}}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}, "TEST_FORCE_SHORT": {"base": {"hot_water_target": 340.0}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "selector": {"level": -1.0, "type": "force"}, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}, "TEST_ZERO": {"base": {"hot_water_target": 340.0}, "delta": 0.25, "fold_of_station": {"albuquerque22": "A", "anchorage22": "B", "boise22": "A", "chicago22": "B", "elpaso22": "A", "fargo22": "B", "kansascity22": "A", "miami22": "B", "newyork22": "A", "saltlake22": "B", "sandiego22": "A", "sanfrancisco22": "B", "tampa22": "A"}, "knob": "precondition_hours", "ref": 1.0, "selector": {"type": "none"}, "signal": {"agg": "mean", "name": "prefix_mean_outdoor", "window": 228}}}'''
INTERFACE = json.loads(INTERFACE_TXT); SPECS = json.loads(SPECS_TXT)
parts = key.split("@"); assert parts[0] == "pgx" and parts[1] in ("static", "ctx", "sup"), key
d = Path(os.environ["SB_SCENARIOS_DIR"]) / building / weather
env = ev.load_hybrid_environment(d / "config.gin"); tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False); env.metrics_path = None; env._occupancy_normalization_constant = 125.0
meta = ev.create_building_meta_data(env, d); aspec = env.action_spec(); policy = ev.initialize_policy(T.DeepFineLeanPolicy, meta, aspec); ad = PenguinAdapter(policy, INTERFACE); sup, fold, spec_sha, status, error = None, None, None, "OK", None
ros, ros_on, tr_on, ros_params, ros_act = None, None, None, None, "ph"; pc, pc_on, pc_params = None, 0, {}
try:
    if parts[1] == "static":
        for it in filter(None, (parts[2] if len(parts) > 2 else "").split(",")): k, v = it.split("="); ad.set(k, float(v), step=-1)
    elif parts[1] == "ctx":
        sp0 = SPECS[parts[2]]; spec_sha = hashlib.sha256(json.dumps(sp0, sort_keys=True).encode()).hexdigest()
        sp, fold = resolve_spec(sp0, weather.rsplit("_w", 1)[0]); sup = ContextSupervisor(sp, ad)
    else:                                                                               # 'sup': on top of the native configuration (no static shift)
        ros_params = {}
        for it in filter(None, (parts[2] if len(parts) > 2 else "").split(",")):
            k, v = it.split("=")
            if k == "ROS": ros_on = int(v)
            elif k == "ACT": ros_act = v
            elif k == "TR": tr_on = int(v)
            elif k == "PC": pc_on = int(v)
            elif k.startswith("PC_"): assert k in PenguinPrecool.DEFAULTS, f"unknown PC parameter {k}"; pc_params[k] = float(v)
            else: assert k in RecoveryStartCore.DEFAULTS, f"unknown pgx@sup parameter {k}"; ros_params[k] = float(v)
        assert ros_on in (0, 1) and tr_on in (0, 1), "pgx@sup needs ROS=0|1 and TR=0|1"
        if tr_on == 1: raise NotImplementedError("TR-P is not implemented for Penguin. " + TR_NOT_IMPLEMENTED)
        assert ros_act in ("ph", "standby"), ros_act
        if ros_on == 1: ros = PenguinROS(policy, ad, ros_params) if ros_act == "ph" else PenguinROSStandby(policy, ad, ros_params)
        assert pc_on in (0, 1) and not (pc_on == 1 and ros_on == 1), "PC=1 runs without ROS"
        if pc_on == 1: pc = PenguinPrecool(policy, pc_params)
except Exception as e: status, error = "ERROR", f"{type(e).__name__}: {e}"
tracker.reset(); obs = tracker.last_obs; t_start = str(env.building.current_timestamp); illegal, t_pol, wall0, done = 0, 0.0, time.perf_counter(), 0; act_hist, oat_hist = [], []
if pc is not None: pc.oat_hist = oat_hist                                              # v3pc: the gate reads the outside air observed BEFORE this step
if status == "OK":
    try:
        for i in range(steps):
            if ros is not None: ros.before(i, obs)                                     # ROS decision (sets precondition_hours before Penguin acts)
            if pc is not None: pc.before(i, obs)                                       # PC decision (sets standby_maximum before Penguin acts)
            if sup is not None: sup.before_step(i)
            t = time.perf_counter(); raw = ad.act(obs); t_pol += time.perf_counter() - t
            sv = ad.source_value(obs); oat_hist.append(sv)
            if sup is not None: sup.after_step(sv)
            if ros is not None: ros.after(i)                                           # ROS learning (tau update, model check)
            if pc is not None: pc.after(i)                                             # PC learning (morning excess, safety check)
            a = ev.validate_action(raw, aspec); c, dsc = a["continuous_action"], a["discrete_action"]
            if not (np.isfinite(c).all() and c.min() >= -1.0 and c.max() <= 1.0 and set(int(x) for x in dsc) <= {0, 1}): illegal += 1
            act_hist.append(np.asarray(c, dtype=np.float32)); obs, _, term, trunc, _ = tracker.step(a); done = i + 1
            if (i + 1) % 1000 == 0: print(f"step {i + 1}/{steps} wall_s={time.perf_counter() - wall0:.0f} mean_reward={float(np.mean(tracker.rewards)):.6f}", flush=True)
            if term or trunc: status = f"ENV_ENDED_AT_{i + 1}"; break
    except Exception as e: status, error = "ERROR", f"{type(e).__name__}: {e}"
wall = time.perf_counter() - wall0; rewards = np.asarray(tracker.rewards, dtype=float)
sk = key.replace("@", "__").replace(",", "_").replace("=", "-")
if len(sk) > 100: sk = sk[:50] + "_h" + hashlib.sha1(key.encode()).hexdigest()[:16]
tag = f"{building}_{weather}_{sk}_{steps}"
comp_keys = sorted({k for r in tracker.reward_components for k, v in r.items() if isinstance(v, (int, float, np.floating))}); comps = np.asarray([[float(r.get(k, np.nan)) for k in comp_keys] for r in tracker.reward_components], dtype=float)
np.savez_compressed(out / f"{tag}.npz", rewards=rewards, comp_keys=np.asarray(comp_keys), comps=comps, occupancy=np.asarray(tracker.occupancy, dtype=float), ts=np.asarray([ts.isoformat() for ts in tracker.timestamps]), actions_cont=np.asarray(act_hist), oat_c=np.asarray(oat_hist, dtype=float), **(ros.arrays() if ros is not None else {}), **(pc.arrays() if pc is not None else {}))
n = len(rewards); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary = {"status": status, "error": error, "tag": tag, "building": building, "weather": weather, "policy_key": key, "policy_name": "penguin_v2_supervisor_v3", "steps_requested": steps, "steps_done": done, "n_rewards": n, "kit_commit": kit_head,
           "decisions": (sup.decisions if sup is not None else None), "switch_log": ad.switch_log, "reference_knobs": ad.ref, "fold": fold, "spec_id": (parts[2] if parts[1] == "ctx" else None), "spec_sha256": spec_sha, "interface_sha256": hashlib.sha256(INTERFACE_TXT.encode()).hexdigest(),
           "state_sha256": ad.fingerprint(), "code_sha256": {"run_episode_penguin_v3.py": sha(here), "their_k07_policy.py": sha(Path(their_dir) / "their_k07_policy.py")}, "scenario_dir": str(d), "scenario_config_sha256": sha(d / "config.gin"), "sim_start": t_start,
           "illegal_actions": illegal, "mean_reward": float(rewards.mean()) if n else None, "score_abs_mean_reward": float(abs(rewards.mean())) if n else None, "sum_reward": float(rewards.sum()) if n else None,
           "component_means": {k: float(np.nanmean(comps[:, j])) for j, k in enumerate(comp_keys)} if n else {}, "policy_ms_per_step": 1000.0 * t_pol / max(done, 1), "wall_s": wall, "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "note": "simulator result, one scenario; PUBLIC controller (Penguin deep-fine notebook) with the generic supervisor; not a leaderboard score"}
summary["runner_file"] = Path(here).name
if parts[1] == "sup":
    summary.update({"policy_name": "penguin_v3sup_ros", "sup": {"ROS": ros_on, "TR": tr_on, "ACT": ros_act, "params_given": ros_params}, "ros": (ros.summary() if ros is not None else None),
                    "tr": {"implemented": False, "reason": TR_NOT_IMPLEMENTED}, "pc": (pc.summary() if pc is not None else None)})
(out / f"{tag}.summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps({k: summary[k] for k in ("status", "error", "tag", "steps_done", "score_abs_mean_reward", "illegal_actions", "policy_ms_per_step", "wall_s")}), flush=True)
sys.exit(0 if status == "OK" else 3)
