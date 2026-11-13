# v9supe_gnn (12 Nov 2026, built by gnnsup/make_runner_gnn.py from v9supe_ax): the Residual PPO controller whose residual network can be
# recorded (GNN_REC_DIR, imitation data) or replaced by a graph network on the building's own structure (GNN_WEIGHTS).  The supervisor
# code below is unchanged.  Keys: k07@ and k07sup@;ROS=a,TR=b.
# v9supe_ax (11 Nov 2026, built by axsup/make_runner_ax.py from v9supe_zt): the controller is the public Residual PPO controller of
# alexchilton (Apache-2.0; zt rule of malyshevdanil + CMA constants + off-hours rule + residual PPO network, one shared net per entity
# type).  The supervisor code below is unchanged; it reads and writes the zt rule inside the controller (sky.floor.pol), and TR floors
# the final command after the residual.  Weights loaded with the safe loader (weights_only=True).  Keys: k07@ and k07sup@;ROS=a,TR=b.
# v9supe_zt (11 Nov 2026, maintenance test PREREG_ZTSUP_v1): run_episode_k07_v9supe.py with ONE change, the same as run_episode_k07_v9_zt.py made to v8: the base class of every
# policy is APResearchPolicy (the public zt-best controller of Malyshev Danil, Apache-2.0, the author's later version of K07) instead of AdaptiveColdPolicy.  The directory 'k07' of
# the experiment holds the zt-best classes under the file names their_k07_policy.py / k07_submission_constants.json (official_adapter/zt/their_zt).  Key 'k07@' = zt-best exactly;
# 'k07sup@;ROS=1,TR=1' = zt-best + the frozen supervisor, unchanged.
# v9supe: a NEW file = run_episode_k07_v9supd.py + ONE optional
# 'k07sup' parameter UNS_COOL_K (default 0 = off: every line as in v9supd).  With UNS_COOL_K > 0 (ASHRAE Guideline 36 unoccupied cooling setpoint, 305.4 K = 90 F): outside K07's
# own PI hours each zone's cooling limit is max(its sensor/eco limit, UNS_COOL_K), i.e. the empty building is not cooled below 32.2 C; heating side unchanged.  Inside PI hours
# (K07's pre-start and occupancy) nothing changes.  Combined with ROS=1 the learned optimal start decides how early the morning recovery begins.
# v9supd (10 Nov 2026, 23:55): a NEW file = run_episode_k07_v9supc.py + ONE optional parameter BEYOND_NOCHECK for 'k07sup' keys (default 0 = every line as in v9supc).
# With BEYOND_NOCHECK=1 a morning start whose needed lead exceeds the standard bound LMAX (4 h) - possible only through the deep-cold freedom - is NOT used by the model check,
# exactly as fix 1 excludes capped mornings: the first-order recovery model is not trusted beyond the lead range it was identified on.  Reason (development data only):
# with LMAX_DEEP = 6 the check froze the supervisor on 52103 fairbanks_3 after two long-lead deep-cold mornings (capacity-limited recovery).
# v9supc: a NEW file = run_episode_k07_v9sup.py + an optional WEATHER-CLUSTER
# FREEDOM for the morning supervisor (ROS) of 'k07sup' keys.  Parameters GATE_OAT_C (default 99 = off: every line then runs exactly as in v9sup), DEEP_OAT_C (default -15)
# and LMAX_DEEP (default 0 = none).  With GATE_OAT_C < 90, once per morning, at the first step of the widest possible lead window, the supervisor classifies the morning
# from the mean outside-air temperature the policy observed over the last 288 steps (24 h; fewer at the episode start): 'deep' if below DEEP_OAT_C, 'cold' if below
# GATE_OAT_C, else 'warm' ('nan' if no finite value).  warm/nan: no extension this morning (K07's own start, as G30).  cold: the usual lead bound.  deep: the lead bound
# LMAX_DEEP (if larger and after the identification mornings).  The class is logged as a 'gate' event and reset at noon.  TR is untouched.
# K07 OBSERVE-LEARN-ADAPT SUPERVISOR run_episode_k07_v9sup.py (10 Nov 2026; INTERNAL designs k07fb_design + k07tr_design): a NEW file built from run_episode_k07_v9ros.py (unchanged).
# Adds the key families 'k07tr@<overrides>;<TR params>' (request-driven duct static-pressure trim & respond per AHU, ASHRAE Guideline 36) and 'k07sup@<overrides>;ROS=0|1,TR=0|1,<params>'
# (both supervisors in one run).  TR acts AFTER the controller: it raises only the static-pressure command of a running AHU in PI hours, only upward, never above SPMAX (400 Pa), from
# zone requests (cooling PI output >= 0.95 and zone above the comfort band; 2 requests above +1 K; release below 0.85); +15 Pa per request (<= 32 Pa per step), -12 Pa per step without
# requests, reset to 200 Pa (G30's effective pressure) at the first PI step of each day; zones that keep requesting at SPMAX for 2 h on 2 days stop counting; an AHU overridden on > 90 % of
# its PI steps on 3 consecutive weekdays without fewer requests is frozen at 200 Pa.  K07's file and state are untouched.  'k07ros' keys behave exactly as in v9ros; every other key as in v9s.
# K07 RECOVERY-LEARNING OPTIMAL START run_episode_k07_v9ros.py (10 Nov 2026; INTERNAL design k07fb_design/DESIGN_PROPOSAL.md): a NEW file built from run_episode_k07_v9s.py (unchanged).
# One addition only: the key family 'k07ros@<k07 overrides>;<supervisor params>'.  The controller is built exactly as for 'k07@<overrides>'.  A supervisor reads, before the controller
# acts at step i, only the observation (zone temperatures vs the comfort band, the time of day and day of week) and K07's own pi_active state of the previous step; it learns this building's
# recovery time constant tau from its own weekday mornings (first-order recovery, log-EWMA) and starts the morning conditioning when tau*ln(max(D,D*)/D*) >= the time left to the opening, inside
# the lead window [LMIN, LMAX] h (5-minute resolution through PRESTART_H = opening hour - current hour).  Only PRESTART_H / _base_prestart ever change; reset to floor(LMIN) at noon.  Public principle:
# optimal start / recovery-time prediction (Seem, Armstrong & Hancock 1989; ASHRAE Guideline 36).  Not an optimised controller: fixed, physically motivated defaults.  Every other key behaves as in v9s.
# Fix 1 (10 Nov, from the fidelity run, before any outcome): a morning whose start was capped by the lead bound (the rule asked for more lead than LMAX_ID / LMAX) is not used by the model check.
# K07 SHRINKAGE VARIANT run_episode_k07_v9s.py (10 Nov 2026): a NEW file built from run_episode_k07_v9x.py (unchanged).  One addition only: the key suffix '-shrink50'.
# Every regime profile (E, C, M, H) is moved half way towards the robust global profile G30: for each real-valued parameter v = 0.5 * (regime value) + 0.5 * (G30 value),
# where a value not set by a profile is the one the unified policy would apply (B's override, else the controller default); integer parameters, and parameters that are switched off (None)
# on either side, keep the regime's value.
# Schedule, classes, thresholds and the start under B are unchanged; the blended table is written to the summary ('shrink_table').  Every other key behaves as in v9x.
"""PRE-CHECK VERSION of run_episode_k07_v2.py (4 Nov 2026): identical except that the validated continuous action of every step is saved in the npz as actions_cont.  Used only to verify, on DEVELOPMENT cells, the step at which a PRESTART_H change first changes any control action."""
"""SB26 paper runner, version 2 (2 Nov 2026, experiment E09): the same as run_episode_k07.py plus ONE addition, a passive per-step log of what the controller can observe.
For the default arm only (policy key 'k07', or LOG_FEATURES=1 in the environment) the npz also holds `feat` (steps x 8, float32) and `feat_names`: the mean and the spread of the zone
air temperatures, the mean and the maximum deviation of a zone from its own comfort band (K), the share of zones outside their band, the mean and the mean absolute PI output of K07's
zone loops (positive = cooling; NaN outside the PI hours) and whether the PI hours are on.  They are computed from the observation that the policy has just used and from K07's own
state after its action; nothing is written back, so every reward is the same as the one of run_episode_k07.py (checked by the dry run against the first steps of a loss-matrix run).
Everything below this paragraph is the text of run_episode_k07.py.

SB26 paper runner: the published K07 controller (classes extracted by k07/extract_k07_policy.py, exact text) on one scenario, locally or on Kaggle.

usage:  python -I run_episode_k07.py <kit_root> <their_dir> <out_dir> <building> <weather> <policy_key> <steps>
policy_key: k07 (SubmissionPolicy constants as published) or k07@NAME=VALUE,NAME=VALUE (the same with class-constant overrides), or
k07sw@CONST=NAME,V0=..,V1=..,V2=..,V3=.. (our causal switch: NAME follows K07's own 288-step outside-air EMA through the strata cuts 0/16/22 C).
A k07sw key may also carry fixed overrides (e.g. KI=0.0025), applied under the switch.  From 2026-11-01 (GEN2) it may also carry:
  CUTS=a_b_..  its own cuts in C, '_'-separated and ascending (then V0..Vn with n = the number of cuts; default 0/16/22 and V0..V3);
  EVERY=N      the regime is chosen only at steps 0, N, 2N, ... (EVERY=288: once a day, from the EMA at the day's first step); default 1.
  e.g. k07sw@CONST=COLD_SAT,CUTS=25.72,V0=-0.4,V1=-0.85,EVERY=288.  Keys without CUTS and EVERY behave exactly as before.
Copy of E03_pilot3_warm_side/run_episode_k07_p3.py with one change, as in the Kaggle runner of the authors' other experiments: when the
simulator is not a git checkout (the Kaggle competition mount), kit_commit is "nogit:evaluate.py-sha256=<sha256 of smart_buildings/evaluate.py>".
The episode loop and the summary fields are unchanged, so its runs compare with the pilots' runs.
From E05 on (2026-11-01) the npz also holds, per step, the outside-air temperature the policy observed (oat_c) and K07's own 288-step
outside-air EMA after that step's update (ema_c), both in C (NaN where the policy has none); they are read after get_action and change
nothing in the episode.  The summary lists them in trace_fields.

Third-party code (K07): on Kaggle it runs inside the run kernel the authors push (runner/kernel_runner.py); locally the authors start it, after reading every executed line.
Exit 0 on a complete run, 3 on any stop, with the reason in the summary.
"""
# N10-S9 RUNNER v7 = v6 + component ablations (modes dailyhot-noguard, dailyhot-noC, dailyhot-noM; also valid for the other modes)
# K07-X RUNNER v9x = v8 (M-gate) plus the key suffix -g30nonM (one candidate policy, TABLE change only)
# N10-S6 RUNNER v6 = v5 with update schedules (keys k07unif@listhot:<days joined by ->, k07unif@trighot:<N days>) and a per-day decision log (class of the day, active class before/after, switched, pending days)
# N10-S5b RUNNER v5 = v4 unified causal policy with the HOT component = hot2 incumbent (class H -> hot2 incumbent; keys k07unif@statichot / k07unif@dailyhot)
# N10-S RUNNER v3-log (5 Nov 2026): the N8 v1.1 rule runner (keys k07@..., k07rule@...) PLUS per-step logging of every zone's air temperature (K) and PI output: npz arrays zone_t_k (steps x zones, float32) and zone_u (steps x zones),
# read after get_action from the same observation the policy used; logging only READS (no policy, action or reward is changed; checked bitwise against the N8 runs in the pilot).
# N8 v1.1 AMENDMENT (4 Nov 2026, after the failed pilot P0): this file REPLACES run_episode_k07_v2_rule.py (kept unchanged as the record of v1).
# CAUSE OF THE FAILURE: the published K07 (k07/their_k07_policy.py l. 409, 421) stores self._base_prestart = self.PRESTART_H once in __init__ and rewrites self.PRESTART_H = EXTREME_PRESTART_H if extreme else self._base_prestart
# at every step in _update_regime(); PRESTART_H is read only by _pi_hours (l. 132).  v1 set only the attribute PRESTART_H, which was overwritten at the next step.  FIX (adapter only; the published K07 and the
# constant policies B and ALWAYS-1 are untouched): at step 228 the rule sets BOTH policy._base_prestart and policy.PRESTART_H.  The adapter now also logs per step, read after get_action and changing nothing,
# the internal states pi_active, PRESTART_H, _base_prestart, sum of the zone integrals and sum of the zone outputs (npz arrays pi_active_c, prestart_attr_c, base_prestart_c, zone_integral_sum_c, zone_u_sum_c).
# N8 PILOT P0 VERSION (4 Nov 2026) of run_episode_k07_v2_actions.py (the same runner plus the validated continuous action of every step in the npz as actions_cont).
# ONE ADDITION: policy key  k07rule@<K07 overrides>  = the run-time WEATHER RULE of N7/N8.  The policy starts exactly as k07@<overrides> (B has PRESTART_H = 2 from the K07
# constants) and, once, before the action of step RULE_STEP = 228, sets PRESTART_H = 1 if the mean of the outside-air temperature the policy OBSERVED at steps 0..227 (oat_c, the
# first 19 hours) is >= 11.0 C, and otherwise leaves it unchanged.  RULE_STEP, RULE_THRESHOLD_C and RULE_PRESTART_H are frozen; PRESTART_H may not be overridden in a k07rule key.
# Nothing else differs from the k07@ keys.  The summary holds rule{} (observed T19, decision, PRESTART_H before and after).
import hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path

import numpy as np

kit, their_dir, out, building, weather, key, steps = sys.argv[1:8]
steps = int(steps)
out = Path(out).resolve()
out.mkdir(parents=True, exist_ok=True)
their_dir = str(Path(their_dir).resolve())
here = str(Path(__file__).resolve())
os.chdir(kit)
sys.path.insert(0, kit)
sys.path.insert(0, their_dir)

def git(*a):
    return subprocess.run(["git", "-C", kit, *a], capture_output=True, text=True, check=True).stdout.strip()

try:  # Kaggle: the competition mount is not a git checkout; identify it by the sha256 of its evaluate.py instead
    kit_head = git("rev-parse", "HEAD")
    kit_dirty = git("status", "--porcelain", "--untracked-files=no")
except Exception:
    kit_head = "nogit:evaluate.py-sha256=" + hashlib.sha256(Path(kit, "smart_buildings", "evaluate.py").read_bytes()).hexdigest()
    kit_dirty = ""
from smart_buildings import evaluate as ev
from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
import shutil as _sh, tempfile as _tf
AXD = Path(os.environ['AX_DIR']).resolve()
if os.environ.get('AX_TORCH'): sys.path.append(os.environ['AX_TORCH'])
_pkg = Path(_tf.mkdtemp(prefix='axpkg_')); (_pkg / 'skynet').mkdir()
for _f in ('__init__.py', 'base.py', 'features.py', 'nets.py', 'policy.py', 'zt_controller.py'): _sh.copy(AXD / _f, _pkg / 'skynet' / _f)
sys.path.insert(0, str(_pkg))
import torch; torch.set_num_threads(1)
from skynet.policy import SkynetPolicy
from skynet.nets import FlatNets
from skynet import zt_controller as their_k07_policy      # trace_point reads _outside_kelvin from here
def _load_safe(p):
    ck = torch.load(p, map_location='cpu', weights_only=True)
    net = FlatNets.from_checkpoint(ck); net.hold = ck['args'].get('hold', 1)
    return net, ck['params'], ck['args']['scale'], ck['args'].get('latch', 1)
AX_NET, AX_PARAMS, AX_SCALE, AX_LATCH = _load_safe(AXD / 'skynet.pt')
GNN_MODE = 'rec' if os.environ.get('GNN_REC_DIR') else ('gnn' if os.environ.get('GNN_WEIGHTS') else None)
if GNN_MODE:
    sys.path.insert(0, str(Path(os.environ['GNN_DIR']).resolve())); import gnn_net as _gnn
    if GNN_MODE == 'gnn':
        _hold = AX_NET.hold; AX_NET = _gnn.load_safe(os.environ['GNN_WEIGHTS']); AX_NET.hold = _hold
    else:
        AX_NET = _gnn.Recorder(AX_NET)

RULE_STEP, RULE_THRESHOLD_C, RULE_PRESTART_H = 228, 11.0, 1     # FROZEN (N7/N8): decision at step 228 from the observed T19 of steps 0..227
INT_KEYS = {"HOLD_STEPS", "EMA_STEPS", "PRESTART_H"}
TABLE = {"B": "k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75", "E": "k07@RAMP=0.3096,COLD_SAT=-0.2481,KI=0.0058,KP=0.0714,HEAT_U_SCALE=0.8628,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=2.9775,MILD_SHIFT_K=0.8475,WATER_SCALE=1.0,WARM_SAT=-0.7764,MIN_DAMPER=0.013", "C": "k07@RAMP=0.3096,COLD_SAT=-0.2481,KI=0.0058,KP=0.0714,HEAT_U_SCALE=0.8628,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=2.9775,MILD_SHIFT_K=0.8475,WATER_SCALE=0.8842,WARM_SAT=-0.7764,MIN_DAMPER=0.013", "M": "k07@RAMP=0.25,COLD_SAT=-0.8,HEAT_U_SCALE=0.875,KP=0.075,NIGHT_HEAT_DROP_K=3,MILD_SHIFT_K=0.75,PRESTART_H=1", "H": "k07@RAMP=0.1711,COLD_SAT=-0.8432,KI=0.0051,KP=0.0671,HEAT_U_SCALE=0.864,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=2.9772,MILD_SHIFT_K=0.5459,FORCED_SAT=-0.5923,FORCED_PRESSURE=-0.9813,MIN_DAMPER=0.0728"}   # N10-S4 class table (frozen unified policy); classes by the observed mean outside temperature: E < -15, C < 11, M < 24.5, H >= 24.5   # PRESTART_H is compared with an integer hour
name, _, rest = key.partition("@")
ROS = None; TRP = None; ROS_ON = False; TR_ON = False
UNS_COOL_K = 0.0
if name in ("k07tr", "k07sup"):
    rest, _, _supp = rest.partition(";"); _sp = dict(_it.split("=") for _it in filter(None, _supp.split(",")))
    UNS_COOL_K = float(_sp.pop("UNS_COOL_K", "0"))                                  # v9supe: G36 unoccupied cooling setup (0 = off)
    ROS_ON = name == "k07sup" and _sp.pop("ROS", "0") == "1"; TR_ON = name == "k07tr" or _sp.pop("TR", "0") == "1"; _sp.pop("ROS", None); _sp.pop("TR", None)
    TRP = {"SPMIN": 200.0, "SPMAX": 400.0, "I": 0.0, "U_REQ": 0.95, "U_REL": 0.85, "HI_EXTRA_K": 1.0, "TRIM": 12.0, "RES": 15.0, "RESMAX": 32.0, "UNRESP_STEPS": 24.0, "UNRESP_DAYS": 2.0, "GUARD_FRAC": 0.9, "GUARD_DAYS": 3.0}
    ROS = {"DSTAR": 0.3, "LMIN": 2.0, "LMAX": 4.0, "LMAX_ID": 3.0, "TAU0": 1.0, "ALPHA": 0.5, "TAU_LO": 0.3, "TAU_HI": 3.0, "INF_D0": 0.3, "CHECK_MARGIN": 0.5, "GATE_OAT_C": 99.0, "DEEP_OAT_C": -15.0, "LMAX_DEEP": 0.0, "BEYOND_NOCHECK": 0.0}
    for _k, _v in _sp.items():
        assert _k in TRP or _k in ROS, f"unknown supervisor parameter {_k}"; (TRP if _k in TRP else ROS)[_k] = float(_v)
    if not ROS_ON: ROS = None
    assert TRP["SPMIN"] <= TRP["SPMAX"] <= 2000.0
    assert ROS is None or (ROS["DEEP_OAT_C"] < ROS["GATE_OAT_C"] and 0.0 <= ROS["LMAX_DEEP"] <= 6.0)
if name == "k07ros":
    ROS_ON = True
    rest, _, _rosp = rest.partition(";")
    ROS = {"DSTAR": 0.3, "LMIN": 2.0, "LMAX": 4.0, "LMAX_ID": 3.0, "TAU0": 1.0, "ALPHA": 0.5, "TAU_LO": 0.3, "TAU_HI": 3.0, "INF_D0": 0.3, "CHECK_MARGIN": 0.5}
    for _it in filter(None, _rosp.split(",")):
        _k, _v = _it.split("="); assert _k in ROS, f"unknown k07ros parameter {_k}"; ROS[_k] = float(_v)
    assert 0.0 < ROS["LMIN"] <= ROS["LMAX_ID"] <= ROS["LMAX"] <= 6.0 and ROS["DSTAR"] > 0 and 0 < ROS["TAU_LO"] < ROS["TAU0"] < ROS["TAU_HI"]
UNIF_MODE = None
if name == "k07unif":
    UNIF_MODE = rest; UNIF_ABL = None
    for _suf in ("-noguard", "-noC", "-noM", "-g30nonM", "-shrink50"):                      # component ablations (v7): the class entry is replaced, the rest of the table is unchanged
        if UNIF_MODE.endswith(_suf): UNIF_ABL = _suf; UNIF_MODE = UNIF_MODE[:-len(_suf)]
    if UNIF_ABL == "-noguard": TABLE["E"] = TABLE["C"]            # extreme cold uses the cold optimum WITHOUT the water-scale guard
    elif UNIF_ABL == "-noC": TABLE["C"] = TABLE["B"]
    elif UNIF_ABL == "-noM": TABLE["M"] = TABLE["B"]
    elif UNIF_ABL == "-g30nonM":                                   # K07-X candidate (v9x): the class profiles E, C and H are replaced by the global G30 profile; M keeps its profile and the M-gate (daily re-selection after an M day 1) is unchanged
        TABLE["E"] = TABLE["C"] = TABLE["H"] = "k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0"
    assert UNIF_MODE in ("statichot", "dailyhot", "mgatehot") or UNIF_MODE.startswith("listhot:") or UNIF_MODE.startswith("trighot:"), key
    UNIF_DAYS = set(int(x) for x in UNIF_MODE.split(":")[1].split("-")) if UNIF_MODE.startswith("listhot:") else set(); UNIF_HOLD = int(UNIF_MODE.split(":")[1]) if UNIF_MODE.startswith("trighot:") else 0
    UNIF_ACTIVE = None; UNIF_PEND = 0; UNIF_PEND_C = None
    MG_FLAG = False       # v8: M-gate = at the day-1 decision (step 228, observed mean of the first 228 outdoor readings) the class is read ONCE: M -> daily re-selection (PD5 rule) for the whole episode, otherwise select-once (PS5 rule)
    rest = TABLE["B"].partition("@")[2]
if name not in ("k07", "k07sup") or UNS_COOL_K != 0.0:
    raise SystemExit(f"v9supe_ax runs only k07@ and k07sup@;ROS=a,TR=b: {key!r}")
overrides = {}
for item in filter(None, rest.split(",")):
    k, v = item.split("=")
    overrides[k] = v if k in ("CONST", "CUTS") else int(round(float(v))) if k in INT_KEYS | {"EVERY"} else float(v)
assert not overrides, f"v9supe_ax takes no controller overrides: {overrides}"
consts = {}      # the class built below from their_k07_policy (= the controller's zt module) is never used
if name in ("k07", "k07rule", "k07unif", "k07ros", "k07tr", "k07sup"):
    if name == "k07rule" and "PRESTART_H" in overrides:
        raise SystemExit("PRESTART_H is set by the rule in a k07rule key")
    for k in overrides:
        if not hasattr(their_k07_policy.APResearchPolicy, k):
            raise SystemExit(f"override {k} is not a constant of APResearchPolicy")
    cls = type("K07Submission", (their_k07_policy.APResearchPolicy,), {**consts, **overrides})
else:
    SW_CUTS = tuple(float(c) for c in overrides["CUTS"].split("_")) if "CUTS" in overrides else (0.0, 16.0, 22.0)
    SW_EVERY = overrides.get("EVERY", 1)
    vkeys = sorted(k for k in overrides if k[0] == "V" and k[1:].isdigit())
    if vkeys != sorted(f"V{i}" for i in range(len(SW_CUTS) + 1)) or list(SW_CUTS) != sorted(SW_CUTS) or SW_EVERY < 1:
        raise SystemExit(f"k07sw needs ascending CUTS, V0..V{len(SW_CUTS)} and EVERY >= 1: {key!r}")
    SW_CONST, SW_VALUES = overrides["CONST"], [overrides[f"V{i}"] for i in range(len(SW_CUTS) + 1)]
    extra = {k: v for k, v in overrides.items() if k not in ("CONST", "CUTS", "EVERY", *vkeys)}
    for k in [SW_CONST, *extra]:
        if not hasattr(their_k07_policy.APResearchPolicy, k):
            raise SystemExit(f"{k} is not a constant of APResearchPolicy")
    base_cls = type("K07Submission", (their_k07_policy.APResearchPolicy,), {**consts, **extra})

    class K07StratumSwitch(base_cls):
        """Our causal switch: one K07 constant set from the stratum of K07's own outside-air EMA (EMA_STEPS = 288), through SW_CUTS_C
        (default 0/16/22 C), chosen at steps 0, SW_EVERY, 2 SW_EVERY, ... and kept in between."""
        SW_CUTS_C = SW_CUTS

        def __init__(self, building_meta_data, action_spec):
            super().__init__(building_meta_data, action_spec)
            self._sw_ema, self._sw_step, self._sw_k = None, 0, None
            self.switch_counts = [0] * (len(SW_CUTS) + 1)

        def _base_action(self, obs):
            if self._oat_any is not None:
                t = their_k07_policy._outside_kelvin(self, obs[self._oat_any]) - 273.15
                self._sw_ema = t if self._sw_ema is None else self._sw_ema + (t - self._sw_ema) / self.EMA_STEPS
                if self._sw_k is None or self._sw_step % SW_EVERY == 0:
                    self._sw_k = sum(self._sw_ema >= c for c in self.SW_CUTS_C)
                setattr(self, SW_CONST, SW_VALUES[self._sw_k])
                self.switch_counts[self._sw_k] += 1
            self._sw_step += 1
            return super()._base_action(obs)

    cls = K07StratumSwitch

def trace_point(policy, obs):
    """(observed outside air, K07's outside-air EMA) in C at this step, NaN when the policy has none; it never raises."""
    try:
        idx = getattr(policy, "_oat_any", None)
        oat = float(their_k07_policy._outside_kelvin(policy, obs[idx])) - 273.15 if idx is not None else float("nan")
    except Exception:
        oat = float("nan")
    try:
        ema = getattr(policy, "_oat_ema", None)
        ema = float(ema) if ema is not None else float("nan")
    except Exception:
        ema = float("nan")
    return oat, ema


FEAT_NAMES = ["zone_t_mean_c", "zone_t_spread_k", "band_dev_mean_k", "band_dev_max_k", "frac_zones_out", "pi_u_mean", "pi_u_absmean", "pi_active"]
LOG_FEATURES = os.environ.get("LOG_FEATURES", "1" if key == "k07" else "0") == "1"


def features(policy, obs):
    """Observation-derived response features at this step (see the docstring); NaN when anything is missing; it never raises and never writes."""
    nan = float("nan")
    try:
        zones = [z for ctrl in policy.ahu_controllers for z in ctrl["zones"]]
        tk = np.array([float(policy._denormalize_zone_temp(float(obs[i]))) for _, i, _ in zones])
        lo_c, hi_c = policy.comfort_temp_window
        dev = []
        for (vav, _, _), t in zip(zones, tk):
            idx = getattr(policy, "_sensor_band_indices", {}).get(vav)
            lo = float(obs[idx[0]]) if idx and idx[0] is not None else lo_c
            hi = float(obs[idx[1]]) if idx and idx[1] is not None else hi_c
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                lo, hi = lo_c, hi_c
            dev.append(max(lo - t, 0.0, t - hi))
        dev = np.asarray(dev)
        u = np.asarray(list(getattr(policy, "_zone_u", {}).values()), dtype=float)
        return [float(tk.mean() - 273.15), float(tk.max() - tk.min()), float(dev.mean()), float(dev.max()), float((dev > 0).mean()),
                float(u.mean()) if u.size else nan, float(np.abs(u).mean()) if u.size else nan, float(bool(getattr(policy, "_pi_active", False)))]
    except Exception:
        return [nan] * len(FEAT_NAMES)


safe_key = key.replace("@", "__").replace(",", "_").replace("=", "-")
if len(safe_key) > 100:      # N10-S: file names must stay below the 255-byte limit; long keys get a hash suffix (the full key is in summary.json policy_key)
    import hashlib as _hl; safe_key = safe_key[:50] + "_h" + _hl.sha1(key.encode()).hexdigest()[:16]
tag = f"{building}_{weather}_{safe_key}_{steps}"
summary_path, npz_path = out / f"{tag}.summary.json", out / f"{tag}.npz"

scen_base = Path(os.environ["SB_SCENARIOS_DIR"])
d = scen_base / building / weather
env = ev.load_hybrid_environment(d / "config.gin")
tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
meta = ev.create_building_meta_data(env, d)
spec = env.action_spec()
sky = SkynetPolicy(meta, spec, AX_NET, AX_PARAMS, AX_SCALE, PIDWaterResetPolicy, ev.validate_action, AX_LATCH)
if GNN_MODE: AX_NET.set_graph(sky.ix.zone_ahu, sky.ix.K)
policy = sky.floor.pol      # the zt rule inside the controller: the supervisor reads and writes it as on ZT Best
if UNS_COOL_K > 0:                                                                   # v9supe: outside K07 PI hours, cooling limit >= UNS_COOL_K
    _uns_orig_band = policy._zone_band
    def _uns_zone_band(o, vav_id, low, high, _f=_uns_orig_band):
        lo, hi = _f(o, vav_id, low, high)
        if not getattr(policy, "_pi_active", False): hi = max(hi, UNS_COOL_K)
        return lo, hi
    policy._zone_band = _uns_zone_band
    policy.eco_temp_window = (policy.eco_temp_window[0], max(float(policy.eco_temp_window[1]), UNS_COOL_K))
unif_info = []
ros_D, ros_tau = [], []
if ROS_ON:
    import math
    _LO, _HI = (float(x) for x in policy.comfort_temp_window); _MS = float(policy.morning_start_hour)
    _ZI = [iz for c_ in policy.ahu_controllers for _, iz, _ in c_["zones"]]
    _PB = int(math.floor(ROS["LMIN"]))                                      # K07's own start when the supervisor does nothing (LMIN = 2 -> exactly G30's PRESTART_H = 2)
    policy._base_prestart = _PB; policy.PRESTART_H = _PB
    RS = {"lt": math.log(ROS["TAU0"]), "n_inf": 0, "started": False, "s_step": None, "D_s": None, "ext_bad": 0, "frozen": False, "pred6": None, "pi_prev": False, "log": []}
    _GATE = float(ROS.get("GATE_OAT_C", 99.0)); RS["gate_cls"] = None
    def _D_of(o):
        try:
            tz = [float(policy._denormalize_zone_temp(float(o[j]))) for j in _ZI]; return float(np.mean([max(_LO - x, 0.0, x - _HI) for x in tz]))
        except Exception: return float("nan")
    def _tod_of(o):
        a = math.atan2(float(o[policy.hod_sin_index]), float(o[policy.hod_cos_index])) % (2 * math.pi); return 24.0 * a / (2 * math.pi)
if name == "k07unif":
    def _parse(k):
        d = {}
        for it in filter(None, k.partition("@")[2].split(",")):
            kk, vv = it.split("="); d[kk] = int(round(float(vv))) if kk in INT_KEYS else float(vv)
        return d
    CLS = {c: _parse(TABLE[c]) for c in "ECMH"}; BOV = _parse(TABLE["B"]); NAMES = sorted(set().union(*[set(d) for d in CLS.values()]) | set(BOV)); DEF = {n: getattr(policy, n) for n in NAMES}
    SHRINK_TABLE = None
    if UNIF_ABL == "-shrink50":                                    # v9s: every regime profile half way towards G30 (real-valued parameters); integers keep the regime value
        _G = _parse("k07@RAMP=0.3548,COLD_SAT=-1.0,KI=0.0068,KP=0.0699,HEAT_U_SCALE=0.8747,PRESTART_H=2,EVENING_STOP_H=0,NIGHT_HEAT_DROP_K=3.5886,MILD_SHIFT_K=0.7701,WATER_SCALE=0.9907,MIN_DAMPER=0.0")
        NAMES = sorted(set(NAMES) | set(_G)); DEF = {n: getattr(policy, n) for n in NAMES}
        _val = lambda d, n: d.get(n, BOV.get(n, DEF[n]))
        def _blend(c, n):
            rv, gv = _val(CLS[c], n), _val(_G, n)
            return rv if (n in INT_KEYS or rv is None or gv is None) else 0.5 * float(rv) + 0.5 * float(gv)      # a feature switched off (None) on either side is not blended: the regime value is kept
        CLS = {c: {n: _blend(c, n) for n in NAMES} for c in "ECMH"}
        SHRINK_TABLE = {c: dict(CLS[c]) for c in "ECMH"}
    def _classify(T): return "E" if T < -15 else "C" if T < 11 else "M" if T < 24.5 else "H"
    def _apply(c):
        for n in NAMES:
            v = CLS[c].get(n, BOV.get(n, DEF[n])); setattr(policy, n, v)
            if n == "PRESTART_H": policy._base_prestart = v
tr_sp, tr_rq, tr_kp, tr_ov = [], [], [], []
if TR_ON:
    import math
    _CT = policy.ahu_controllers; _HI_TR = float(policy.comfort_temp_window[1]); _NA = len(_CT)
    TS = {"sp": [TRP["SPMIN"]] * _NA, "pi_prev": False, "req_on": {}, "m": {}, "run_at_max": {}, "marked": {}, "frozen": [False] * _NA, "ovr_steps": [0] * _NA, "pi_steps": [0] * _NA,
          "day_ovr": {}, "day_req": {}, "req_steps": {}, "unresp_set": {}, "log": []}
    def TRS_summary():
        return {"params": TRP, "ahu_map": [{"vavs": [str(v) for v, _, _ in c["zones"]], "pressure_idx": int(c["pressure_idx"]) if c.get("pressure_idx") is not None else None, "run_discrete_idx": int(c["run_discrete_idx"]) if c.get("run_discrete_idx") is not None else None} for c in _CT],
                "override_steps": TS["ovr_steps"], "pi_steps": TS["pi_steps"], "frozen_ahus": [a for a in range(_NA) if TS["frozen"][a]], "unresponsive_zones": TS["unresp_set"],
                "request_hours": {str(k): v / 12.0 for k, v in TS["req_steps"].items()}, "log": TS["log"][:2000]}
policy_name = ('gnn_residual' if GNN_MODE == 'gnn' else 'alex_residual_ppo') + ('+sup' if (ROS_ON or TR_ON) else '')

tracker.reset()
obs = tracker.last_obs
t_start = str(env.building.current_timestamp)
disc_hist, illegal, t_policy = [], 0, 0.0
rule_info = None
act_hist = []        # ACTION LOG (pre-check version only): the validated continuous action of every step
oat_hist, ema_hist, feat_hist = [], [], []
st_pi, st_attr, st_base, st_zint, st_zu = [], [], [], [], []
zt_hist, zu_hist = [], []; NZ = None      # internal states, read after get_action (never written)
if not hasattr(policy, '_base_prestart'):
    raise SystemExit('the policy has no _base_prestart: the rule adapter does not fit this K07 file')
status, error, wall0 = "OK", None, time.perf_counter()
done_steps = 0
try:
    for i in range(steps):
        if name == "k07unif" and i >= RULE_STEP and (i - RULE_STEP) % 288 == 0:      # a decision opportunity every day (day 1 = step 228)
            k_ = (i - RULE_STEP) // 288; day_ = k_ + 1; seen_ = np.asarray(oat_hist[288 * k_:288 * k_ + RULE_STEP], dtype=float)
            if len(seen_) == RULE_STEP and np.isfinite(seen_).all():
                T_ = float(seen_.mean()); c_ = _classify(T_); before_ = UNIF_ACTIVE
                if day_ == 1:
                    apply_ = True
                    if UNIF_MODE == "mgatehot": MG_FLAG = bool(c_ == "M")
                elif UNIF_MODE.startswith("daily"): apply_ = True
                elif UNIF_MODE.startswith("static"): apply_ = False
                elif UNIF_MODE == "mgatehot": apply_ = MG_FLAG
                elif UNIF_MODE.startswith("listhot:"): apply_ = day_ in UNIF_DAYS
                elif UNIF_MODE.startswith("trighot:"):
                    if c_ != UNIF_ACTIVE: UNIF_PEND = (UNIF_PEND + 1) if UNIF_PEND_C == c_ else 1; UNIF_PEND_C = c_
                    else: UNIF_PEND = 0; UNIF_PEND_C = None
                    apply_ = UNIF_PEND >= UNIF_HOLD
                if apply_ and (c_ != UNIF_ACTIVE or day_ == 1): _apply(c_); UNIF_ACTIVE = c_; UNIF_PEND = 0; UNIF_PEND_C = None
                unif_info.append({"step": i, "day": day_, "t_obs_c": T_, "class": c_, "active_before": before_, "active_after": UNIF_ACTIVE, "switched": bool(UNIF_ACTIVE != before_ and day_ > 1), "pending_days": UNIF_PEND, "mgate_flag": (MG_FLAG if UNIF_MODE == "mgatehot" else None)})
            else: unif_info.append({"step": i, "day": day_, "t_obs_c": None, "class": None})
        if ROS_ON:                                                            # ROS decision: before the controller acts, from this step's observation only
            _hour, _dow = policy._get_hour_and_day(obs); _tod = _tod_of(obs); _D = _D_of(obs); ros_D.append(_D); ros_tau.append(math.exp(RS["lt"]))
            if _dow < 5 and math.isfinite(_D) and not RS["started"] and not RS["frozen"]:
                _lmax = ROS["LMAX"] if RS["n_inf"] >= 2 else ROS["LMAX_ID"]; _fire, _why, _need = False, None, None
                _open = True
                if _GATE < 90.0:                                      # v9supc weather-cluster freedom (off by default)
                    _ldeep = ROS["LMAX_DEEP"] if (RS["n_inf"] >= 2 and ROS["LMAX_DEEP"] > _lmax) else _lmax
                    if RS["gate_cls"] is None and _MS - _ldeep <= _tod < _MS - ROS["LMIN"]:
                        _w = [x for x in oat_hist[-288:] if math.isfinite(x)]; _Tm = float(np.mean(_w)) if _w else float("nan")
                        RS["gate_cls"] = "nan" if not math.isfinite(_Tm) else ("deep" if _Tm < ROS["DEEP_OAT_C"] else "cold" if _Tm < _GATE else "warm")
                        RS["log"].append({"step": i, "event": "gate", "tod_h": _tod, "oat_mean_c": _Tm, "n_obs": len(_w), "cls": RS["gate_cls"], "lmax_h": (_ldeep if RS["gate_cls"] == "deep" else _lmax)})
                    _open = RS["gate_cls"] in ("deep", "cold")
                    if RS["gate_cls"] == "deep": _lmax = _ldeep
                if _open and _MS - _lmax <= _tod < _MS - ROS["LMIN"]:
                    _need = math.exp(RS["lt"]) * math.log(max(_D, ROS["DSTAR"]) / ROS["DSTAR"])
                    if (_MS - _tod) <= _need: _fire, _why = True, "need"
                elif _PB < ROS["LMIN"] and _MS - ROS["LMIN"] <= _tod < _MS: _fire, _why = True, "latest"      # a non-integer LMIN: start at the latest allowed time
                if _fire:
                    _v = int(_MS - _hour); policy._base_prestart = _v; policy.PRESTART_H = _v; RS["started"] = True
                    _capped = _why == "need" and (_tod - (_MS - _lmax)) < 1.0 / 12 and _need > (_MS - _tod)     # fix 1: the rule wanted more lead than the bound allows -> no model check on this morning
                    _beyond = ROS.get("BEYOND_NOCHECK", 0.0) >= 0.5 and _why == "need" and _need is not None and _need > ROS["LMAX"]     # v9supd: beyond the standard bound -> no model check either
                    RS["pred6"] = _D * math.exp(-(_MS - _tod) / math.exp(RS["lt"])) if (_why == "need" and not _capped and not _beyond) else None
                    RS["log"].append({"step": i, "event": "start", "why": _why, "tod_h": _tod, "D_k": _D, "tau_h": math.exp(RS["lt"]), "prestart_set": _v, "lmax_h": _lmax, "need_h": _need, "capped": bool(_capped)})
            if _hour >= 12:
                if int(policy._base_prestart) != _PB and not RS["frozen"]: policy._base_prestart = _PB; policy.PRESTART_H = _PB
                RS["started"] = False
                RS["gate_cls"] = None
            if RS["frozen"] and int(policy._base_prestart) != 2: policy._base_prestart = 2; policy.PRESTART_H = 2       # model check failed twice: G30 schedule for the rest of the episode
        if name == "k07rule" and i == RULE_STEP:
            seen = np.asarray(oat_hist[:RULE_STEP], dtype=float)
            t19_obs = float(seen.mean()) if len(seen) == RULE_STEP and np.isfinite(seen).all() else float("nan")
            rule_info = {"step": RULE_STEP, "t19_observed_c": t19_obs, "threshold_c": RULE_THRESHOLD_C, "prestart_attr_before": int(policy.PRESTART_H), "base_prestart_before": int(policy._base_prestart), "switched": False}
            if np.isfinite(t19_obs) and t19_obs >= RULE_THRESHOLD_C:
                policy._base_prestart = RULE_PRESTART_H     # the state the published policy copies into PRESTART_H at every step
                policy.PRESTART_H = RULE_PRESTART_H
                rule_info["switched"] = True
            rule_info["prestart_attr_after"] = int(policy.PRESTART_H); rule_info["base_prestart_after"] = int(policy._base_prestart)
        t = time.perf_counter()
        raw = sky.get_action(obs)
        t_policy += time.perf_counter() - t
        if TR_ON:                                                             # TR: after the controller, writes only the static-pressure command of running AHUs in PI hours, only upward
            _pi = bool(getattr(policy, "_pi_active", False)); _hr, _dw = policy._get_hour_and_day(obs); _day = i // 288
            _row_sp, _row_rq, _row_kp, _row_ov = [], [], [], []
            if _pi and not TS["pi_prev"]: TS["sp"] = [TRP["SPMIN"]] * _NA                                    # first PI step of the day: back to G30
            for _a, _c in enumerate(_CT):
                _R = 0.0; _rz = {}
                for _vav, _iz, _ in _c["zones"]:
                    try: _T = float(policy._denormalize_zone_temp(float(obs[_iz]))); _u = float(getattr(policy, "_zone_u", {}).get(_vav, float("nan")))
                    except Exception: _T, _u = float("nan"), float("nan")
                    if not (math.isfinite(_T) and math.isfinite(_u)) or not _pi: TS["req_on"][_vav] = False; continue
                    _open = _u >= TRP["U_REQ"] or (TS["req_on"].get(_vav, False) and _u >= TRP["U_REL"]); _on = _open and _T > _HI_TR; TS["req_on"][_vav] = _on
                    _r = 0 if not _on else (2 if _T > _HI_TR + TRP["HI_EXTRA_K"] else 1); _rz[_vav] = _r; _R += TS["m"].get(_vav, 1.0) * _r
                    if _r: TS["req_steps"][_vav] = TS["req_steps"].get(_vav, 0) + 1
                _pidx, _ridx = _c.get("pressure_idx"), _c.get("run_discrete_idx")
                try: _kp = (float(raw["continuous_action"][_pidx]) + 1.0) * 1000.0 if _pidx is not None else float("nan")
                except Exception: _kp = float("nan")
                _ovr = False
                if _pi and not TS["frozen"][_a]:
                    TS["pi_steps"][_a] += 1
                    if _R > TRP["I"]: TS["sp"][_a] = min(TS["sp"][_a] + min(TRP["RES"] * (_R - TRP["I"]), TRP["RESMAX"]), TRP["SPMAX"])
                    else: TS["sp"][_a] = max(TS["sp"][_a] - TRP["TRIM"], TRP["SPMIN"])
                    _runs = True if _ridx is None else bool(int(raw["discrete_action"][_ridx]))
                    if _pidx is not None and _runs and math.isfinite(_kp) and TS["sp"][_a] > max(_kp, 200.0) + 0.5:
                        raw["continuous_action"][_pidx] = np.float32(TS["sp"][_a] / 1000.0 - 1.0); _ovr = True; TS["ovr_steps"][_a] += 1
                        if len(TS["log"]) < 2000 and (not tr_ov or not tr_ov[-1][_a]): TS["log"].append({"step": i, "ahu": _a, "event": "override_on", "sp_pa": TS["sp"][_a], "R": _R})
                    for _vav, _r in _rz.items():                                                   # unresponsive-zone rule (Guideline 36 rogue-zone provision)
                        if _r > 0 and TS["sp"][_a] >= TRP["SPMAX"]: TS["run_at_max"][_vav] = TS["run_at_max"].get(_vav, 0) + 1
                        else: TS["run_at_max"][_vav] = 0
                        if TS["run_at_max"][_vav] >= TRP["UNRESP_STEPS"]:
                            TS["marked"].setdefault(_vav, set()).add(_day); TS["run_at_max"][_vav] = 0
                            if len(TS["marked"][_vav]) >= TRP["UNRESP_DAYS"] and TS["m"].get(_vav, 1.0) != 0.0: TS["m"][_vav] = 0.0; TS["unresp_set"][str(_vav)] = i
                    _k = (_a, _day); TS["day_ovr"][_k] = TS["day_ovr"].get(_k, [0, 0]); TS["day_ovr"][_k][0] += int(_ovr); TS["day_ovr"][_k][1] += 1; TS["day_req"][_k] = TS["day_req"].get(_k, 0.0) + _R
                _row_sp.append(TS["sp"][_a]); _row_rq.append(int(_R)); _row_kp.append(_kp); _row_ov.append(_ovr)
            if not _pi and TS["pi_prev"] and _dw < 5:                                          # end of a PI period: coarse guard per AHU
                for _a in range(_NA):
                    _ds = sorted(d for (a_, d) in TS["day_ovr"] if a_ == _a)[-int(TRP["GUARD_DAYS"]):]
                    if len(_ds) == int(TRP["GUARD_DAYS"]) and not TS["frozen"][_a] and all(TS["day_ovr"][(_a, d)][0] > TRP["GUARD_FRAC"] * TS["day_ovr"][(_a, d)][1] for d in _ds) and TS["day_req"][(_a, _ds[-1])] >= TS["day_req"][(_a, _ds[0])]:
                        TS["frozen"][_a] = True; TS["sp"][_a] = TRP["SPMIN"]; TS["log"].append({"step": i, "ahu": _a, "event": "frozen"})
            TS["pi_prev"] = _pi; tr_sp.append(_row_sp); tr_rq.append(_row_rq); tr_kp.append(_row_kp); tr_ov.append(_row_ov)
        oat, ema = trace_point(policy, obs)
        try:
            zl = [z for ctrl in policy.ahu_controllers for z in ctrl['zones']]; zt_hist.append(np.array([float(policy._denormalize_zone_temp(float(obs[iz]))) for _, iz, _ in zl], dtype=np.float32)); zu_hist.append(np.array([float(getattr(policy, '_zone_u', {}).get(vav, np.nan)) for vav, _, _ in zl], dtype=np.float32))
        except Exception:
            zt_hist.append(np.array([], dtype=np.float32)); zu_hist.append(np.array([], dtype=np.float32))
        if ROS_ON:                                                            # ROS learning: uses only what was observed up to this step
            _pi = bool(getattr(policy, "_pi_active", False))
            if _pi and not RS["pi_prev"] and _dow < 5 and _hour < _MS: RS["s_step"], RS["D_s"] = i, _D
            if RS["s_step"] is not None and i == RS["s_step"] + 12:
                if math.isfinite(_D) and RS["D_s"] >= ROS["INF_D0"] and 0.0 < _D < RS["D_s"]:
                    _td = 1.0 / math.log(RS["D_s"] / _D)
                    RS["lt"] = min(max((1 - ROS["ALPHA"]) * RS["lt"] + ROS["ALPHA"] * math.log(_td), math.log(ROS["TAU_LO"])), math.log(ROS["TAU_HI"])); RS["n_inf"] += 1
                RS["log"].append({"step": i, "event": "update", "D_s": RS["D_s"], "D_1h": _D, "tau_after_h": math.exp(RS["lt"]), "n_inf": RS["n_inf"]}); RS["s_step"] = None
            if _dow < 5 and _hour == int(_MS) and RS["pred6"] is not None:
                RS["ext_bad"] = RS["ext_bad"] + 1 if (math.isfinite(_D) and _D > RS["pred6"] + ROS["CHECK_MARGIN"]) else 0
                RS["log"].append({"step": i, "event": "check", "D_k": _D, "pred_k": RS["pred6"], "ext_bad": RS["ext_bad"]}); RS["pred6"] = None
                if RS["ext_bad"] >= 2: RS["frozen"] = True; RS["log"].append({"step": i, "event": "frozen"})
            RS["pi_prev"] = _pi
        st_pi.append(float(bool(getattr(policy, '_pi_active', False)))); st_attr.append(float(policy.PRESTART_H)); st_base.append(float(policy._base_prestart))
        st_zint.append(float(sum(float(v) for v in getattr(policy, '_zone_integral', {}).values()))); st_zu.append(float(sum(float(v) for v in getattr(policy, '_zone_u', {}).values())))
        oat_hist.append(oat)
        ema_hist.append(ema)
        if LOG_FEATURES:
            feat_hist.append(features(policy, obs))
        a = ev.validate_action(raw, spec)
        c, dsc = a["continuous_action"], a["discrete_action"]
        if not (np.isfinite(c).all() and c.min() >= -1.0 and c.max() <= 1.0 and set(int(x) for x in dsc) <= {0, 1}):
            illegal += 1
        disc_hist.append(dsc.astype(np.int8)); act_hist.append(np.asarray(c, dtype=np.float32))
        obs, _, terminated, truncated, _ = tracker.step(a)
        done_steps = i + 1
        if (i + 1) % 1000 == 0:
            print(f"step {i + 1}/{steps} wall_s={time.perf_counter() - wall0:.0f} rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} "
                  f"mean_reward={float(np.mean(tracker.rewards)):.6f}", flush=True)
        if terminated or truncated:
            status = f"ENV_ENDED_AT_{i + 1}"
            break
except Exception as e:  # keep what was logged
    status, error = "ERROR", f"{type(e).__name__}: {e}"

wall = time.perf_counter() - wall0
rewards = np.asarray(tracker.rewards, dtype=float)
comp_keys = sorted({k for r in tracker.reward_components for k, v in r.items() if isinstance(v, (int, float, np.floating))})
comps = np.asarray([[float(r.get(k, np.nan)) for k in comp_keys] for r in tracker.reward_components], dtype=float)
np.savez_compressed(npz_path, rewards=rewards, comp_keys=np.asarray(comp_keys), comps=comps,
                    occupancy=np.asarray(tracker.occupancy, dtype=float), ts=np.asarray([ts.isoformat() for ts in tracker.timestamps]),
                    disc=np.asarray(disc_hist), actions_cont=np.asarray(act_hist), oat_c=np.asarray(oat_hist, dtype=float), ema_c=np.asarray(ema_hist, dtype=float),
                    zone_t_k=np.asarray([z if len(z) else np.full(max(len(q) for q in zt_hist), np.nan, np.float32) for z in zt_hist]), zone_u=np.asarray([z if len(z) else np.full(max(len(q) for q in zu_hist), np.nan, np.float32) for z in zu_hist]), pi_active_c=np.asarray(st_pi), prestart_attr_c=np.asarray(st_attr), base_prestart_c=np.asarray(st_base), zone_integral_sum_c=np.asarray(st_zint), zone_u_sum_c=np.asarray(st_zu),
                    **({"feat": np.asarray(feat_hist, dtype=np.float32), "feat_names": np.asarray(FEAT_NAMES)} if LOG_FEATURES else {}), **({"ros_D": np.asarray(ros_D, dtype=float), "ros_tau": np.asarray(ros_tau, dtype=float)} if ROS_ON else {}), **({"tr_sp_pa": np.asarray(tr_sp, dtype=np.float32), "tr_req": np.asarray(tr_rq, dtype=np.int16), "tr_k07_p": np.asarray(tr_kp, dtype=np.float32), "tr_ovr": np.asarray(tr_ov, dtype=bool)} if TR_ON else {}))
n = len(rewards)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary = {
    "status": status, "error": error, "tag": tag, "building": building, "weather": weather, "policy_key": key, "overrides": overrides,
    "policy_name": policy_name, "steps_requested": steps, "steps_done": done_steps, "n_rewards": n,
    "kit_commit": kit_head, "kit_dirty": kit_dirty, "numpy": np.__version__,
    "rule": rule_info, "unif": unif_info, "ros": ({"params": ROS, "base_prestart": _PB, "n_informative": RS["n_inf"], "frozen": RS["frozen"], "tau_final_h": math.exp(RS["lt"]), "log": RS["log"]} if ROS_ON else None), "tr": (TRS_summary() if TR_ON else None), "shrink_table": globals().get("SHRINK_TABLE"), "code_sha256": {"run_episode_k07_v9supe_ax.py": sha(here), **{f: sha(AXD / f) for f in ('skynet.pt', 'policy.py', 'nets.py', 'features.py', 'base.py', 'zt_controller.py')}},
    "scenario_dir": str(d), "scenario_config_sha256": sha(d / "config.gin"), "sim_start": t_start,
    "illegal_actions": illegal, "switch_counts": getattr(policy, "switch_counts", None), "trace_fields": ["oat_c", "ema_c"] + (["feat"] if LOG_FEATURES else []), "features_logged": LOG_FEATURES, "mean_reward": float(rewards.mean()) if n else None,
    "score_abs_mean_reward": float(abs(rewards.mean())) if n else None, "sum_reward": float(rewards.sum()) if n else None,
    "component_means": {k: float(np.nanmean(comps[:, j])) for j, k in enumerate(comp_keys)} if n else {},
    "policy_ms_per_step": 1000.0 * t_policy / max(done_steps, 1), "wall_s": wall,
    "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "note": "simulator result, one scenario; PUBLIC Residual PPO controller (alexchilton, Apache-2.0) with or without the unchanged supervisor; not a leaderboard score",
}
summary["gnn"] = {"mode": GNN_MODE, "weights_sha256": (sha(os.environ['GNN_WEIGHTS']) if GNN_MODE == 'gnn' else None),
                  "gnn_net_sha256": (sha(Path(os.environ['GNN_DIR']) / 'gnn_net.py') if GNN_MODE else None), "n_rec": (len(AX_NET.z) if GNN_MODE == 'rec' else None)}
if GNN_MODE == 'rec': AX_NET.save(Path(os.environ['GNN_REC_DIR']) / f"{tag}.rec.npz")
summary_path.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: summary[k] for k in ("status", "error", "tag", "steps_done", "score_abs_mean_reward", "illegal_actions", "policy_ms_per_step",
                                          "wall_s", "ru_maxrss_mb")}), flush=True)
sys.exit(0 if status == "OK" else 3)
