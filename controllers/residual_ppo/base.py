"""The base controller under the residual: pid_water with a parameter set, plus a minimum on-time for every switch.

Split into propose() (the PID's action before the hold) and finish() (the hold, applied after the residual), so that
with a zero residual the policy is exactly cma_search.episode's policy.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAIN_KEYS = ("kp", "ki", "kd", "integral_limit", "zone_damper_full_k")
CLASS_KEYS = ("REHEAT_FULL_K", "RESET_FULL_K", "RESET_EFFORT_QUANTILE", "RESET_DEADBAND_K")


def cma_best():
    """best_params of the last CMA generation logged."""
    return json.loads((ROOT / "logs/cma/history.jsonl").read_text().splitlines()[-1])["best_params"]


class Floor:
    def __init__(self, meta, spec, params, PIDWaterResetPolicy, validate_action):
        self.spec, self.validate = spec, validate_action
        self.pid = PIDWaterResetPolicy(meta, spec, **{k: params[k] for k in GAIN_KEYS})
        for k in CLASS_KEYS:
            setattr(self.pid, k, params[k])
        self.hold = int(round(params["min_on_steps"]))
        self.on_for = np.zeros(meta["n_discrete_action"], int)

    def propose(self, obs):
        a = self.validate(self.pid.get_action(obs), self.spec)
        return a["continuous_action"], a["discrete_action"]

    def finish(self, cont, disc):
        d = disc.copy()
        if self.hold > 1:
            d = np.where((self.on_for > 0) & (self.on_for < self.hold), 1, d).astype(np.int32)
        self.on_for = np.where(d == 1, self.on_for + 1, 0)
        return {"continuous_action": cont, "discrete_action": d}


class ZTFloor:
    """The public per-zone PI controller (skynet/zt_controller.py) as the base; its own holds live inside it."""

    def __init__(self, meta, spec, validate_action):
        from skynet.zt_controller import ZTPolicy
        self.spec, self.validate = spec, validate_action
        self.pol = ZTPolicy(meta, spec)

    def propose(self, obs):
        a = self.validate(self.pol.get_action(obs), self.spec)
        return a["continuous_action"], a["discrete_action"]

    def finish(self, cont, disc):
        return {"continuous_action": cont, "discrete_action": disc}


class ZTOffFloor(ZTFloor):
    """zt in its PI hours; outside them the off-hours rule (every continuous command -1, recirculation 1, every switch
    off). `free` says whether the residual may act this step: the residual is trained and applied in PI hours only.
    PRESTART_H / EVENING_STOP_H (zt's window edges, hours) come from params when given."""

    def __init__(self, meta, spec, validate_action, params):
        super().__init__(meta, spec, validate_action)
        for k in ("PRESTART_H", "EVENING_STOP_H"):
            if k in params:
                setattr(self.pol, k, params[k])
        if "PRESTART_H" in params:                    # zt re-derives PRESTART_H from _base_prestart every step
            self.pol._base_prestart = params["PRESTART_H"]
        for k, v in params.get("attrs", {}).items():  # zt constants (tools/cma_zt.py tunes them)
            assert hasattr(self.pol, k), k
            setattr(self.pol, k, v)
            if k == "PRESTART_H":                     # see above: zt resets PRESTART_H from _base_prestart each step
                self.pol._base_prestart = v
        self.off_c = np.full(meta["n_continuous_action"], -1.0, np.float32)
        self.off_c[list(meta["recirculation_indices"])] = 1.0
        self.off_d = np.zeros(meta["n_discrete_action"], np.int32)
        self.free = True

    def propose(self, obs):
        pc, pd = super().propose(obs)                # zt runs every step so its own state stays current
        self.free = bool(self.pol._pi_active)
        return (pc, pd) if self.free else (self.off_c.copy(), self.off_d.copy())


class ZTOffAllFloor(ZTOffFloor):
    """ZTOffFloor whose residual may act at every hour: outside PI hours the off-hours rule is only the starting action,
    and the residual can override it (turn an AHU or the pump on, raise a setpoint)."""

    def propose(self, obs):
        pc, pd = super().propose(obs)
        self.free = True
        return pc, pd


def make_floor(meta, spec, params, PIDWaterResetPolicy, validate_action):
    """params is a pid_water parameter set (Floor), {"controller": "zt"} (ZTFloor), {"controller": "zt_off", ...}
    (ZTOffFloor) or {"controller": "zt_off_all", ...} (ZTOffAllFloor)."""
    if params.get("controller") == "zt_off_all":
        return ZTOffAllFloor(meta, spec, validate_action, params)
    if params.get("controller") == "zt_off":
        return ZTOffFloor(meta, spec, validate_action, params)
    if params.get("controller") == "zt":
        return ZTFloor(meta, spec, validate_action)
    return Floor(meta, spec, params, PIDWaterResetPolicy, validate_action)
