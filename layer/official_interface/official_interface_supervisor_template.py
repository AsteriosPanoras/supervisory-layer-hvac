# ===== SB26: official-interface adapter for K07 = G30 + supervisor (ROS + TR) [+ weather clustering] (cell 3 of the notebook) =====
# Same decision, learning and trim & respond code as run_episode_k07_v9supc.py ('ROS decision' before the controller acts; 'TR' right after it; 'ROS learning' after that),
# frozen 2026-11-10 (sha256 in BUILD_MANIFEST_SUP.json).  Public principles: optimal start / recovery-time prediction (Seem, Armstrong & Hancock 1989; ASHRAE Guideline 36) and
# trim & respond of the duct static-pressure setpoint from zone requests (ASHRAE Guideline 36).  Weather clustering (GATE_OAT_C < 90 only): once per morning, from the mean outside
# air observed over the last 24 h: warm -> no morning extension, cold -> usual bound, deep -> 6-h bound.  Fixed, non-optimised defaults.
import math, types
import numpy as np
from kaggle_evaluation import smart_buildings_inference_server

POLICY_KEY = "__POLICY_KEY__"           # k07@<G30 overrides>: the robust base
CONSTS = __CONSTS__                     # k07_submission_constants.json
ROS = __ROS__                           # frozen ROS defaults (+ GATE_OAT_C, DEEP_OAT_C, LMAX_DEEP; GATE_OAT_C 99 = no weather clustering)
TRP = __TRP__                           # frozen trim & respond defaults (ASHRAE Guideline 36)
INT_KEYS = {"HOLD_STEPS", "EMA_STEPS", "PRESTART_H"}
their_k07_policy = types.SimpleNamespace(AdaptiveColdPolicy=AdaptiveColdPolicy, _outside_kelvin=_outside_kelvin)
_name, _, _rest = POLICY_KEY.partition("@"); assert _name == "k07", POLICY_KEY
_overrides = {}
for _item in filter(None, _rest.split(",")):
    _k, _v = _item.split("="); _overrides[_k] = int(round(float(_v))) if _k in INT_KEYS else float(_v)
for _k in _overrides: assert hasattr(their_k07_policy.AdaptiveColdPolicy, _k), _k
assert 0.0 < ROS["LMIN"] <= ROS["LMAX_ID"] <= ROS["LMAX"] <= 6.0 and ROS["DSTAR"] > 0 and 0 < ROS["TAU_LO"] < ROS["TAU0"] < ROS["TAU_HI"]
assert ROS["DEEP_OAT_C"] < ROS["GATE_OAT_C"] and 0.0 <= ROS["LMAX_DEEP"] <= 6.0 and TRP["SPMIN"] <= TRP["SPMAX"] <= 2000.0
_BaseK07 = type("K07Submission", (their_k07_policy.AdaptiveColdPolicy,), {**CONSTS, **_overrides})


class OfficialK07Policy(_BaseK07, smart_buildings_inference_server.Policy):
    """One instance per scenario (the gateway builds a fresh policy on the first turn of every scenario): all state starts here."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        assert hasattr(self, "_base_prestart"), "the K07 policy has no _base_prestart"
        self._k = 0; self.unif_info = []; self._oat_hist = []
        self._LO, self._HI = (float(x) for x in self.comfort_temp_window); self._MS = float(self.morning_start_hour)
        self._ZI = [iz for c_ in self.ahu_controllers for _, iz, _ in c_["zones"]]
        self._PB = int(math.floor(ROS["LMIN"])); self._base_prestart = self._PB; self.PRESTART_H = self._PB
        self.RS = {"lt": math.log(ROS["TAU0"]), "n_inf": 0, "started": False, "s_step": None, "D_s": None, "ext_bad": 0, "frozen": False, "pred6": None, "pi_prev": False, "log": []}
        self._GATE = float(ROS.get("GATE_OAT_C", 99.0)); self.RS["gate_cls"] = None
        self._CT = self.ahu_controllers; self._HI_TR = float(self.comfort_temp_window[1]); self._NA = len(self._CT)
        self.TS = {"sp": [TRP["SPMIN"]] * self._NA, "pi_prev": False, "req_on": {}, "m": {}, "run_at_max": {}, "marked": {}, "frozen": [False] * self._NA, "ovr_steps": [0] * self._NA,
                   "pi_steps": [0] * self._NA, "day_ovr": {}, "day_req": {}, "req_steps": {}, "unresp_set": {}, "log": []}
        self._tr_ov_prev = None

    def _D_of(self, o):
        try:
            tz = [float(self._denormalize_zone_temp(float(o[j]))) for j in self._ZI]; return float(np.mean([max(self._LO - x, 0.0, x - self._HI) for x in tz]))
        except Exception:
            return float("nan")

    def _tod_of(self, o):
        a = math.atan2(float(o[self.hod_sin_index]), float(o[self.hod_cos_index])) % (2 * math.pi); return 24.0 * a / (2 * math.pi)

    def _oat_of(self, o):
        try:
            idx = getattr(self, "_oat_any", None)
            return float(their_k07_policy._outside_kelvin(self, o[idx])) - 273.15 if idx is not None else float("nan")
        except Exception:
            return float("nan")

    def get_action(self, obs):
        i, RS, TS, _MS, _PB, _GATE = self._k, self.RS, self.TS, self._MS, self._PB, self._GATE
        # ---- ROS decision: before the controller acts, from this step's observation (and the outside air observed before it) only ----
        _hour, _dow = self._get_hour_and_day(obs); _tod = self._tod_of(obs); _D = self._D_of(obs)
        if _dow < 5 and math.isfinite(_D) and not RS["started"] and not RS["frozen"]:
            _lmax = ROS["LMAX"] if RS["n_inf"] >= 2 else ROS["LMAX_ID"]; _fire, _why, _need = False, None, None
            _open = True
            if _GATE < 90.0:                                      # weather-cluster freedom (off when GATE_OAT_C >= 90)
                _ldeep = ROS["LMAX_DEEP"] if (RS["n_inf"] >= 2 and ROS["LMAX_DEEP"] > _lmax) else _lmax
                if RS["gate_cls"] is None and _MS - _ldeep <= _tod < _MS - ROS["LMIN"]:
                    _w = [x for x in self._oat_hist[-288:] if math.isfinite(x)]; _Tm = float(np.mean(_w)) if _w else float("nan")
                    RS["gate_cls"] = "nan" if not math.isfinite(_Tm) else ("deep" if _Tm < ROS["DEEP_OAT_C"] else "cold" if _Tm < _GATE else "warm")
                    RS["log"].append({"step": i, "event": "gate", "tod_h": _tod, "oat_mean_c": _Tm, "n_obs": len(_w), "cls": RS["gate_cls"], "lmax_h": (_ldeep if RS["gate_cls"] == "deep" else _lmax)})
                _open = RS["gate_cls"] in ("deep", "cold")
                if RS["gate_cls"] == "deep": _lmax = _ldeep
            if _open and _MS - _lmax <= _tod < _MS - ROS["LMIN"]:
                _need = math.exp(RS["lt"]) * math.log(max(_D, ROS["DSTAR"]) / ROS["DSTAR"])
                if (_MS - _tod) <= _need: _fire, _why = True, "need"
            elif _PB < ROS["LMIN"] and _MS - ROS["LMIN"] <= _tod < _MS: _fire, _why = True, "latest"
            if _fire:
                _v = int(_MS - _hour); self._base_prestart = _v; self.PRESTART_H = _v; RS["started"] = True
                _capped = _why == "need" and (_tod - (_MS - _lmax)) < 1.0 / 12 and _need > (_MS - _tod)
                RS["pred6"] = _D * math.exp(-(_MS - _tod) / math.exp(RS["lt"])) if (_why == "need" and not _capped) else None
                RS["log"].append({"step": i, "event": "start", "why": _why, "tod_h": _tod, "D_k": _D, "tau_h": math.exp(RS["lt"]), "prestart_set": _v, "lmax_h": _lmax, "need_h": _need, "capped": bool(_capped)})
        if _hour >= 12:
            if int(self._base_prestart) != _PB and not RS["frozen"]: self._base_prestart = _PB; self.PRESTART_H = _PB
            RS["started"] = False
            RS["gate_cls"] = None
        if RS["frozen"] and int(self._base_prestart) != 2: self._base_prestart = 2; self.PRESTART_H = 2
        raw = super().get_action(obs)
        # ---- TR: after the controller, writes only the static-pressure command of running AHUs in PI hours, only upward ----
        _pi = bool(getattr(self, "_pi_active", False)); _hr, _dw = self._get_hour_and_day(obs); _day = i // 288
        _row_ov = []
        if _pi and not TS["pi_prev"]: TS["sp"] = [TRP["SPMIN"]] * self._NA
        for _a, _c in enumerate(self._CT):
            _R = 0.0; _rz = {}
            for _vav, _iz, _ in _c["zones"]:
                try: _T = float(self._denormalize_zone_temp(float(obs[_iz]))); _u = float(getattr(self, "_zone_u", {}).get(_vav, float("nan")))
                except Exception: _T, _u = float("nan"), float("nan")
                if not (math.isfinite(_T) and math.isfinite(_u)) or not _pi: TS["req_on"][_vav] = False; continue
                _opn = _u >= TRP["U_REQ"] or (TS["req_on"].get(_vav, False) and _u >= TRP["U_REL"]); _on = _opn and _T > self._HI_TR; TS["req_on"][_vav] = _on
                _r = 0 if not _on else (2 if _T > self._HI_TR + TRP["HI_EXTRA_K"] else 1); _rz[_vav] = _r; _R += TS["m"].get(_vav, 1.0) * _r
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
                    if len(TS["log"]) < 2000 and (not self._tr_ov_prev or not self._tr_ov_prev[_a]): TS["log"].append({"step": i, "ahu": _a, "event": "override_on", "sp_pa": TS["sp"][_a], "R": _R})
                for _vav, _r in _rz.items():                                                   # unresponsive-zone rule (Guideline 36 rogue-zone provision)
                    if _r > 0 and TS["sp"][_a] >= TRP["SPMAX"]: TS["run_at_max"][_vav] = TS["run_at_max"].get(_vav, 0) + 1
                    else: TS["run_at_max"][_vav] = 0
                    if TS["run_at_max"][_vav] >= TRP["UNRESP_STEPS"]:
                        TS["marked"].setdefault(_vav, set()).add(_day); TS["run_at_max"][_vav] = 0
                        if len(TS["marked"][_vav]) >= TRP["UNRESP_DAYS"] and TS["m"].get(_vav, 1.0) != 0.0: TS["m"][_vav] = 0.0; TS["unresp_set"][str(_vav)] = i
                _kk = (_a, _day); TS["day_ovr"][_kk] = TS["day_ovr"].get(_kk, [0, 0]); TS["day_ovr"][_kk][0] += int(_ovr); TS["day_ovr"][_kk][1] += 1; TS["day_req"][_kk] = TS["day_req"].get(_kk, 0.0) + _R
            _row_ov.append(_ovr)
        if not _pi and TS["pi_prev"] and _dw < 5:                                          # end of a PI period: coarse guard per AHU
            for _a in range(self._NA):
                _ds = sorted(d for (a_, d) in TS["day_ovr"] if a_ == _a)[-int(TRP["GUARD_DAYS"]):]
                if len(_ds) == int(TRP["GUARD_DAYS"]) and not TS["frozen"][_a] and all(TS["day_ovr"][(_a, d)][0] > TRP["GUARD_FRAC"] * TS["day_ovr"][(_a, d)][1] for d in _ds) and TS["day_req"][(_a, _ds[-1])] >= TS["day_req"][(_a, _ds[0])]:
                    TS["frozen"][_a] = True; TS["sp"][_a] = TRP["SPMIN"]; TS["log"].append({"step": i, "ahu": _a, "event": "frozen"})
        TS["pi_prev"] = _pi; self._tr_ov_prev = _row_ov
        _oat = self._oat_of(obs)
        # ---- ROS learning: uses only what was observed up to this step ----
        _pi = bool(getattr(self, "_pi_active", False))
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
        self._oat_hist.append(_oat)
        self.unif_info = RS["log"]
        self._k += 1
        return raw
