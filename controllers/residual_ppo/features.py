"""Index maps and per-entity features for one building, built from building_meta_data alone.

Entities: one per zone (VAV), one per air handler (AHU), one plant (the hot-water system).
Every building gives the same feature WIDTH per entity type, so one set of weights serves any building.
Built once in __init__; each step is numpy gathers.

Scaling: one fixed physical scale per field type (ranges measured on building_train_8, atlanta_1 night and phoenix_4 day,
2026-10-25). The zone temperature arrives normalised and is denormalised with the metadata's own constants.
No running statistics: they would be per building and would have to ship with the weights.
"""
import re

import numpy as np

# zone field suffix -> (offset, scale) applied as (x - offset) / scale
ZONE_FIELDS = [
    ("zone_air_temperature_sensor", None),                      # denormalised to K, then (K - 295) / 10
    ("zone_air_heating_temperature_setpoint_sensor", (295.0, 10.0)),
    ("zone_air_cooling_temperature_setpoint_sensor", (295.0, 10.0)),
    ("supply_air_damper_percentage_command", (0.0, 1.0)),
    ("reheat_valve_percentage_command", (0.0, 1.0)),
    ("supply_air_flowrate_setpoint", (0.0, 3.0)),
]
AHU_FIELDS = [
    ("supply_air_temperature_sensor", (295.0, 10.0)),
    ("supply_air_temperature_setpoint", (295.0, 10.0)),
    ("supply_air_static_pressure_sensor", (0.0, 1000.0)),
    ("supply_air_static_pressure_setpoint", (0.0, 1000.0)),
    ("supply_air_flowrate_sensor", (0.0, 3.0)),
    ("supply_air_flowrate_setpoint", (0.0, 3.0)),
    ("outside_air_flowrate_sensor", (0.0, 3.0)),
    ("outside_air_temperature_sensor", (295.0, 10.0)),
    ("recirculation_percentage_command", (0.0, 1.0)),
    ("supply_fan_speed_percentage_command", (0.0, 1.0)),
    ("discharge_fan_speed_percentage_command", (0.0, 1.0)),
    ("supervisor_run_command", (0.0, 1.0)),
    ("supply_fan_run_command", (0.0, 1.0)),
    ("exhaust_fan_run_command", (0.0, 1.0)),
    ("cooling_request_count", (0.0, 10.0)),
]
PLANT_FIELDS = [
    ("hws_supply_water_temperature_sensor", (330.0, 20.0)),
    ("hws_supply_water_temperature_setpoint", (330.0, 20.0)),
    ("hws_differential_pressure_sensor", (0.0, 20.0)),
    ("hws_differential_pressure_setpoint", (0.0, 1.0)),
    ("hws_run_status", (0.0, 1.0)),
    ("hws_supervisor_run_command", (0.0, 1.0)),
    ("hws_heating_request_count", (0.0, 30.0)),
]
GLOBAL_FIELDS = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000", "comfort_mode_now", "comfort_mode_soon"]

N_GLOBAL = len(GLOBAL_FIELDS) + 3          # + occupants, outside air temperature, in-comfort-hours flag
N_MEM = 8                                  # episode memory every entity sees, appended last (see BuildingIndex.features)
N_ZONE_MEM = 3                             # the zone's own memory: temperature change, its 1 h average, 1 h average reheat valve
N_ZONE = len(ZONE_FIELDS) + 2 + 2 + 2 + N_GLOBAL + N_ZONE_MEM + N_MEM   # own, band errors, PID damper/reheat, AHU supply temp + run
N_AHU = len(AHU_FIELDS) + 6 + 4 + N_GLOBAL + N_MEM        # own, pooled zones, PID temp/pressure/recirc/run
N_PLANT = len(PLANT_FIELDS) + 3 + 3 + N_GLOBAL + N_MEM    # own, pooled AHUs, PID water temp/pressure/pump
# Memory columns come after every older column, so a net that reads only the first columns of each entity (a checkpoint
# from before them, FlatNets n_in) sees exactly the features it was trained on.
EMA_1H, EMA_24H = 1.0 - np.exp(-1.0 / 12.0), 1.0 - np.exp(-1.0 / 288.0)    # per 5-minute step
ZONE_OUT, AHU_OUT, PLANT_OUT = 3, 4, 3                    # residuals: (damper, reheat, target offset), (temp, pressure, recirc, run vote),
                                                          # (water temp, water pressure, pump vote)
TARGET_K = 1.0                                            # K of zone-target offset per unit of the third zone output


def _suffix_num(name):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


class BuildingIndex:
    def __init__(self, meta):
        names = list(meta["observation_field_names"]); pos = {n: i for i, n in enumerate(names)}
        norm = meta["observation_normalization_metadata"]["zone_air_temperature_sensor"]
        self.t_mean, self.t_std = float(norm["sample_mean"]), float(norm["sample_std"])
        self.comfort, self.eco = meta["comfort_temp_window"], meta["eco_temp_window"]
        self.morning, self.evening = meta["morning_start_hour"], meta["evening_start_hour"]

        # zones in AHU order, exactly as the PID builds them (room_n is served by VAV_n)
        ahu_ids = sorted(meta["ahu_zone_assignments"], key=_suffix_num)[: len(meta["temp_indices"])]
        self.K = len(ahu_ids)
        zone_obs, zone_act, zone_ahu, zone_vav = [], [], [], []
        for k, ahu in enumerate(ahu_ids):
            for room in meta["ahu_zone_assignments"][ahu]:
                vav = room.replace("room_", "VAV_")
                if f"{vav}_zone_air_temperature_sensor" not in pos:
                    continue                                     # the PID skips a zone without a sensor; so do we
                acts = meta["zone_action_indices"].get(vav, {})
                zone_obs.append([pos[f"{vav}_{f}"] for f, _ in ZONE_FIELDS])
                zone_act.append([acts.get("supply_air_damper_percentage_command", -1), acts.get("reheat_valve_percentage_command", -1)])
                zone_ahu.append(k); zone_vav.append(vav)
        self.zone_obs, self.zone_act, self.zone_ahu = np.array(zone_obs), np.array(zone_act), np.array(zone_ahu)
        self.V = len(zone_ahu); self.zone_vav = zone_vav
        self.ahu_obs = np.array([[pos[f"ahs_{ahu}_{f}"] for f, _ in AHU_FIELDS] for ahu in ahu_ids])
        self.ahu_act = np.array([[meta["temp_indices"][k], meta["pressure_indices"][k], meta["recirculation_indices"][k]] for k in range(self.K)])
        self.plant_obs = np.array([pos[f] for f, _ in PLANT_FIELDS])
        self.water_temp_idx = meta["water_indices"][0]
        self.water_pressure_idx = meta["pressure_indices"][-1]
        self.n_discrete = meta["n_discrete_action"]; self.n_continuous = meta["n_continuous_action"]
        self.global_obs = np.array([pos[f] for f in GLOBAL_FIELDS])
        self.occ_idx = pos["num_occupants"]
        self.oat_idx = int(self.ahu_obs[0, [f for f, _ in AHU_FIELDS].index("outside_air_temperature_sensor")])
        self.hreq_idx = int(self.plant_obs[[f for f, _ in PLANT_FIELDS].index("hws_heating_request_count")])
        self.mem = None                                  # episode memory; one BuildingIndex per episode

        self.zone_off = np.array([0.0] + [s[0] for _, s in ZONE_FIELDS[1:]]); self.zone_scale = np.array([1.0] + [s[1] for _, s in ZONE_FIELDS[1:]])
        self.ahu_off = np.array([s[0] for _, s in AHU_FIELDS]); self.ahu_scale = np.array([s[1] for _, s in AHU_FIELDS])
        self.plant_off = np.array([s[0] for _, s in PLANT_FIELDS]); self.plant_scale = np.array([s[1] for _, s in PLANT_FIELDS])
        # members of each AHU, for mean pooling
        self.ahu_members = np.zeros((self.K, self.V)); self.ahu_members[self.zone_ahu, np.arange(self.V)] = 1.0
        self.ahu_members /= np.maximum(self.ahu_members.sum(1, keepdims=True), 1.0)

    def _band(self, obs):
        hour = int(np.floor(24 * (np.arctan2(obs[self.global_obs[0]], obs[self.global_obs[1]]) % (2 * np.pi)) / (2 * np.pi)))
        day = int(np.floor(7 * (np.arctan2(obs[self.global_obs[2]], obs[self.global_obs[3]]) % (2 * np.pi)) / (2 * np.pi)))
        occupied = day < 5 and self.morning <= hour < self.evening
        return (self.comfort if occupied else self.eco), float(occupied)

    def _occupancy_clock(self, obs):
        """(hours until the next occupied start, hours since the last occupied end); both 0 while occupied."""
        hour = 24 * (np.arctan2(obs[self.global_obs[0]], obs[self.global_obs[1]]) % (2 * np.pi)) / (2 * np.pi)
        day = int(np.floor(7 * (np.arctan2(obs[self.global_obs[2]], obs[self.global_obs[3]]) % (2 * np.pi)) / (2 * np.pi)))
        now = day * 24 + hour
        starts = [d * 24 + self.morning for d in range(-7, 15) if d % 7 < 5]
        ends = [d * 24 + self.evening for d in range(-7, 15) if d % 7 < 5]
        if day < 5 and self.morning <= hour < self.evening:
            return 0.0, 0.0
        return min(s - now for s in starts if s > now), min(now - e for e in ends if e <= now)

    def _remember(self, obs, temp_k):
        """Update the episode memory with this step's observation; called once per step, by features()."""
        valve = obs[self.zone_obs[:, 4]]; hreq = float(obs[self.hreq_idx]); oat = float(obs[self.oat_idx])
        m = self.mem
        if m is None:
            self.mem = dict(temp=temp_k, dtemp=np.zeros(self.V), dtemp_1h=np.zeros(self.V), valve_1h=valve.copy(),
                            since_heat=3 * 288.0, hreq_1h=hreq, heat_1h=valve.mean(), heat_24h=valve.mean(),
                            oat=oat, doat_1h=0.0, oat_24h=oat)
            return
        m["dtemp"] = temp_k - m["temp"]; m["temp"] = temp_k
        m["dtemp_1h"] += EMA_1H * (m["dtemp"] - m["dtemp_1h"])
        m["valve_1h"] += EMA_1H * (valve - m["valve_1h"])
        m["since_heat"] = 0.0 if hreq > 0 else m["since_heat"] + 1.0
        m["hreq_1h"] += EMA_1H * (hreq - m["hreq_1h"])
        m["heat_1h"] += EMA_1H * (valve.mean() - m["heat_1h"])
        m["heat_24h"] += EMA_24H * (valve.mean() - m["heat_24h"])
        m["doat_1h"] += EMA_1H * ((oat - m["oat"]) - m["doat_1h"]); m["oat"] = oat
        m["oat_24h"] += EMA_24H * (oat - m["oat_24h"])

    def features(self, obs, pid_cont, pid_disc):
        """(zone [V, N_ZONE], ahu [K, N_AHU], plant [N_PLANT]) float32 for one observation and the PID's proposal.
        Call it once per step, every step: it also updates the episode memory."""
        obs = np.asarray(obs, np.float64)
        (low, high), occupied = self._band(obs)
        g = np.concatenate([obs[self.global_obs], [obs[self.occ_idx] / 125.0, (obs[self.oat_idx] - 295.0) / 10.0, occupied]])

        zr = obs[self.zone_obs]
        temp_k = zr[:, 0] * self.t_std + self.t_mean
        zf = (zr - self.zone_off) / self.zone_scale; zf[:, 0] = (temp_k - 295.0) / 10.0
        band = np.stack([(temp_k - low) / 5.0, (high - temp_k) / 5.0], 1)
        pid_zone = np.where(self.zone_act >= 0, pid_cont[np.maximum(self.zone_act, 0)], 0.0)
        af = (obs[self.ahu_obs] - self.ahu_off) / self.ahu_scale
        ahu_ctx = np.stack([af[self.zone_ahu, 0], pid_disc[self.zone_ahu].astype(float)], 1)
        self._remember(obs, temp_k); m = self.mem
        to_occ, since_occ = self._occupancy_clock(obs)
        mg = np.array([min(m["since_heat"] / 288.0, 3.0), m["hreq_1h"] / 30.0, m["heat_1h"], m["heat_24h"],
                       m["doat_1h"] / 0.5, (m["oat_24h"] - 295.0) / 10.0, to_occ / 24.0, min(since_occ, 72.0) / 24.0])
        zmem = np.stack([m["dtemp"] / 0.2, m["dtemp_1h"] / 0.1, m["valve_1h"]], 1)
        zone = np.concatenate([zf, band, pid_zone, ahu_ctx, np.broadcast_to(g, (self.V, len(g))), zmem,
                               np.broadcast_to(mg, (self.V, N_MEM))], 1)

        err = np.minimum(band[:, 0], 0.0) + np.maximum(-band[:, 1], 0.0)           # <0 too cold, >0 too warm, 0 in band
        M = self.ahu_members
        cold = (band[:, 0] < 0).astype(float); warm = (band[:, 1] < 0).astype(float)
        pooled = np.stack([M @ err, M @ cold, M @ warm, M @ pid_zone[:, 0], M @ pid_zone[:, 1],
                           np.array([err[self.zone_ahu == k].max() if (self.zone_ahu == k).any() else 0.0 for k in range(self.K)])], 1)
        pid_ahu = np.concatenate([pid_cont[self.ahu_act], pid_disc[: self.K, None].astype(float)], 1)
        ahu = np.concatenate([af, pooled, pid_ahu, np.broadcast_to(g, (self.K, len(g))), np.broadcast_to(mg, (self.K, N_MEM))], 1)

        pf = (obs[self.plant_obs] - self.plant_off) / self.plant_scale
        plant = np.concatenate([pf, [err.mean(), cold.mean(), pid_disc[: self.K].mean()],
                                [pid_cont[self.water_temp_idx], pid_cont[self.water_pressure_idx], float(pid_disc[-1])], g, mg])
        clip = lambda x: np.clip(x, -10.0, 10.0).astype(np.float32)
        return clip(zone), clip(ahu), clip(plant)

    def target_offsets(self, zone_a, scale):
        """{VAV id: K} for the PI's next zone targets, from the third zone output (the PI clips to the band)."""
        return {v: float(TARGET_K * scale * zone_a[i, 2]) for i, v in enumerate(self.zone_vav)}

    def compose(self, pid_cont, pid_disc, zone_a, ahu_a, plant_a, scale):
        """Residual on the PID: zero raw output gives the PID's action exactly. Votes in the dead zone defer to the PID."""
        cont = pid_cont.astype(np.float64).copy()
        ok = self.zone_act[:, 0] >= 0; cont[self.zone_act[ok, 0]] += scale * zone_a[ok, 0]
        ok = self.zone_act[:, 1] >= 0; cont[self.zone_act[ok, 1]] += scale * zone_a[ok, 1]
        cont[self.ahu_act.ravel()] += scale * ahu_a[:, :3].ravel()
        cont[self.water_temp_idx] += scale * plant_a[0]; cont[self.water_pressure_idx] += scale * plant_a[1]
        disc = pid_disc.copy()
        votes = np.concatenate([ahu_a[:, 3], plant_a[2:3]])
        sw = list(range(self.K)) + [self.n_discrete - 1]
        disc[sw] = np.where(votes > 0.5, 1, np.where(votes < -0.5, 0, disc[sw]))
        return np.clip(cont, -1.0, 1.0).astype(np.float32), disc.astype(np.int32)


class Latch:
    """Switch votes as held decisions. A vote past +-0.5 on a free switch sets it on/off for `hold` steps; while held,
    that switch's vote is ignored and masked out of the policy gradient. hold=1 is the plain dead-zone vote of compose().
    Switch order: the K AHU run commands, then the plant pump (as compose)."""

    def __init__(self, n, hold):
        self.hold, self.left, self.val = int(hold), np.zeros(n, int), np.zeros(n, int)

    def apply(self, disc, sw, votes):
        """disc after compose with neutral votes; returns (disc, decision mask [n] float32 for this step)."""
        free = self.left == 0
        set_on, set_off = free & (votes > 0.5), free & (votes < -0.5)
        self.val[set_on], self.val[set_off] = 1, 0
        self.left[set_on | set_off] = self.hold
        held = self.left > 0
        d = disc.copy(); d[np.array(sw)[held]] = self.val[held]
        self.left[held] -= 1
        return d, free.astype(np.float32)
