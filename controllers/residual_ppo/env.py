"""A worker process that runs scenarios back to back, full 30 days each, and steps them with residual actions.

The PID, the hold and the feature gather live in the worker; the main process only sees features and sends raw
residuals. Protocol over a Pipe:
    ("reset", cursor)            -> features of the first step of scenario number `cursor` in this worker's list
    ("step", (zone_a, ahu_a, plant_a, scale)) -> (features, reward, done, info)
With paired=True the reward is the step's reward minus a zt shadow's on the same scenario (info["raw"] keeps the raw one).
    ("rollout", (state_dict, cfg, target, scale, hold, seed)) -> (samples, last value): the worker samples its own residuals
        with a copy of the net and steps until the step counter shared by all workers reaches `target`, so a fast
        building contributes more steps and no worker waits for the slowest one at every step
    ("close", None)
`done` marks the end of a scenario; the worker then starts the next scenario in its list by itself.
"""
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _worker(conn, scenarios, params, max_steps, latch, paired, counter):
    sys.path.insert(0, str(ROOT))
    import sbrl
    from skynet.base import make_floor
    from skynet.features import BuildingIndex, Latch
    E = sbrl._kit()
    if params.get("bill") == "honest":            # skynet_train / skynet_eval; tools/cma_zt.py and the gas audits bill as shipped
        sbrl.honest_bill()
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

    state = {}

    def start(cursor):
        for _ in range(len(scenarios)):          # a generated floor plan the simulator cannot build is skipped, loudly
            b, w = scenarios[cursor % len(scenarios)]
            try:
                _, env, meta, spec, _ = sbrl.make_base(b, w)
                break
            except Exception as e:                   # 2026-10-31: gen floor plan -> "Wasn't able to determine which corner the CV (6, 352) is"
                print(f"SKIP scenario {b}/{w}: simulator failed to build: {str(e).splitlines()[0][:160]}", flush=True)
                cursor += 1
        else:
            raise RuntimeError("no scenario in this worker's list builds")
        state.update(env=env, floor=make_floor(meta, spec, params, PIDWaterResetPolicy, E.validate_action),
                     ix=BuildingIndex(meta), cursor=cursor, t=0, name=f"{b}/{w}")
        state["latch"] = Latch(state["ix"].K + 1, latch)
        state["obs"], _ = env.reset()
        if paired:                                   # a zt shadow on the same scenario, stepped in lockstep
            _, senv, smeta, sspec, _ = sbrl.make_base(b, w)
            state.update(senv=senv, sfloor=make_floor(smeta, sspec, {"controller": "zt"}, PIDWaterResetPolicy, E.validate_action))
            state["sobs"], _ = senv.reset()
        return observe()

    def step(arg):
        zone_a, ahu_a, plant_a, scale = arg
        ix = state["ix"]
        votes = np.concatenate([ahu_a[:, 3], plant_a[2:3]])
        na, npl = ahu_a.copy(), plant_a.copy(); na[:, 3] = 0.0; npl[2] = 0.0      # votes go through the latch
        cont, disc = ix.compose(state["pc"], state["pd"], zone_a, na, npl, scale)
        free = getattr(state["floor"], "free", True)
        if free:
            disc, vmask = state["latch"].apply(disc, list(range(ix.K)) + [ix.n_discrete - 1], votes)
        else:                                         # zt_off base outside PI hours: the rule, no residual, latch untouched
            cont, disc, vmask = state["pc"].copy(), state["pd"].copy(), np.zeros(ix.K + 1, np.float32)
        overrides = int((disc != state["pd"]).sum())          # switches the votes changed, before the hold
        act = state["floor"].finish(cont, disc)
        if hasattr(state["floor"], "pol") and zone_a.shape[1] > 2:   # zone-target head; 2-column callers (a running cma_zt) skip it
            state["floor"].pol.target_offset = ix.target_offsets(zone_a, scale) if free else {}
        obs, r, te, tr, _ = state["env"].step(act)
        raw = float(r)
        if paired:                                   # reward relative to zt on the same weather and occupancy
            sf = state["sfloor"]; state["sobs"], sr, *_ = state["senv"].step(sf.finish(*sf.propose(state["sobs"])))
            r = raw - float(sr)
        state["obs"], state["t"] = obs, state["t"] + 1
        done = bool(te or tr or state["t"] >= max_steps)
        info = dict(scenario=state["name"], t=state["t"], overrides=overrides, cursor=state["cursor"], vmask=vmask, free=float(free), raw=raw)
        feats = start(state["cursor"] + 1) if done else observe()
        return feats, float(r), done, info

    def rollout(arg):
        import torch
        from skynet.nets import FlatNets, pad, pad_vmask
        sd, cfg, target, scale, hold, seed = arg
        if "net" not in state:
            torch.set_num_threads(1); state["net"] = FlatNets(**cfg); state["tick"], state["held"] = 0, None
        net = state["net"]; net.load_state_dict(sd); torch.manual_seed(seed)
        out = []
        while True:
            with counter.get_lock():
                if counter.value >= target:
                    break
                counter.value += 1
            f = state["feats"]; Z, zm, A, am, P = pad([f])
            with torch.no_grad():
                mu, v = net(Z, zm, A, am, P)
                acts_t = [m + ls.exp() * torch.randn_like(m) for m, ls in zip(mu, net.stds())]
            act = (acts_t[0][0].numpy(), acts_t[1][0].numpy(), acts_t[2][0].numpy())
            d = state["tick"] % hold == 0 or state["held"] is None; state["tick"] += 1     # as skynet_train's --hold
            if d: state["held"] = act
            else: act = state["held"]
            nf, r, done, info = step((act[0], act[1], act[2], scale))
            with torch.no_grad():                    # the latch mask is known only after the step
                lp, _ = net.log_prob_entropy(mu, acts_t, zm, am, pad_vmask([info["vmask"]], A.shape[1]))
            fr = info["free"] * float(d)                 # no residual acted, or a held one: no log-prob
            out.append((f, act, float(lp[0]) * fr, float(v[0]), r, done, info, fr))
            if not info["free"]: state["held"] = None    # off-hours: the next free step samples afresh
            if done: state["tick"], state["held"] = 0, None
            state["feats"] = nf
        with torch.no_grad():
            v_last = float(net(*pad([state["feats"]]))[1][0])
        return out, v_last

    def observe():
        pc, pd = state["floor"].propose(state["obs"])
        state["pc"], state["pd"] = pc, pd
        return state["ix"].features(state["obs"], pc, pd)

    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            state["feats"] = start(arg); conn.send(state["feats"])
        elif cmd == "rollout":
            conn.send(rollout(arg))
        elif cmd == "step":
            res = step(arg); state["feats"] = res[0]; conn.send(res)
        elif cmd == "close":
            conn.close(); return


class Workers:
    def __init__(self, scenario_lists, params, max_steps=8640, latch=1, paired=False):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        self.counter = ctx.Value("i", 0)            # steps taken in the current rollout, shared by every worker
        for scen in scenario_lists:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, scen, params, max_steps, latch, paired, self.counter), daemon=True); p.start()
            self.conns.append(a); self.procs.append(p)

    def reset(self, cursors):
        for c, cur in zip(self.conns, cursors): c.send(("reset", cur))
        return [c.recv() for c in self.conns]

    def step(self, actions, scale):
        for c, (z, a, p) in zip(self.conns, actions): c.send(("step", (z, a, p, scale)))
        return [c.recv() for c in self.conns]

    def rollout(self, state_dict, cfg, target, scale, hold, seeds):
        """Every worker samples and steps until `target` steps in total; returns [(samples, last value)] per worker."""
        self.counter.value = 0
        for c, s in zip(self.conns, seeds): c.send(("rollout", (state_dict, cfg, target, scale, hold, int(s))))
        return [c.recv() for c in self.conns]

    def close(self):
        for c in self.conns:
            try: c.send(("close", None))
            except Exception: pass
        for p in self.procs: p.join(timeout=5)
