"""The skynet policy behind the competition interface: the base floor, the per-entity features, the shared nets'
mean actions, the residual composition and the hold. Used by the Kaggle kernel and by the local equality test, so
the board scores exactly what skynet_eval.py scored.
"""
import numpy as np
import torch

from skynet.base import make_floor
from skynet.features import BuildingIndex, Latch
from skynet.nets import FlatNets


def load(path):
    """A checkpoint written by skynet_train.py -> (net, base params, residual scale)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = FlatNets.from_checkpoint(ck)
    net.hold = ck["args"].get("hold", 1)                # steps a residual is held (skynet_train.py --hold)
    return net, ck["params"], ck["args"]["scale"], ck["args"].get("latch", 1)


class SkynetPolicy:
    def __init__(self, meta, spec, net, params, scale, PIDWaterResetPolicy, validate_action, latch=1):
        self.floor = make_floor(meta, spec, params, PIDWaterResetPolicy, validate_action)
        self.ix, self.net, self.scale = BuildingIndex(meta), net, scale
        self.latch = Latch(self.ix.K + 1, latch)
        self.hold, self.t, self.held = int(getattr(net, "hold", 1)), 0, None     # a new residual every `hold` steps

    def get_action(self, obs):
        pc, pd = self.floor.propose(obs)
        self.feats = self.ix.features(obs, pc, pd)       # every step, as skynet/env.py: it also updates the episode memory
        decide = self.t % self.hold == 0; self.t += 1     # counted on every step, as in training (skynet/env.py)
        if not getattr(self.floor, "free", True):     # zt_off base outside PI hours: the rule, no residual
            self.held = None
            if hasattr(self.floor, "pol"): self.floor.pol.target_offset = {}
            return self.floor.finish(pc, pd)
        return self.act(obs, pc, pd, decide)

    @torch.no_grad()
    def act(self, obs, pc, pd, decide=True):
        if decide or self.held is None:
            z, a, p = self.feats
            t = torch.from_numpy
            (mz, ma, mp), _ = self.net(t(z)[None], torch.ones(1, len(z)), t(a)[None], torch.ones(1, len(a)), t(p)[None])
            self.held = (mz[0].numpy(), ma[0].numpy(), mp[0].numpy())
        mz, ma, mp = self.held
        votes = np.concatenate([ma[:, 3], mp[2:3]])
        na, npl = ma.copy(), mp.copy(); na[:, 3] = 0.0; npl[2] = 0.0
        cont, disc = self.ix.compose(pc, pd, mz, na, npl, self.scale)
        disc, _ = self.latch.apply(disc, list(range(self.ix.K)) + [self.ix.n_discrete - 1], votes)
        if hasattr(self.floor, "pol") and mz.shape[1] > 2:       # the PI's zone targets for the next step, as skynet/env.py
            self.floor.pol.target_offset = self.ix.target_offsets(mz, self.scale)
        return self.floor.finish(cont, disc)


class MergedPolicy(SkynetPolicy):
    """The two levers merged by time of day (docs/RL_DIAGNOSIS_2026-10-27.md): the skynet residual on the zt base in
    zt's PI hours, and outside them every continuous command -1, recirculation 1 and every switch off."""

    def __init__(self, meta, *args, **kw):
        super().__init__(meta, *args, **kw)
        self.off_c = np.full(meta["n_continuous_action"], -1.0, np.float32)
        self.off_c[list(meta["recirculation_indices"])] = 1.0
        self.off_d = np.zeros(meta["n_discrete_action"], np.int32)

    def get_action(self, obs):
        pc, pd = self.floor.propose(obs)             # zt runs every step so its own state stays current
        self.feats = self.ix.features(obs, pc, pd)
        if not self.floor.pol._pi_active:
            return self.floor.finish(self.off_c.copy(), self.off_d.copy())
        return self.act(obs, pc, pd)
