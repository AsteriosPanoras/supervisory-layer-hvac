"""M1: flat shared fast nets. One zone net (a copy per VAV), one AHU net (a copy per AHU), one plant net.

Each net reads only its own entity's features (skynet/features.py) and outputs a residual; the last layer is zero,
so an untrained policy is the base controller exactly. The critic is separate and sees the mean-pooled embeddings
of all entities. Batches are padded to the largest building in the batch; masks keep padded entities out of every
sum, mean and loss.
"""
import numpy as np
import torch
import torch.nn as nn

from skynet.features import N_ZONE, N_AHU, N_PLANT, ZONE_OUT, AHU_OUT, PLANT_OUT


def mlp(n_in, d, depth):
    layers, n = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(n, d), nn.Tanh()]; n = d
    return nn.Sequential(*layers)


def masked_mean(x, mask):                      # x [B, N, d], mask [B, N]
    m = mask.unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class FlatNets(nn.Module):
    def __init__(self, d=64, depth=2, log_std=-1.0, zone_out=ZONE_OUT, n_in=(N_ZONE, N_AHU, N_PLANT)):
        super().__init__()
        n_in = tuple(int(n) for n in n_in)               # input columns read per entity: an older checkpoint reads fewer
        self.cfg = dict(d=d, depth=depth, zone_out=zone_out, n_in=n_in); self.n_in = n_in
        N_ZONE, N_AHU, N_PLANT = n_in
        self.zone, self.ahu, self.plant = mlp(N_ZONE, d, depth), mlp(N_AHU, d, depth), mlp(N_PLANT, d, depth)
        self.zone_mu, self.ahu_mu, self.plant_mu = nn.Linear(d, zone_out), nn.Linear(d, AHU_OUT), nn.Linear(d, PLANT_OUT)
        for head in (self.zone_mu, self.ahu_mu, self.plant_mu):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        self.zone_log_std = nn.Parameter(torch.full((zone_out,), log_std))
        self.ahu_log_std = nn.Parameter(torch.full((AHU_OUT,), log_std))
        self.plant_log_std = nn.Parameter(torch.full((PLANT_OUT,), log_std))
        self.c_zone, self.c_ahu, self.c_plant = mlp(N_ZONE, d, depth), mlp(N_AHU, d, depth), mlp(N_PLANT, d, depth)
        self.c_head = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 1))

    @classmethod
    def from_checkpoint(cls, ck):
        """The net of a skynet_train.py checkpoint, eval mode; checkpoints from before the zone-target head have 2 zone outputs."""
        n_in = tuple(ck["model"][f"{e}.0.weight"].shape[1] for e in ("zone", "ahu", "plant"))
        net = cls(**{"zone_out": ck["model"]["zone_mu.weight"].shape[0], "n_in": n_in, **ck["cfg"]}); net.load_state_dict(ck["model"]); net.eval()
        return net

    def forward(self, zone, zmask, ahu, amask, plant):
        zone, ahu, plant = (x[..., :n].contiguous() for x, n in zip((zone, ahu, plant), self.n_in))   # contiguous: same float path as before
        mu = (self.zone_mu(self.zone(zone)), self.ahu_mu(self.ahu(ahu)), self.plant_mu(self.plant(plant)))
        h = torch.cat([masked_mean(self.c_zone(zone), zmask), masked_mean(self.c_ahu(ahu), amask), self.c_plant(plant)], -1)
        return mu, self.c_head(h).squeeze(-1)

    def stds(self):
        return self.zone_log_std, self.ahu_log_std, self.plant_log_std

    def log_prob_entropy(self, mu, acts, zmask, amask, vmask=None):
        """Sum of Gaussian log-probabilities over every real entity's outputs; padded entities contribute nothing.
        vmask [B, K+1] (AHU run votes, then the pump vote): 0 where the switch is held by the latch, so that vote had
        no effect and is left out of the log-probability and the entropy."""
        lp, ent = 0.0, 0.0
        for j, (m, a, ls, mask) in enumerate(zip(mu, acts, self.stds(), (zmask, amask, None))):
            dist = torch.distributions.Normal(m, ls.exp().expand_as(m))
            l, e = dist.log_prob(a), dist.entropy()
            if vmask is not None and j == 1:
                w = torch.ones_like(l); w[..., 3] = vmask[:, : l.shape[1]]; l, e = l * w, e * w
            if vmask is not None and j == 2:
                w = torch.ones_like(l); w[..., 2] = vmask[:, -1]; l, e = l * w, e * w
            if mask is None:
                lp, ent = lp + l.sum(-1), ent + e.sum(-1)
            else:
                lp, ent = lp + (l.sum(-1) * mask).sum(-1), ent + (e.sum(-1) * mask).sum(-1)
        return lp, ent


def pad(batch):
    """List of (zone [V,F], ahu [K,F], plant [F]) -> padded tensors and masks."""
    B = len(batch); V = max(z.shape[0] for z, _, _ in batch); K = max(a.shape[0] for _, a, _ in batch)
    Z = np.zeros((B, V, batch[0][0].shape[1]), np.float32); zm = np.zeros((B, V), np.float32)
    A = np.zeros((B, K, batch[0][1].shape[1]), np.float32); am = np.zeros((B, K), np.float32)
    P = np.stack([p for _, _, p in batch])
    for i, (z, a, _) in enumerate(batch):
        Z[i, : len(z)] = z; zm[i, : len(z)] = 1; A[i, : len(a)] = a; am[i, : len(a)] = 1
    t = torch.from_numpy
    return t(Z), t(zm), t(A), t(am), t(P)


def pad_acts(acts, V, K):
    B = len(acts); za = np.zeros((B, V, acts[0][0].shape[1]), np.float32); aa = np.zeros((B, K, acts[0][1].shape[1]), np.float32)
    for i, (z, a, _) in enumerate(acts):
        za[i, : len(z)] = z; aa[i, : len(a)] = a
    return torch.from_numpy(za), torch.from_numpy(aa), torch.from_numpy(np.stack([p for _, _, p in acts]))


def pad_vmask(vms, K):
    """Per-sample latch masks [K_i + 1] -> [B, K + 1]: AHU votes padded with 1 (padded AHUs are masked anyway), pump last."""
    V = np.ones((len(vms), K + 1), np.float32)
    for i, m in enumerate(vms):
        V[i, : len(m) - 1] = m[:-1]; V[i, -1] = m[-1]
    return torch.from_numpy(V)
