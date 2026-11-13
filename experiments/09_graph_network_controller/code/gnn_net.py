"""A small graph network on the building's own structure, used as the residual network of the public Residual PPO controller in place of
its per-entity networks.

Graph: the building as the simulator describes it.  Nodes are the zones (VAVs), the air handlers (AHUs) and the hot-water plant.  Edges
are zone-AHU (which AHU serves each zone, from the metadata's ahu_zone_assignments) and AHU-plant.  Two rounds of message passing:
 - a zone reads its AHU;
 - an AHU reads the mean of its zones and the plant;
 - the plant reads the mean of the AHUs.

Inputs per node: the raw fields of that device, the base rule's command for it, and the observation's clock and occupancy fields. These
are columns of the controller's feature arrays, only rescaled by fixed physical constants. The derived columns are NOT used: band
errors, AHU context in zones, pooled zone and AHU summaries, and episode memory.

Outputs: the same residual means as the teacher's network (zone 3, AHU 4, plant 3).  The residual composition, the hold, the latch
and the base rule are the controller's own, unchanged.

Training (train_gnn.py) is by imitation of the teacher's residual means on training buildings. The blind architectures are never used.
No third-party code: torch only.
"""
import numpy as np
import torch
import torch.nn as nn

ZONE_COLS = list(range(0, 6)) + [8, 9] + list(range(12, 21))                      # raw 6, base command 2, global 9  -> 17
AHU_COLS = list(range(0, 15)) + list(range(21, 25)) + list(range(25, 34))          # raw 15, base command 4, global 9 -> 28
PLANT_COLS = list(range(0, 7)) + list(range(10, 13)) + list(range(13, 22))         # raw 7, base command 3, global 9  -> 19
ZONE_OUT, AHU_OUT, PLANT_OUT = 3, 4, 3


def _mlp(n_in, d):
    return nn.Sequential(nn.Linear(n_in, d), nn.Tanh(), nn.Linear(d, d), nn.Tanh())


class GraphNets(nn.Module):
    def __init__(self, d=64, rounds=2):
        super().__init__()
        self.cfg = dict(d=d, rounds=rounds)
        self.enc_z, self.enc_a, self.enc_p = _mlp(len(ZONE_COLS), d), _mlp(len(AHU_COLS), d), _mlp(len(PLANT_COLS), d)
        self.upd_z = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.Tanh()) for _ in range(rounds)])
        self.upd_a = nn.ModuleList([nn.Sequential(nn.Linear(3 * d, d), nn.Tanh()) for _ in range(rounds)])
        self.upd_p = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.Tanh()) for _ in range(rounds)])
        self.zone_mu, self.ahu_mu, self.plant_mu = nn.Linear(d, ZONE_OUT), nn.Linear(d, AHU_OUT), nn.Linear(d, PLANT_OUT)
        self.register_buffer("zc", torch.tensor(ZONE_COLS)); self.register_buffer("ac", torch.tensor(AHU_COLS))
        self.register_buffer("pc", torch.tensor(PLANT_COLS))
        self.graph = None; self.hold = 1

    def set_graph(self, zone_ahu, K):
        """The building's zone -> AHU index (BuildingIndex.zone_ahu) and the number of AHUs; once per episode."""
        self.graph = (torch.as_tensor(np.asarray(zone_ahu), dtype=torch.long)[None], int(K))

    def propagate(self, zone, zmask, ahu, amask, plant, zone_ahu):
        """zone [B,V,F], zmask [B,V], ahu [B,K,F], amask [B,K], plant [B,F], zone_ahu [B,V] (padded zones: any valid index, masked)."""
        hz = self.enc_z(zone.index_select(-1, self.zc)); ha = self.enc_a(ahu.index_select(-1, self.ac)); hp = self.enc_p(plant.index_select(-1, self.pc))
        K = ahu.shape[1]
        member = torch.nn.functional.one_hot(zone_ahu, K).float() * zmask.unsqueeze(-1)            # [B,V,K]
        cnt = member.sum(1).clamp(min=1.0)                                                         # [B,K]
        acnt = amask.sum(1, keepdim=True).clamp(min=1.0)                                           # [B,1]
        for r in range(len(self.upd_z)):
            parent = torch.gather(ha, 1, zone_ahu.unsqueeze(-1).expand(-1, -1, ha.shape[-1]))      # [B,V,d]
            zmean = torch.einsum("bvk,bvd->bkd", member, hz) / cnt.unsqueeze(-1)                     # [B,K,d]
            amean = (ha * amask.unsqueeze(-1)).sum(1) / acnt                                         # [B,d]
            hz_n = self.upd_z[r](torch.cat([hz, parent], -1))
            ha_n = self.upd_a[r](torch.cat([ha, zmean, hp.unsqueeze(1).expand(-1, K, -1)], -1))
            hp_n = self.upd_p[r](torch.cat([hp, amean], -1))
            hz, ha, hp = hz_n, ha_n, hp_n
        return self.zone_mu(hz), self.ahu_mu(ha), self.plant_mu(hp)

    def forward(self, zone, zmask, ahu, amask, plant):
        """The teacher network's call signature (SkynetPolicy.act): one building, batch 1, graph set by set_graph()."""
        assert self.graph is not None and zone.shape[0] == 1 and ahu.shape[1] == self.graph[1], "set_graph() first, batch 1"
        mz, ma, mp = self.propagate(zone, zmask, ahu, amask, plant, self.graph[0])
        return (mz, ma, mp), torch.zeros(zone.shape[0])


def load_safe(path):
    """A file written by train_gnn.py: {'cfg': ..., 'state': state_dict}; tensors and plain types only, so weights_only=True loads it."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = GraphNets(**ck["cfg"]); net.load_state_dict(ck["state"]); net.eval()
    return net


class Recorder:
    """Wraps the teacher's network: returns exactly its output and keeps, for every call, the inputs it was given and the means it
    returned (the imitation data).  Nothing it does reaches the controller."""

    def __init__(self, net):
        self.net, self.hold = net, getattr(net, "hold", 1)
        self.z, self.a, self.p, self.mz, self.ma, self.mp = [], [], [], [], [], []
        self.zone_ahu, self.K = None, None

    def set_graph(self, zone_ahu, K):
        self.zone_ahu, self.K = np.asarray(zone_ahu), int(K)

    def __call__(self, zone, zmask, ahu, amask, plant):
        out = self.net(zone, zmask, ahu, amask, plant)
        (mz, ma, mp), _ = out
        self.z.append(zone[0].numpy().astype(np.float32)); self.a.append(ahu[0].numpy().astype(np.float32)); self.p.append(plant[0].numpy().astype(np.float32))
        self.mz.append(mz[0].numpy().astype(np.float32)); self.ma.append(ma[0].numpy().astype(np.float32)); self.mp.append(mp[0].numpy().astype(np.float32))
        return out

    def save(self, path):
        if not self.z:
            np.savez_compressed(path, n=0); return
        np.savez_compressed(path, n=len(self.z), z=np.stack(self.z), a=np.stack(self.a), p=np.stack(self.p), mz=np.stack(self.mz),
                            ma=np.stack(self.ma), mp=np.stack(self.mp), zone_ahu=self.zone_ahu, K=self.K)
