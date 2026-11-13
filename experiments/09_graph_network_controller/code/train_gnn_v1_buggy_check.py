"""Trains gnn_net.GraphNets by imitation of the Residual PPO network's residual means (the teacher), from the recordings that
run_episode_k07_v9supe_gnn.py writes in recording mode (<tag>.rec.npz).

The training buildings come only from the held-out set of Section 4 (k07x) and the official training buildings; the blind
architectures (k07x3) are never read.  The loss is masked mean squared error per entity type, summed over the three types:
 - zones: 3 outputs;
 - AHUs: 4 outputs;
 - plant: 3 outputs.
Validation is on whole episodes of buildings that are not in training. It reports R^2 per output and the loss.

usage: python -I train_gnn.py <out_weights.pt> <out_report.json> --train <rec dirs or files...> --val <rec files...> [--iters N] [--seed S]"""
import sys, json, glob, os, time, hashlib, argparse
import numpy as np
import torch

H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import gnn_net

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("report")
ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--val", nargs="+", required=True)
ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--d", type=int, default=64); ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); torch.set_num_threads(args.threads)


def files(items):
    out = []
    for it in items:
        out += sorted(glob.glob(os.path.join(it, "*.rec.npz"))) if os.path.isdir(it) else [it]
    return out


def load(paths):
    eps = []
    for f in paths:
        r = np.load(f)
        if int(r["n"]) == 0:
            continue
        assert "k07x3" not in f and "5220" not in os.path.basename(f) and "5230" not in os.path.basename(f), f"blind architecture in the data: {f}"
        eps.append(dict(f=f, z=r["z"], a=r["a"], p=r["p"], mz=r["mz"], ma=r["ma"], mp=r["mp"], zone_ahu=r["zone_ahu"].astype(np.int64), K=int(r["K"])))
    return eps


TR, VA = files(args.train), files(args.val)
assert not set(map(os.path.basename, TR)) & set(map(os.path.basename, VA)), "an episode is in both training and validation"
tr, va = load(TR), load(VA)
bld = lambda f: os.path.basename(f).split("_")[0] + "_" + os.path.basename(f).split("_")[1] if os.path.basename(f).startswith("building") else os.path.basename(f)
assert not {bld(e["f"]) for e in tr} & {bld(e["f"]) for e in va}, "a building is in both training and validation"


def batch(eps, idx):
    """idx: list of (episode, step) -> padded tensors."""
    Vm = max(eps[e]["z"].shape[1] for e, _ in idx); Km = max(eps[e]["K"] for e, _ in idx); B = len(idx)
    Fz, Fa, Fp = eps[0]["z"].shape[2], eps[0]["a"].shape[2], eps[0]["p"].shape[1]
    z = np.zeros((B, Vm, Fz), np.float32); a = np.zeros((B, Km, Fa), np.float32); p = np.zeros((B, Fp), np.float32)
    zm = np.zeros((B, Vm), np.float32); am = np.zeros((B, Km), np.float32); za = np.zeros((B, Vm), np.int64)
    tz = np.zeros((B, Vm, 3), np.float32); ta = np.zeros((B, Km, 4), np.float32); tp = np.zeros((B, 3), np.float32)
    for i, (e, t) in enumerate(idx):
        E = eps[e]; V, K = E["z"].shape[1], E["K"]
        z[i, :V] = E["z"][t]; a[i, :K] = E["a"][t]; p[i] = E["p"][t]; zm[i, :V] = 1; am[i, :K] = 1; za[i, :V] = E["zone_ahu"]
        tz[i, :V] = E["mz"][t]; ta[i, :K] = E["ma"][t]; tp[i] = E["mp"][t]
    T = lambda x: torch.from_numpy(x)
    return T(z), T(zm), T(a), T(am), T(p), T(za), T(tz), T(ta), T(tp)


def loss_fn(net, b):
    z, zm, a, am, p, za, tz, ta, tp = b
    mz, ma, mp = net.propagate(z, zm, a, am, p, za)
    lz = (((mz - tz) ** 2).sum(-1) * zm).sum() / (zm.sum() * 3); la = (((ma - ta) ** 2).sum(-1) * am).sum() / (am.sum() * 4)
    lp = ((mp - tp) ** 2).mean()
    return lz + la + lp, (lz.item(), la.item(), lp.item())


def evaluate(net, eps):
    """R^2 per output over every step of the given episodes, and the loss."""
    pred = {"z": [], "a": [], "p": []}; true = {"z": [], "a": [], "p": []}
    with torch.no_grad():
        for e in range(len(eps)):
            E = eps[e]
            for s0 in range(0, len(E["z"]), 256):
                idx = [(e, t) for t in range(s0, min(s0 + 256, len(E["z"])))]
                z, zm, a, am, p, za, tz, ta, tp = batch(eps, idx)
                mz, ma, mp = net.propagate(z, zm, a, am, p, za)
                pred["z"].append(mz[zm > 0].numpy()); true["z"].append(tz[zm > 0].numpy())
                pred["a"].append(ma[am > 0].numpy()); true["a"].append(ta[am > 0].numpy())
                pred["p"].append(mp.numpy()); true["p"].append(tp.numpy())
    out = {}
    for k in "zap":
        P, Y = np.concatenate(pred[k]), np.concatenate(true[k])
        out[k] = {"r2": [float(1 - ((P[:, j] - Y[:, j]) ** 2).sum() / max(((Y[:, j] - Y[:, j].mean()) ** 2).sum(), 1e-12)) for j in range(Y.shape[1])],
                  "mse": float(((P - Y) ** 2).mean()), "teacher_std": [float(Y[:, j].std()) for j in range(Y.shape[1])]}
    return out


net = gnn_net.GraphNets(d=args.d); opt = torch.optim.Adam(net.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr / 20)
pool = [(e, t) for e in range(len(tr)) for t in range(len(tr[e]["z"]))]
t0 = time.time(); hist = []
for it in range(args.iters):
    idx = [pool[i] for i in rng.integers(0, len(pool), args.batch)]
    l, parts = loss_fn(net, batch(tr, idx)); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    if it % 500 == 0 or it == args.iters - 1:
        hist.append({"iter": it, "loss": l.item(), "parts": parts, "s": round(time.time() - t0, 1)}); print(hist[-1], flush=True)
net.eval()
torch.save({"cfg": net.cfg, "state": net.state_dict()}, args.out)
rep = {"weights": args.out, "weights_sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(), "args": vars(args),
       "gnn_net_sha256": hashlib.sha256(open(os.path.join(H, "gnn_net.py"), "rb").read()).hexdigest(),
       "train_files": {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in TR},
       "val_files": {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in VA},
       "n_train_steps": len(pool), "n_train_episodes": len(tr), "n_val_episodes": len(va), "n_params": sum(p.numel() for p in net.parameters()),
       "history": hist, "train_fit": evaluate(net, tr), "val_fit": evaluate(net, va), "wall_s": round(time.time() - t0, 1)}
json.dump(rep, open(args.report, "w"), indent=1)
print("saved", args.out, rep["weights_sha256"][:16], "| val R2 zone", [round(x, 3) for x in rep["val_fit"]["z"]["r2"]], "ahu", [round(x, 3) for x in rep["val_fit"]["a"]["r2"]],
      "plant", [round(x, 3) for x in rep["val_fit"]["p"]["r2"]])
