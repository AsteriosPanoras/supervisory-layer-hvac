"""p287: fidelity of PREREG_GNNSUP_v1.
 F1: recording mode equals the plain Residual PPO runner bit for bit (continuous and discrete actions and rewards), smoke scene
     building_52000 minneapolis_2, 600 steps (work_smoke_ax vs work_smoke_rec).
 F2, F3: axsup/p275_ax_fidelity.py, unchanged, on the GN fidelity runs (GNN mode, the given weights; work_gnfid).
 ALL_PASS = F1 and p275's ALL_PASS, and every GN fidelity run used the given weights.
usage: python -I p287_gnn_fidelity.py <weights_sha256> <out.json>"""
import sys, os, glob, json, subprocess, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H); WSHA, OUT = sys.argv[1], sys.argv[2]


def one(pat):
    f = glob.glob(pat, recursive=True); assert len(f) == 1, (pat, f); s = json.load(open(f[0]))
    return s, np.load(f[0].replace(".summary.json", ".npz"), allow_pickle=True)


(sa, za), (sr, zr) = (one(f"{H}/laptop/work_smoke_{w}/out/**/building_52000_minneapolis_2_k07___600.summary.json") for w in ("ax", "rec"))
f1 = (sa["status"] == sr["status"] == "OK" and sa["steps_done"] == sr["steps_done"] == 600 and (sr.get("gnn") or {}).get("mode") == "rec"
      and (sa.get("gnn") or {}).get("mode") is None and all(np.array_equal(za[k], zr[k]) for k in ("actions_cont", "disc", "rewards")))
res = {"F1_recording_equals_teacher": bool(f1), "F1_n_recorded_calls": (sr.get("gnn") or {}).get("n_rec")}
outs = [d for d in glob.glob(f"{H}/laptop/work_gnfid/out/*") if os.path.isdir(d)]   # amendment 3: the runner also writes manifest_*.json here
assert len(outs) == 1, outs
wsh = {json.load(open(f)).get("gnn", {}).get("weights_sha256") for f in glob.glob(f"{outs[0]}/*.summary.json")}
res["gn_runs_use_the_weights"] = wsh == {WSHA}
p = subprocess.run([sys.executable, "-I", f"{N}/axsup/p275_ax_fidelity.py", outs[0], f"{H}/RESULT_GNNSUP_FIDELITY_p275.json"], capture_output=True, text=True)
r275 = json.load(open(f"{H}/RESULT_GNNSUP_FIDELITY_p275.json")); res["p275"] = r275
res["ALL_PASS"] = bool(f1 and res["gn_runs_use_the_weights"] and r275.get("ALL_PASS"))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "p275"}), "p275 ALL_PASS", r275.get("ALL_PASS"))
