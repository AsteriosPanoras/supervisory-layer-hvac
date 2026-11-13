"""Builds run_episode_k07_v9supe_gnn.py from run_episode_k07_v9supe_ax.py (sha 557d4aab..., the runner of the Residual PPO blind test).
It uses a fixed list of textual replacements, so the supervisor code stays byte-identical: ROS = learned optimal start, TR = trim and
respond of the duct static-pressure command.

The controller is the same Residual PPO controller, base rule included. Its residual network is replaced only when an environment
variable says so:
 - GNN_REC_DIR set (recording mode): the teacher's network runs unchanged, wrapped by gnn_net.Recorder. The episode is the Residual PPO
   episode bit for bit. Every network call's inputs and outputs go to <GNN_REC_DIR>/<tag>.rec.npz, as imitation data.
 - GNN_WEIGHTS set (GNN mode): gnn_net.GraphNets with those weights, loaded with weights_only=True, replaces the network.
   - From the teacher's checkpoint it keeps the base rule's parameters, the residual scale, the latch and the hold.
   - set_graph() gives it the building's zone -> AHU index from the controller's own BuildingIndex.
 - Neither set: the Residual PPO controller exactly, as run_episode_k07_v9supe_ax.py.
GNN_DIR = folder of gnn_net.py.  Keys: k07@ and k07sup@;ROS=a,TR=b only (unchanged).  usage: python3 make_runner_gnn.py"""
import hashlib, os
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
SRC, DST = f"{N}/run_episode_k07_v9supe_ax.py", f"{N}/run_episode_k07_v9supe_gnn.py"
s = open(SRC).read(); assert hashlib.sha256(s.encode()).hexdigest().startswith("557d4aabec5a8383")
HEAD = ("# v9supe_gnn (12 Nov 2026, built by gnnsup/make_runner_gnn.py from v9supe_ax): the Residual PPO controller whose residual network can be\n"
        "# recorded (GNN_REC_DIR, imitation data) or replaced by a graph network on the building's own structure (GNN_WEIGHTS).  The supervisor\n"
        "# code below is unchanged.  Keys: k07@ and k07sup@;ROS=a,TR=b.\n")
R = [
 ("AX_NET, AX_PARAMS, AX_SCALE, AX_LATCH = _load_safe(AXD / 'skynet.pt')\n",
  "AX_NET, AX_PARAMS, AX_SCALE, AX_LATCH = _load_safe(AXD / 'skynet.pt')\n"
  "GNN_MODE = 'rec' if os.environ.get('GNN_REC_DIR') else ('gnn' if os.environ.get('GNN_WEIGHTS') else None)\n"
  "if GNN_MODE:\n"
  "    sys.path.insert(0, str(Path(os.environ['GNN_DIR']).resolve())); import gnn_net as _gnn\n"
  "    if GNN_MODE == 'gnn':\n"
  "        _hold = AX_NET.hold; AX_NET = _gnn.load_safe(os.environ['GNN_WEIGHTS']); AX_NET.hold = _hold\n"
  "    else:\n"
  "        AX_NET = _gnn.Recorder(AX_NET)\n"),
 ("sky = SkynetPolicy(meta, spec, AX_NET, AX_PARAMS, AX_SCALE, PIDWaterResetPolicy, ev.validate_action, AX_LATCH)\n",
  "sky = SkynetPolicy(meta, spec, AX_NET, AX_PARAMS, AX_SCALE, PIDWaterResetPolicy, ev.validate_action, AX_LATCH)\n"
  "if GNN_MODE: AX_NET.set_graph(sky.ix.zone_ahu, sky.ix.K)\n"),
 ("policy_name = 'alex_residual_ppo' + ('+sup' if (ROS_ON or TR_ON) else '')\n",
  "policy_name = ('gnn_residual' if GNN_MODE == 'gnn' else 'alex_residual_ppo') + ('+sup' if (ROS_ON or TR_ON) else '')\n"),
 ("summary_path.write_text(json.dumps(summary, indent=2))\n",
  "summary[\"gnn\"] = {\"mode\": GNN_MODE, \"weights_sha256\": (sha(os.environ['GNN_WEIGHTS']) if GNN_MODE == 'gnn' else None),\n"
  "                  \"gnn_net_sha256\": (sha(Path(os.environ['GNN_DIR']) / 'gnn_net.py') if GNN_MODE else None), \"n_rec\": (len(AX_NET.z) if GNN_MODE == 'rec' else None)}\n"
  "if GNN_MODE == 'rec': AX_NET.save(Path(os.environ['GNN_REC_DIR']) / f\"{tag}.rec.npz\")\n"
  "summary_path.write_text(json.dumps(summary, indent=2))\n"),
]
for o, n in R:
    assert s.count(o) == 1, o[:70]; s = s.replace(o, n)
s = HEAD + s
open(DST, "w").write(s)
# the supervisor is untouched: every line of the source, except the replaced anchor lines that the replacement drops, appears in the
# output in the same order
dropped = {l for o, n in R for l in o.splitlines() if l not in n.splitlines()}
src_lines = [l for l in open(SRC).read().splitlines() if l not in dropped]; out_lines = s.splitlines(); j = 0
for line in src_lines:
    while out_lines[j] != line: j += 1
    j += 1
assert len(dropped) == 1, dropped        # only the policy_name line
print("written", DST, hashlib.sha256(s.encode()).hexdigest())
