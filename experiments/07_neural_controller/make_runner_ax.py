"""Builds run_episode_k07_v9supe_ax.py from run_episode_k07_v9supe_zt.py (sha 2b8144cb..., the runner of the ZT Best maintenance test)
by a fixed list of textual replacements, so that the supervisor code (ROS = learned optimal start, TR = trim & respond of the duct
static-pressure command) stays byte-identical.  Only the controller changes: the public Residual PPO controller of alexchilton
(Kaggle notebook 'Yes, RL Can Run a Building: Residual PPO', dataset smart-buildings-residual-rl, Apache-2.0): the zt rule of
malyshevdanil (Apache-2.0) with CMA-tuned constants and an off-hours rule, plus a residual PPO network with one shared network per
entity type (zone, air handler, plant).  The supervisor reads and writes the zt rule inside it (sky.floor.pol) exactly as it does
on ZT Best, and TR floors the FINAL command, after the network's residual.
Weights: loaded with torch.load(weights_only=True), the safe loader (the checkpoint's pickle was inspected statically: it imports
only collections.OrderedDict, torch.FloatStorage and torch._utils._rebuild_tensor_v2); the author's load() (weights_only=False) is
never called.  Keys: 'k07@' = the Residual PPO controller exactly; 'k07sup@;ROS=a,TR=b' = with the supervisor; nothing else.
Environment: AX_DIR = folder of the dataset files (skynet.pt and the python files, flat); AX_TORCH = folder exposing torch and its
pure-python dependencies only, appended at the END of sys.path (the simulator's own packages, e.g. numpy, keep precedence).
usage: python3 make_runner_ax.py   (writes ../run_episode_k07_v9supe_ax.py and prints its sha256)"""
import hashlib, os
H = os.path.dirname(os.path.abspath(__file__)); N = os.path.dirname(H)
SRC, DST = f"{N}/run_episode_k07_v9supe_zt.py", f"{N}/run_episode_k07_v9supe_ax.py"
s = open(SRC).read(); assert hashlib.sha256(s.encode()).hexdigest().startswith("2b8144cb1d959372")
HEAD = ("# v9supe_ax (11 Nov 2026, built by axsup/make_runner_ax.py from v9supe_zt): the controller is the public Residual PPO controller of\n"
        "# alexchilton (Apache-2.0; zt rule of malyshevdanil + CMA constants + off-hours rule + residual PPO network, one shared net per entity\n"
        "# type).  The supervisor code below is unchanged; it reads and writes the zt rule inside the controller (sky.floor.pol), and TR floors\n"
        "# the final command after the residual.  Weights loaded with the safe loader (weights_only=True).  Keys: k07@ and k07sup@;ROS=a,TR=b.\n")
R = [
 # 1. the controller package, torch (appended at the end of sys.path) and the safe weight loader, right after the kit imports
 ("from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker\n",
  "from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker\n"
  "from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy\n"
  "import shutil as _sh, tempfile as _tf\n"
  "AXD = Path(os.environ['AX_DIR']).resolve()\n"
  "if os.environ.get('AX_TORCH'): sys.path.append(os.environ['AX_TORCH'])\n"
  "_pkg = Path(_tf.mkdtemp(prefix='axpkg_')); (_pkg / 'skynet').mkdir()\n"
  "for _f in ('__init__.py', 'base.py', 'features.py', 'nets.py', 'policy.py', 'zt_controller.py'): _sh.copy(AXD / _f, _pkg / 'skynet' / _f)\n"
  "sys.path.insert(0, str(_pkg))\n"
  "import torch; torch.set_num_threads(1)\n"
  "from skynet.policy import SkynetPolicy\n"
  "from skynet.nets import FlatNets\n"
  "from skynet import zt_controller as their_k07_policy      # trace_point reads _outside_kelvin from here\n"
  "def _load_safe(p):\n"
  "    ck = torch.load(p, map_location='cpu', weights_only=True)\n"
  "    net = FlatNets.from_checkpoint(ck); net.hold = ck['args'].get('hold', 1)\n"
  "    return net, ck['params'], ck['args']['scale'], ck['args'].get('latch', 1)\n"
  "AX_NET, AX_PARAMS, AX_SCALE, AX_LATCH = _load_safe(AXD / 'skynet.pt')\n"),
 # 2. keys: only k07@ and k07sup@ without overrides, no unoccupied setup; no K07/ZT class is built
 ("if name not in (\"k07\", \"k07sw\", \"k07rule\", \"k07unif\", \"k07ros\", \"k07tr\", \"k07sup\"):\n    raise SystemExit(f\"unknown policy key {key!r}\")\n",
  "if name not in (\"k07\", \"k07sup\") or UNS_COOL_K != 0.0:\n    raise SystemExit(f\"v9supe_ax runs only k07@ and k07sup@;ROS=a,TR=b: {key!r}\")\n"),
 ("import their_k07_policy\nconsts = json.load(open(Path(their_dir) / \"k07_submission_constants.json\"))\nif name in (",
  "assert not overrides, f\"v9supe_ax takes no controller overrides: {overrides}\"\nconsts = {}      # the class built below from their_k07_policy (= the controller's zt module) is never used\nif name in ("),
 # 3. the controller
 ("policy = ev.initialize_policy(cls, meta, spec)\n",
  "sky = SkynetPolicy(meta, spec, AX_NET, AX_PARAMS, AX_SCALE, PIDWaterResetPolicy, ev.validate_action, AX_LATCH)\n"
  "policy = sky.floor.pol      # the zt rule inside the controller: the supervisor reads and writes it as on ZT Best\n"),
 ("policy_name = ev.get_policy_name(policy, key)\n", "policy_name = 'alex_residual_ppo' + ('+sup' if (ROS_ON or TR_ON) else '')\n"),
 ("        raw = policy.get_action(obs)\n", "        raw = sky.get_action(obs)\n"),
 # 4. provenance
 ("\"code_sha256\": {\"run_episode_k07_v3_log.py\": sha(here), \"their_k07_policy.py\": sha(Path(their_dir) / \"their_k07_policy.py\"),\n                    \"k07_submission_constants.json\": sha(Path(their_dir) / \"k07_submission_constants.json\")},",
  "\"code_sha256\": {\"run_episode_k07_v9supe_ax.py\": sha(here), **{f: sha(AXD / f) for f in ('skynet.pt', 'policy.py', 'nets.py', 'features.py', 'base.py', 'zt_controller.py')}},"),
 ("\"note\": \"simulator result, one scenario; K07 run for the SB26 paper; not a leaderboard score\",",
  "\"note\": \"simulator result, one scenario; PUBLIC Residual PPO controller (alexchilton, Apache-2.0) with or without the unchanged supervisor; not a leaderboard score\","),
]
for o, n in R:
    assert s.count(o) == 1, o[:70]; s = s.replace(o, n)
s = HEAD + s
open(DST, "w").write(s)
print("written", DST, hashlib.sha256(s.encode()).hexdigest())
