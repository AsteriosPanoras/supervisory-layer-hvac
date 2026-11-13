import os, sys
from pathlib import Path
candidates = [Path('/kaggle/input/competitions/neurips-2026-smart-buildings'), Path('/kaggle/input/neurips-2026-smart-buildings')]
if os.getenv('SMART_BUILDINGS_DATA_ROOT'):
    candidates.insert(0, Path(os.environ['SMART_BUILDINGS_DATA_ROOT']))
DATA_ROOT = next((p for p in candidates if (p / 'smart_buildings').exists()), None)
if DATA_ROOT is None:
    raise FileNotFoundError('Attach competition data or set SMART_BUILDINGS_DATA_ROOT')
sys.path.insert(0, str(DATA_ROOT))
sys.path.insert(0, str(DATA_ROOT / 'kaggle_evaluation'))
IS_RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
print('Competition data located; hidden rerun:', IS_RERUN)

import os, hashlib, shutil, sys
from pathlib import Path
import torch
W = Path(os.environ['AX_DIR']) if os.getenv('AX_DIR') else next(Path(p) for p, d, f in os.walk('/kaggle/input') if 'skynet.pt' in f)
assert hashlib.sha256((W / 'skynet.pt').read_bytes()).hexdigest() == '8b1b4d0a4ff045760be670fbae2d559e02e63494439029bc3feb33af9bbb3e66', 'not the checkpoint of the study'
os.makedirs('/tmp/axpkg/skynet', exist_ok=True)                              # the dataset stores the package flat (README)
for f in ('__init__.py', 'base.py', 'features.py', 'nets.py', 'policy.py', 'zt_controller.py'):
    shutil.copy(W / f, f'/tmp/axpkg/skynet/{f}')
sys.path.insert(0, '/tmp/axpkg')
from skynet.nets import FlatNets
ck = torch.load(W / 'skynet.pt', map_location='cpu', weights_only=True)     # safe loader: tensors and plain containers only
NET = FlatNets.from_checkpoint(ck); NET.hold = ck['args'].get('hold', 1)
PARAMS, SCALE, LATCH = ck['params'], ck['args']['scale'], ck['args'].get('latch', 1)
print('Residual PPO checkpoint', ck['args'].get('run'), ck.get('steps'), 'scale', SCALE)

torch.set_num_threads(1)
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.evaluate import validate_action
from kaggle_evaluation import smart_buildings_inference_server
from skynet.policy import SkynetPolicy


class SubmissionPolicy(smart_buildings_inference_server.Policy):
    """The author's submission class (notebook 'Yes, RL Can Run a Building: Residual PPO', last cell), unchanged."""
    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self.p = SkynetPolicy(building_meta_data, action_spec, NET, PARAMS, SCALE, PIDWaterResetPolicy, validate_action, LATCH)

    def get_action(self, obs):
        return self.p.get_action(obs)


smart_buildings_inference_server.SmartBuildingsInferenceServer(SubmissionPolicy).run(competition_data_folder=str(DATA_ROOT))
