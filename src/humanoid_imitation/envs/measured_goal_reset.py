"""Train-only measured standing resets; no state writes during execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

KEYS = ('q', 'dq', 'root_pos', 'root_rot', 'root_vel', 'root_ang_vel')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(path):
    spec = json.loads(Path(path).read_text())
    manifest_path = Path(spec['dataset_manifest'])
    if sha(manifest_path) != spec['dataset_manifest_sha256']:
        raise ValueError('Measured-reset dataset changed')
    manifest = json.loads(manifest_path.read_text())
    root = Path(spec['project_root']).resolve()
    entries = {e['id']: e for e in manifest['episodes']}
    history = spec.get('history_steps', 4)
    if not isinstance(history, int) or history < 4:
        raise ValueError('Invalid measured reset history length')
    rows = []
    for row in spec['states']:
        entry = entries[row['episode_id']]
        path = root / entry['state_file']
        if entry['split'] != 'train' or sha(path) != entry['state_sha256']:
            raise ValueError('Measured resets require unchanged train sources')
        with np.load(path, allow_pickle=False) as data:
            t = row['state_index']
            if not history-1 <= t < len(data['q']):
                raise ValueError('Measured reset lacks actual history')
            state = {k: data[k][t].copy() for k in KEYS}
            state.update({'history_'+k: data[k][t-history+1:t+1].copy() for k in KEYS})
        if any(not np.isfinite(v).all() for v in state.values()):
            raise ValueError('Nonfinite measured reset')
        if state['q'].shape != (28,) or abs(np.linalg.norm(state['root_rot'])-1) > .01:
            raise ValueError('Measured reset skeleton/rotation mismatch')
        if np.linalg.norm(state['root_vel'][:2]) > .15 or abs(state['root_ang_vel'][2]) > .2:
            raise ValueError('Measured standing reset is moving')
        xy = np.asarray(row['goal_offset_world_xy'], dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.linalg.norm(xy) > .24:
            raise ValueError('Measured goal offset outside standing reset range')
        state['goal_offset_world_xy'] = xy
        rows.append(state)
    if not rows:
        raise ValueError('Empty measured reset bank')
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


class MeasuredGoalReset:
    def __init__(self, config, device):
        self.probability = config['probability']
        if not math.isfinite(self.probability) or not 0 < self.probability <= 1:
            raise ValueError('Invalid measured reset probability')
        self.states = {k: torch.as_tensor(v, dtype=torch.float32, device=device)
                       for k, v in load_bank(config['bank']).items()}

    def sample(self, count):
        device = self.states['q'].device
        mask = torch.rand(count, device=device) < self.probability
        indices = torch.randint(len(self.states['q']), (int(mask.sum()),), device=device)
        return mask, {k: v[indices] for k, v in self.states.items()}


def rehearsal_yaw_offsets(count, device):
    """Equal occupied start-heading bins; 180 is one goal, not two labels."""
    bins = torch.tensor([0., 45., -45., 90., -90., 135., -135., 180.], device=device)
    values = bins[torch.randint(len(bins), (count,), device=device)]
    return values * math.pi / 180
