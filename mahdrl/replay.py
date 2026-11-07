"""Immutable replay packing, VA observations, and behavior preview."""
from types import MappingProxyType
import numpy as np
import torch

def frozen_array(value, dtype=np.float32):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite replay array after dtype conversion')
    result.flags.writeable = False
    return result

def frozen_observation(obs):
    return MappingProxyType({k: frozen_array(obs[k]) for k in ('context', 'candidates', 'exists_mask')})

def pack(observations, masks):
    if not observations or len(observations) != len(masks):
        raise ValueError('Empty or mismatched replay batch')
    lengths = np.fromiter((len(x) for x in masks), dtype=np.int64, count=len(masks))
    context = np.stack([x['context'] for x in observations]).astype(np.float32, copy=False)
    n, b = (max(1, int(lengths.max())), len(masks))
    f = observations[0]['candidates'].shape[1]
    if np.all(lengths == n):
        features = np.stack([x['candidates'] for x in observations]).astype(np.float32, copy=False)
        exists = np.stack([x['exists_mask'] for x in observations]).astype(np.float32, copy=False)
        legal = np.stack(masks).astype(np.float32, copy=False)
    else:
        features = np.zeros((b, n, f), np.float32)
        exists = np.zeros((b, n), np.float32)
        legal = np.zeros_like(exists)
        for k in np.unique(lengths):
            if not k:
                continue
            indices = np.flatnonzero(lengths == k)
            features[indices, :k] = np.stack([observations[i]['candidates'] for i in indices])
            exists[indices, :k] = np.stack([observations[i]['exists_mask'] for i in indices])
            legal[indices, :k] = np.stack([masks[i] for i in indices])
    if context.ndim != 2 or features.shape != (b, n, f) or exists.shape != (b, n) or (legal.shape != (b, n)):
        raise ValueError('Invalid packed observation shape')
    if not all((np.isfinite(x).all() for x in (context, features, exists, legal))):
        raise ValueError('Non-finite packed observation')
    if np.any((exists != 0) & (exists != 1)) or np.any((legal != 0) & (legal != 1)) or np.any(legal > exists):
        raise ValueError('Invalid packed masks')
    return (context, features, exists, legal)

def to_device(arrays, device):
    return tuple((torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays))

def collate(observations, masks, device):
    return to_device(pack(observations, masks), device)
from mahdrl.config import APF
import numpy as np

def behavior_probabilities(q, mask, times, epsilon):
    q, mask, times = (np.asarray(x) for x in (q, mask, times))
    if q.ndim != 1 or mask.shape != q.shape or times.shape != q.shape:
        raise ValueError('Invalid preview action arrays')
    legal = np.flatnonzero(mask > 0.5)
    if not len(legal) or not np.isin(mask, (0, 1)).all() or (not np.isfinite(q).all()) or (not np.isfinite(times[legal]).all()) or np.any(times[legal] < 0) or (not np.isfinite(epsilon)) or (not 0 <= epsilon <= 1):
        raise ValueError('Invalid preview probabilities or legal candidates')
    greedy = legal[np.argmax(q[legal])]
    fastest = legal[np.isclose(times[legal], times[legal].min(), rtol=1e-09, atol=1e-09)]
    p = np.zeros(len(q), np.float64)
    p[legal] = epsilon * (1 - APF.guided_probability) / len(legal)
    p[fastest] += epsilon * APF.guided_probability / len(fastest)
    p[greedy] += 1 - epsilon
    return p
from mahdrl.config import VM_CAPACITY
from mahdrl.config import VM_CONTEXT_DIM
from mahdrl.config import VM_FEATURE_DIM
from mahdrl.config import VM_INPUT_DIM
from mahdrl.networks import validate_observation
import numpy as np

def fixed_state(state):
    obs, mask = (state['obs'], np.asarray(state['mask'], dtype=np.float32))
    validate_observation(obs, mask, decision=False)
    k = len(mask)
    if obs['context'].shape != (VM_CONTEXT_DIM,) or obs['candidates'].shape != (k, VM_FEATURE_DIM) or k > VM_CAPACITY:
        raise ValueError('Fixed VA requires context30, VM7 and at most 9 local VMs')
    candidates = np.zeros((VM_CAPACITY, VM_FEATURE_DIM), np.float32)
    candidates[:k] = obs['candidates']
    legal = np.zeros(VM_CAPACITY, np.float32)
    legal[:k] = mask
    result = dict(state, obs=np.concatenate((obs['context'], candidates.ravel())).astype(np.float32), mask=legal)
    if 'performance_times' in state:
        times = np.zeros(VM_CAPACITY, np.float64)
        times[:k] = state['performance_times']
        result['performance_times'] = times
    if 'rows' in state:
        rows = np.zeros((VM_CAPACITY, 3), np.float64)
        rows[:k] = state['rows']
        result['rows'] = rows
    result['candidate_ids'] = list(state.get('candidate_ids', [])) + [None] * (VM_CAPACITY - k)
    return result

def validate_fixed(obs, mask, *, decision=True):
    obs, mask = (np.asarray(obs), np.asarray(mask))
    if obs.shape != (VM_INPUT_DIM,) or mask.shape != (VM_CAPACITY,):
        raise ValueError('Fixed VA requires a 93-vector and a 9-action mask')
    if not np.isfinite(obs).all() or not np.isfinite(mask).all() or (not np.isin(mask, (0, 1)).all()):
        raise ValueError('Invalid fixed VA state or binary mask')
    if decision and (not np.any(mask > 0.5)):
        raise ValueError('No legal VM at a non-terminal decision')
import numpy as np
import torch

def prepare_decision(env, va, host, *, training):
    states = env.preview_vm_states()
    if set(states) != {env.host_ids[i] for i in np.flatnonzero(host['mask'] > 0.5)}:
        raise RuntimeError('Host legality and VM previews disagree')
    version, epsilon = (va._updates, va.epsilon() if training else 0.0)
    with env.profiler.stage('va_preview'):
        obs = torch.as_tensor(np.stack([s['obs'] for s in states.values()]), dtype=torch.float32, device=va.device)
        masks = torch.as_tensor(np.stack([s['mask'] for s in states.values()]), dtype=torch.float32, device=va.device)
        with torch.no_grad():
            values = va.online(obs, masks).detach().cpu().numpy()
    ref = float(env._obs_ref('T_ref', env.task_baseline_norm))
    if not np.isfinite(ref) or ref <= 0 or (not 0 < va.gamma < 1):
        raise ValueError('Preview requires positive source time scale and gamma in (0,1)')
    reward_ref = (1 - env.alpha_delay_vm) * env.energy_reward_scale * 100 * ref + env.alpha_delay_vm
    extra = np.zeros((env.num_hosts, 4), np.float32)
    for i, (hid, state) in enumerate(states.items()):
        q = values[i].copy()
        p = behavior_probabilities(q, state['mask'], state['performance_times'], epsilon)
        rows = state['rows']
        extra[env.host_ids.index(hid)] = [p @ rows[:, 0] / ref, p @ rows[:, 1] / (100 * ref), p @ np.clip(rows[:, 2] / env.task_baseline_norm, 0, 1), (1 - va.gamma) * (p @ q) / reward_ref]
        state.update(obs=frozen_array(state['obs']), mask=frozen_array(state['mask']), q=q, probabilities=p, va_version=version, epsilon=epsilon)
    if not np.isfinite(extra).all():
        raise ValueError('Non-finite VA-aware host features')
    obs = dict(host['obs'])
    obs['candidates'] = np.concatenate((obs['candidates'], extra), axis=1)
    return (dict(host, obs=frozen_observation(obs)), states, extra)
from collections.abc import Sequence
from itertools import chain, islice
from operator import index

class IndexedReplay(Sequence):

    def __init__(self, maxlen):
        self.maxlen = index(maxlen)
        if self.maxlen < 1:
            raise ValueError('Replay capacity must be positive')
        self._items, self._start = ([], 0)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, position):
        position = index(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError('Replay index out of range')
        return self._items[(self._start + position) % len(self)]

    def __iter__(self):
        return chain(islice(self._items, self._start, None), islice(self._items, 0, self._start))

    def append(self, transition):
        if len(self) < self.maxlen:
            self._items.append(transition)
        else:
            self._items[self._start] = transition
            self._start = (self._start + 1) % self.maxlen
