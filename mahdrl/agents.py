"""The TA, HA, and VA learners with masked Double DQN and indexed replay."""
from mahdrl.learning import LEARNING_MODEL
from mahdrl.learning import LearningDiagnostics
from mahdrl.learning import validate_transition
from mahdrl.networks import DuelingQNet
from mahdrl.networks import _polyak_update
from mahdrl.normalization import check_checkpoint
from mahdrl.normalization import checkpoint_metadata
import random
from collections import deque
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class D3QNAgent:

    def __init__(self, input_dim: int, output_dim: int, lr: float=0.0003, gamma: float=0.99, batch_size: int=128, buffer_size: int=100000, eps_start: float=1.0, eps_end: float=0.05, eps_decay_steps: int=150000, target_update_freq: int=2000, target_update_tau: float=0.005, grad_clip: float=10.0, hidden_dims: Tuple[int, int]=(512, 256), device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f11 = LearningDiagnostics()
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.grad_clip = grad_clip
        head_hidden_dims = (hidden_dims[-1], hidden_dims[-1])
        self.online = DuelingQNet(input_dim, output_dim, hidden_dims, head_hidden_dims=head_hidden_dims).to(device)
        self.target = DuelingQNet(input_dim, output_dim, hidden_dims, head_hidden_dims=head_hidden_dims).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.f11.numerics.inputs(self.online, 'online_input')
        self.f11.numerics.inputs(self.target, 'target_input')
        self._eps_start = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = 0
        self._updates = 0
        self.target_update_freq = max(1, int(target_update_freq))
        self.target_update_tau = float(target_update_tau)
        self._action_calls = 0

    def epsilon(self) -> float:
        t = min(self._eps_steps, self._eps_decay_steps)
        frac = 1.0 - t / self._eps_decay_steps
        return float(self._eps_end + (self._eps_start - self._eps_end) * frac)

    def _valid_indices_from_mask(self, mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask > 0.5)
        if idx.size == 0:
            return np.array([self.output_dim - 1], dtype=np.int64)
        return idx

    def select_action(self, state: np.ndarray, mask: np.ndarray, deterministic: bool=False, count_step: bool=True) -> int:
        if count_step:
            self._action_calls += 1
        eps = 0.0 if deterministic else self.epsilon()
        valid = self._valid_indices_from_mask(mask)
        if not deterministic and random.random() < eps:
            return int(np.random.choice(valid))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(s).squeeze(0).clone()
            m = torch.tensor(mask, dtype=torch.float32, device=self.device)
            q[m <= 0.5] = -1000000000.0
            a = int(torch.argmax(q).item())
            return a

    def remember(self, state: np.ndarray, mask: np.ndarray, action: int, reward: float, next_state: np.ndarray, next_mask: np.ndarray, done: float):
        validate_transition(state, mask, action, reward, next_state, next_mask, done)
        self.f11.transition(done)
        self.buffer.append((np.array(state, copy=True), np.array(mask, copy=True), int(action), float(reward), np.array(next_state, copy=True), np.array(next_mask, copy=True), float(done)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, m, a, r, s2, m2, d = map(np.array, zip(*batch))
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        m = torch.tensor(m, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        m2 = torch.tensor(m2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        q_all = self.online(s)
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q2_online = self.online(s2).masked_fill(m2 <= 0.5, -1000000000.0)
            a2 = q2_online.argmax(dim=1, keepdim=True)
            q2_target = self.target(s2).masked_fill(m2 <= 0.5, -1000000000.0)
            q2 = q2_target.gather(1, a2).squeeze(1)
            has_valid = (m2 > 0.5).any(dim=1)
            q2 = torch.where(has_valid, q2, torch.zeros_like(q2))
            target = r + (1.0 - d) * self.gamma * q2
        self.f11.target((1.0 - d) * self.gamma * q2, target)
        loss = F.smooth_l1_loss(q_sa, target)
        self.f11.numerics.observe('Q_selected', q_sa, scalar=True)
        self.f11.numerics.observe('TD_target', target, scalar=True)
        self.f11.numerics.observe('TD_error', target - q_sa, scalar=True)
        loss_value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()
        self._updates += 1
        self.f11.updates += 1
        self._eps_steps += 1
        if self.target_update_tau > 0.0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        elif self._updates % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss_value

    def save(self, path: str):
        torch.save({'normalization': checkpoint_metadata(self), 'learning_model': LEARNING_MODEL, 'online': self.online.state_dict(), 'target': self.target.state_dict(), 'optim': self.optim.state_dict(), 'updates': self._updates, 'eps_steps': self._eps_steps, 'eps_start': self._eps_start, 'eps_end': self._eps_end, 'eps_decay_steps': self._eps_decay_steps, 'tau': self.target_update_tau, 'hard_freq': self.target_update_freq}, path)

    def load(self, path: str, strict: bool=True, require_learning_model: bool=False):
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        check_checkpoint(ckpt.get('normalization') if isinstance(ckpt, dict) else None, checkpoint_metadata(self))
        if require_learning_model and ckpt.get('learning_model') != LEARNING_MODEL:
            raise ValueError(f'Checkpoint is not an F11-trained model: {path}')
        self.online.load_state_dict(ckpt['online'], strict=strict)
        self.target.load_state_dict(ckpt['target'])
        self.optim.load_state_dict(ckpt['optim'])
        self._updates = ckpt.get('updates', 0)
        self._eps_steps = ckpt.get('eps_steps', 0)
        self._eps_start = ckpt.get('eps_start', self._eps_start)
        self._eps_end = ckpt.get('eps_end', self._eps_end)
        self._eps_decay_steps = ckpt.get('eps_decay_steps', self._eps_decay_steps)
        self.target_update_tau = ckpt.get('tau', self.target_update_tau)
        self.target_update_freq = ckpt.get('hard_freq', self.target_update_freq)
from mahdrl.config import APF
from mahdrl.config import ENTITY
from mahdrl.config import TRUNK
from mahdrl.config import _core_set_config_variant_metadata
from mahdrl.learning import LearningDiagnostics
from mahdrl.networks import CandidateDuelingQNet
from mahdrl.networks import _polyak_update
from mahdrl.networks import collate
from mahdrl.networks import copy_observation
from mahdrl.networks import validate_observation
from mahdrl.normalization import check_checkpoint
from collections import deque
from copy import deepcopy
import random
import uuid
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CheckpointIdentity:

    def bind(self, scenario, role):
        self.role = role
        shapes = {k: list(v.shape) for k, v in self.online.state_dict().items()}
        self.identity = dict(_core_set_config_variant_metadata(), role=role, parameter_shapes=shapes)
        self.normalization_metadata = dict(scenario.normalization_metadata(), algorithm_variant=deepcopy(self.identity))
        return self

    def save(self, path):
        if not hasattr(self, 'identity'):
            raise ValueError('Bind source normalization and variant role before saving')
        self.normalization_metadata['checkpoint_group'] = getattr(self, 'checkpoint_group', None) or uuid.uuid4().hex
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError('MAHDRL-AS requires strict checkpoint loading')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        actual = checkpoint.get('normalization', {})
        if actual.get('algorithm_variant') != getattr(self, 'identity', None):
            raise ValueError('MAHDRL-AS variant/architecture/role mismatch')
        check_checkpoint(actual, self.normalization_metadata)
        if not actual.get('checkpoint_group'):
            raise ValueError('Missing MAHDRL-AS checkpoint group')
        super().load(path, strict=True, require_learning_model=require_learning_model)
        self.loaded_group = actual['checkpoint_group']
        self.checkpoint_group = self.loaded_group

class _TaskLearnerBase(CheckpointIdentity, D3QNAgent):
    pass

class _SetLearnerBase(CheckpointIdentity, D3QNAgent):

    def __init__(self, context_dim, candidate_dim, lr=0.0003, gamma=0.95, batch_size=256, buffer_size=120000, eps_start=1.0, eps_end=0.05, eps_decay_steps=80000, target_update_tau=0.005, target_update_freq=2000, grad_clip=10.0, hidden_dims=TRUNK, entity_dims=ENTITY, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.context_dim, self.candidate_dim = (context_dim, candidate_dim)
        self.gamma, self.batch_size, self.grad_clip = (gamma, batch_size, grad_clip)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.f11 = LearningDiagnostics()
        self.online = CandidateDuelingQNet(context_dim, candidate_dim, hidden_dims, entity_dims).to(self.device)
        self.target = CandidateDuelingQNet(context_dim, candidate_dim, hidden_dims, entity_dims).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._eps_start, self._eps_end = (float(eps_start), float(eps_end))
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = self._updates = self._action_calls = 0
        self.target_update_tau, self.target_update_freq = (target_update_tau, target_update_freq)
        self.cfg = APF

    def _observe_inputs(self, state, name):
        self.f11.numerics.observe(name + '_context', state[0])
        features = state[1][state[2] > 0.5]
        if features.numel():
            self.f11.numerics.observe(name + '_candidates', features)

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_observation(state, mask)
        valid = np.flatnonzero(np.asarray(mask) > 0.5)
        if count_step:
            self._action_calls += 1
        if not deterministic and random.random() < self.epsilon():
            if performance_times is None:
                raise ValueError('GE training requires candidate completion times')
            times = np.asarray(performance_times)
            if times.shape != np.shape(mask) or not np.isfinite(times[valid]).all() or np.any(times[valid] < 0):
                raise ValueError('Invalid GE candidate times')
            if random.random() < APF.guided_probability:
                fastest = times[valid].min()
                valid = valid[np.isclose(times[valid], fastest, rtol=1e-09, atol=1e-09)]
            return int(np.random.choice(valid))
        batch = collate([state], [mask], self.device)
        self._observe_inputs(batch, 'online_input')
        with torch.no_grad():
            q = self.online(*batch).masked_fill(batch[3] <= 0.5, -torch.inf)
            return int(q.argmax(1).item())

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        validate_observation(state, mask)
        validate_observation(next_state, next_mask, decision=not done)
        if done not in (0, 1, False, True) or not np.isfinite(reward):
            raise ValueError('Invalid set transition reward/stop flag')
        if not 0 <= int(action) < len(mask) or mask[int(action)] <= 0.5:
            raise ValueError('Illegal recorded set action')
        if state['context'].shape != (self.context_dim,) or state['candidates'].shape[1] != self.candidate_dim:
            raise ValueError('Set transition feature schema mismatch')
        self.f11.transition(done)
        self.buffer.append((copy_observation(state), np.array(mask, copy=True), int(action), float(reward), copy_observation(next_state), np.array(next_mask, copy=True), bool(done)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        obs, mask, actions, rewards, next_obs, next_mask, done = zip(*batch)
        state = collate(obs, mask, self.device)
        self._observe_inputs(state, 'online_input')
        a = torch.tensor(actions, dtype=torch.int64, device=self.device)
        q = self.online(*state).gather(1, a[:, None]).squeeze(1)
        future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            indices = [i for i, stop in enumerate(done) if not stop]
            if indices:
                successor = collate([next_obs[i] for i in indices], [next_mask[i] for i in indices], self.device)
                self._observe_inputs(successor, 'target_input')
                best = self.online(*successor).masked_fill(successor[3] <= 0.5, -torch.inf).argmax(1)
                value = self.target(*successor).gather(1, best[:, None]).squeeze(1)
                future[indices] = self.gamma * value
            target = torch.tensor(rewards, dtype=torch.float32, device=self.device) + future
        self.f11.target(future, target)
        loss = F.smooth_l1_loss(q, target)
        self.f11.numerics.observe('Q_selected', q, scalar=True)
        self.f11.numerics.observe('TD_target', target, scalar=True)
        self.f11.numerics.observe('TD_error', target - q, scalar=True)
        value = self.f11.loss(loss)
        self.optim.zero_grad()
        loss.backward()
        self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        if self.target_update_tau > 0:
            _polyak_update(self.target, self.online, self.target_update_tau)
        elif self._updates % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return value
from mahdrl.config import APF
from mahdrl.config import _core_fast_config_ALGORITHM
from mahdrl.config import _core_fast_config_MODEL_VERSION
from mahdrl.config import _core_fast_config_implementation_metadata
from mahdrl.config import _core_fast_config_variant_metadata
from mahdrl.config import _core_selection_legacy_selection_metadata
from mahdrl.config import _core_selection_legacy_validate_checkpoint_selection
from mahdrl.diagnostics import StageProfiler
from mahdrl.learning import FastDiagnostics
from mahdrl.learning import validate_transition
from mahdrl.networks import _polyak_update
from mahdrl.networks import validate_observation
from mahdrl.replay import collate
from mahdrl.replay import frozen_array
from mahdrl.replay import frozen_observation
from mahdrl.replay import pack
from mahdrl.replay import to_device
import random
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FastLearner:
    is_set = False

    def __init__(self, *args, diagnostics_every=100, profile_mode='sampled', profile_every=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(self.device)
        self.f11.numerics.enabled = False
        self.f11 = FastDiagnostics(diagnostics_every)
        if not self.is_set:
            self.f11.numerics.inputs(self.online, 'online_input')
            self.f11.numerics.inputs(self.target, 'target_input')
        self.profiler = StageProfiler(profile_mode, profile_every)

    def bind(self, scenario, role):
        super().bind(scenario, role)
        self.identity = dict(_core_fast_config_variant_metadata(), role=role, parameter_shapes={key: list(value.shape) for key, value in self.online.state_dict().items()})
        self.normalization_metadata['algorithm_variant'] = deepcopy(self.identity)
        self.normalization_metadata['implementation'] = _core_fast_config_implementation_metadata()
        self.selection_expectation = dict(algorithm=_core_fast_config_ALGORITHM, model_version=_core_fast_config_MODEL_VERSION, learner_role=role, checkpoint_selection=_core_selection_legacy_selection_metadata(), protocol=scenario.selection_protocol(), model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()['source_train_scene'], validation_seeds=scenario.validation_seeds())
        return self

    def save(self, path):
        _core_selection_legacy_validate_checkpoint_selection(self.normalization_metadata.get('mahdrl_as_fast'), self.selection_expectation)
        super().save(path)

    def load(self, path, strict=True, require_learning_model=False):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('normalization'), dict):
            raise ValueError(f'Missing {_core_fast_config_MODEL_VERSION} checkpoint normalization/selection metadata')
        actual = checkpoint.get('normalization', {})
        record = _core_selection_legacy_validate_checkpoint_selection(actual.get('mahdrl_as_fast'), self.selection_expectation)
        for key in ('model_version', 'checkpoint_selection', 'experiment_protocol'):
            if actual.get(key) != self.normalization_metadata.get(key):
                raise ValueError('MAHDRL-AS-Fast checkpoint selection mismatch: ' + key)
        super().load(path, strict=strict, require_learning_model=require_learning_model)
        self.loaded_selection = deepcopy(record)
        self.normalization_metadata['mahdrl_as_fast'] = deepcopy(record)

    def select_action(self, *args, deterministic=False, **kwargs):
        n = self._action_calls
        self.f11.numerics.active = not deterministic and (n == 0 or (n + 1) % self.f11.numerics.every == 0)
        try:
            return super().select_action(*args, deterministic=deterministic, **kwargs)
        finally:
            self.f11.numerics.active = False

    def remember(self, state, mask, action, reward, next_state, next_mask, done):
        if self.is_set:
            validate_observation(state, mask)
            validate_observation(next_state, next_mask, decision=not done)
            for obs in (state, next_state):
                if obs['context'].shape != (self.context_dim,) or obs['candidates'].shape[1] != self.candidate_dim:
                    raise ValueError('Set transition feature schema mismatch')
        else:
            validate_transition(state, mask, action, reward, next_state, next_mask, done)
            if np.shape(state) != (self.input_dim,) or np.shape(next_state) != (self.input_dim,) or np.shape(mask) != (self.output_dim,) or (np.shape(next_mask) != (self.output_dim,)):
                raise ValueError('TA transition feature schema mismatch')
            if any((np.any((np.asarray(m) != 0) & (np.asarray(m) != 1)) for m in (mask, next_mask))):
                raise ValueError('TA masks must be binary')
        if done not in (0, 1, False, True) or not np.isfinite(reward) or int(action) != action or (not 0 <= int(action) < len(mask)) or (mask[int(action)] <= 0.5):
            raise ValueError('Invalid recorded transition/action')
        copy = frozen_observation if self.is_set else frozen_array
        transition = (copy(state), frozen_array(mask), int(action), float(reward), copy(next_state), frozen_array(next_mask), bool(done))
        self.buffer.append(transition)
        self.f11.transition(done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        self.profiler.select(self._updates)
        self.f11.numerics.active = self._updates == 0 or (self._updates + 1) % self.f11.numerics.every == 0
        try:
            return self._update()
        finally:
            self.f11.numerics.active = False
            self.profiler.active = False

    def _update(self):
        p = self.profiler
        with p.stage('replay_pack'):
            batch = random.sample(self.buffer, self.batch_size)
            obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
            if self.is_set:
                current = pack(obs, masks)
                indices = np.flatnonzero(~np.asarray(done, dtype=bool)).tolist()
                following = pack([successors[i] for i in indices], [next_masks[i] for i in indices]) if indices else None
                if np.any(current[3].sum(1) == 0) or (following is not None and np.any(following[3].sum(1) == 0)):
                    raise ValueError('Replay nonterminal decision has no legal action')
            else:
                current = np.stack(obs)
                current_legal = np.stack(masks)
                following, next_legal = (np.stack(successors), np.stack(next_masks))
                if not all((np.isfinite(x).all() for x in (current, following, current_legal, next_legal))):
                    raise ValueError('Non-finite TA replay batch')
                if any((np.any((m != 0) & (m != 1)) for m in (current_legal, next_legal))) or np.any(current_legal.sum(1) == 0) or np.any((next_legal.sum(1) == 0) & ~np.asarray(done, dtype=bool)):
                    raise ValueError('Invalid TA replay masks')
        with p.stage('h2d', self.device):
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            if self.is_set:
                state = to_device(current, self.device)
                following = to_device(following, self.device) if following is not None else None
            else:
                state = (torch.as_tensor(current, dtype=torch.float32, device=self.device),)
                following = (torch.as_tensor(following, dtype=torch.float32, device=self.device),)
                next_legal = torch.as_tensor(next_legal, dtype=torch.float32, device=self.device)
                stops = torch.tensor(done, dtype=torch.float32, device=self.device)
        with p.stage('diagnostics_inputs'):
            if self.is_set and self.f11.numerics.active and self.f11.numerics.enabled:
                self._observe_inputs(state, 'online_input')
        with p.stage('online_current_forward', self.device):
            q = self.online(*state).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            if self.is_set:
                future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
                if following is not None:
                    with p.stage('diagnostics_inputs'):
                        if self.f11.numerics.active and self.f11.numerics.enabled:
                            self._observe_inputs(following, 'target_input')
                    with p.stage('online_next_forward', self.device):
                        best = self.online(*following).masked_fill(following[3] <= 0.5, -torch.inf).argmax(1)
                    with p.stage('target_forward', self.device):
                        value = self.target(*following).gather(1, best[:, None]).squeeze(1)
                    future[indices] = self.gamma * value
            else:
                with p.stage('online_next_forward', self.device):
                    best = self.online(*following).masked_fill(next_legal <= 0.5, -1000000000.0).argmax(1, keepdim=True)
                with p.stage('target_forward', self.device):
                    value = self.target(*following).masked_fill(next_legal <= 0.5, -1000000000.0).gather(1, best).squeeze(1)
                    value = torch.where((next_legal > 0.5).any(1), value, torch.zeros_like(value))
                future = (1.0 - stops) * self.gamma * value
            target = reward + future
        loss = F.smooth_l1_loss(q, target)
        with p.stage('diagnostics'):
            self.f11.target(future, target)
            self.f11.numerics.observe('Q_selected', q, scalar=True)
            self.f11.numerics.observe('TD_target', target, scalar=True)
            if self.f11.numerics.active and self.f11.numerics.enabled:
                self.f11.numerics.observe('TD_error', target - q, scalar=True)
            value = self.f11.loss(loss)
        with p.stage('backward_clip_optimizer', self.device):
            self.optim.zero_grad()
            loss.backward()
            self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip, error_if_nonfinite=True)
            self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        with p.stage('target_update', self.device):
            if self.target_update_tau > 0:
                _polyak_update(self.target, self.online, self.target_update_tau)
            elif self._updates % self.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        return value

class _FastTaskLearner(FastLearner, _TaskLearnerBase):
    pass

class _SetPolicy(_SetLearnerBase):

    def _observe_inputs(self, state, name):
        if self.f11.numerics.active and self.f11.numerics.enabled:
            super()._observe_inputs(state, name)

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_observation(state, mask)
        valid = np.flatnonzero(np.asarray(mask) > 0.5)
        if count_step:
            self._action_calls += 1
        if not deterministic and random.random() < self.epsilon():
            if performance_times is None:
                raise ValueError('GE training requires candidate completion times')
            times = np.asarray(performance_times)
            if times.shape != np.shape(mask) or not np.isfinite(times[valid]).all() or np.any(times[valid] < 0):
                raise ValueError('Invalid GE candidate times')
            if random.random() < APF.guided_probability:
                fastest = times[valid].min()
                valid = valid[np.isclose(times[valid], fastest, rtol=1e-09, atol=1e-09)]
            return int(np.random.choice(valid))
        batch = collate([state], [mask], self.device)
        self._observe_inputs(batch, 'online_input')
        with torch.no_grad():
            q = self.online(*batch).masked_fill(batch[3] <= 0.5, -torch.inf)
            return int(q.argmax(1).item())

class _FastSetLearner(FastLearner, _SetPolicy):
    is_set = True
from mahdrl.config import CHECKPOINT_METADATA_KEY
from mahdrl.config import TRUNK
from mahdrl.config import VM_CAPACITY
from mahdrl.config import VM_INPUT_DIM
from mahdrl.config import selection_metadata
from mahdrl.config import validate_checkpoint_selection
from mahdrl.diagnostics import StageProfiler
from mahdrl.learning import FastDiagnostics
from mahdrl.networks import FixedVMDuelingQNet
from mahdrl.networks import _polyak_update
from mahdrl.replay import behavior_probabilities
from mahdrl.replay import validate_fixed
from collections import deque
from copy import deepcopy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class VariantIdentity:

    def bind(self, scenario, role):
        if role not in getattr(scenario, 'learned_roles', ('TA', 'HA', 'VA')):
            raise ValueError('Unknown learner role')
        if role == 'HA' and (not self.is_set or (self.context_dim, self.candidate_dim) != (23, scenario.mechanism.host_features)):
            raise ValueError('HScore HA observation schema mismatch')
        if role == 'VA' and (self.is_set or (self.input_dim, self.output_dim) != (VM_INPUT_DIM, VM_CAPACITY)):
            raise ValueError('HScore VA requires fixed 93/9 dimensions')
        if role == 'TA' and (self.is_set or (self.input_dim, self.output_dim) != (15, 243)):
            raise ValueError('HScore TA observation schema mismatch')
        CheckpointIdentity.bind(self, scenario, role)
        self.mechanism = scenario.mechanism
        self.checkpoint_key = getattr(scenario, 'checkpoint_key', CHECKPOINT_METADATA_KEY)
        self.eligible_steps = 0
        self.model_version = scenario.model_version
        self.implementation = scenario.implementation_metadata()
        self.identity = dict(scenario.variant_metadata(), role=role, parameter_shapes={k: list(v.shape) for k, v in self.online.state_dict().items()})
        self.normalization_metadata.update(algorithm_variant=deepcopy(self.identity), implementation=deepcopy(self.implementation))
        self.selection_expectation = dict(algorithm=scenario.algorithm, model_version=self.model_version, learner_role=role, checkpoint_selection=selection_metadata(), protocol=scenario.selection_protocol(), model_seed=scenario.model_seed, source_train_scene=scenario.normalization_kwargs()['source_train_scene'], validation_seeds=scenario.validation_seeds(), mechanism=self.identity['mechanism'])
        return self

    def _check_metadata(self, actual):
        if not isinstance(actual, dict):
            raise ValueError('Missing HScore checkpoint metadata')
        for key, expected in dict(model_version=self.model_version, algorithm_variant=self.identity, implementation=self.implementation, experiment_protocol=self.selection_expectation['protocol'], checkpoint_selection=selection_metadata()).items():
            if actual.get(key) != expected:
                raise ValueError('HScore checkpoint mismatch: ' + key)
        return validate_checkpoint_selection(actual.get(self.checkpoint_key), self.selection_expectation)

    def save(self, path):
        self.normalization_metadata['ha_eligible_steps'] = self.eligible_steps
        self._check_metadata(self.normalization_metadata)
        CheckpointIdentity.save(self, path)

    def load(self, path, strict=True, require_learning_model=False):
        if not strict:
            raise ValueError('HScore requires strict checkpoint loading')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('Invalid HScore checkpoint')
        record = self._check_metadata(checkpoint.get('normalization'))
        steps = checkpoint['normalization'].get('ha_eligible_steps')
        if type(steps) is not int or steps < 0:
            raise ValueError('Invalid HA exploration clock')
        if self.role == 'HA' and checkpoint.get('updates') != steps // self.mechanism.ha_update_period or (self.role != 'HA' and steps != 0):
            raise ValueError('HA exploration and optimizer clocks disagree')
        CheckpointIdentity.load(self, path, strict=True, require_learning_model=require_learning_model)
        self.eligible_steps, self.loaded_selection = (steps, deepcopy(record))
        self.normalization_metadata[self.checkpoint_key] = deepcopy(record)

class _TaskLearner(VariantIdentity, _FastTaskLearner):
    pass

class _HostLearner(VariantIdentity, _FastSetLearner):

    def epsilon(self):
        fraction = 1 - min(self.eligible_steps, self._eps_decay_steps) / self._eps_decay_steps
        return float(self._eps_end + (self._eps_start - self._eps_end) * fraction)

class _FixedVMLearner(VariantIdentity, FastLearner, _TaskLearnerBase):
    is_set = False

    def __init__(self, *, lr=0.0003, gamma=0.95, batch_size=256, buffer_size=120000, eps_start=1.0, eps_end=0.05, eps_decay_steps=80000, target_update_tau=0.005, target_update_freq=2000, grad_clip=10.0, hidden_dims=TRUNK, device='cpu', diagnostics_every=100, profile_mode='sampled', profile_every=100):
        self.input_dim, self.output_dim = (VM_INPUT_DIM, VM_CAPACITY)
        self.device = torch.device(device)
        self.gamma, self.batch_size, self.grad_clip = (gamma, batch_size, grad_clip)
        self.buffer = deque(maxlen=buffer_size)
        self.online = FixedVMDuelingQNet(hidden_dims).to(self.device)
        self.target = deepcopy(self.online)
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self._eps_start, self._eps_end = (float(eps_start), float(eps_end))
        self._eps_decay_steps = max(1, int(eps_decay_steps))
        self._eps_steps = self._updates = self._action_calls = 0
        self.target_update_tau, self.target_update_freq = (target_update_tau, target_update_freq)
        self.f11 = FastDiagnostics(diagnostics_every)
        self.f11.numerics.inputs(self.online, 'online_input')
        self.f11.numerics.inputs(self.target, 'target_input')
        self.profiler = StageProfiler(profile_mode, profile_every)

    def select_preview_action(self, state, *, training):
        validate_fixed(state['obs'], state['mask'])
        if self._updates != state['va_version']:
            raise RuntimeError('VA changed between preview and action')
        if state['epsilon'] != (self.epsilon() if training else 0.0):
            raise RuntimeError('VA epsilon changed between preview and action')
        if training:
            self._action_calls += 1
            return int(np.random.choice(VM_CAPACITY, p=state['probabilities']))
        legal = np.flatnonzero(state['mask'] > 0.5)
        return int(legal[np.argmax(state['q'][legal])])

    def select_action(self, state, mask, deterministic=False, count_step=True, performance_times=None):
        validate_fixed(state, mask)
        epsilon = 0.0 if deterministic else self.epsilon()
        if epsilon and performance_times is None:
            raise ValueError('GE training requires candidate completion times')
        obs = torch.as_tensor(np.array(state, copy=True)[None], dtype=torch.float32, device=self.device)
        legal = torch.as_tensor(np.array(mask, copy=True)[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.online(obs, legal).cpu().numpy()[0]
        p = behavior_probabilities(q, mask, np.zeros(VM_CAPACITY) if performance_times is None else performance_times, epsilon)
        if count_step:
            self._action_calls += 1
        if deterministic:
            return int(np.flatnonzero(mask > 0.5)[np.argmax(q[mask > 0.5])])
        return int(np.random.choice(VM_CAPACITY, p=p))

    def _update(self):
        p = self.profiler
        with p.stage('replay_pack'):
            batch = random.sample(self.buffer, self.batch_size)
            obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
            for s, m, n, nm, d in zip(obs, masks, successors, next_masks, done):
                validate_fixed(s, m)
                validate_fixed(n, nm, decision=not d)
            arrays = [np.stack(x) for x in (obs, masks, successors, next_masks)]
        with p.stage('h2d', self.device):
            state, legal, following, next_legal = [torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrays]
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            ongoing = torch.tensor([not d for d in done], dtype=torch.bool, device=self.device)
        with p.stage('online_current_forward', self.device):
            q = self.online(state, legal).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            if ongoing.any():
                nxt, mask = (following[ongoing], next_legal[ongoing])
                with p.stage('online_next_forward', self.device):
                    best = self.online(nxt, mask).masked_fill(mask <= 0.5, -torch.inf).argmax(1)
                with p.stage('target_forward', self.device):
                    future[ongoing] = self.gamma * self.target(nxt, mask).gather(1, best[:, None]).squeeze(1)
            target = reward + future
        loss = F.smooth_l1_loss(q, target)
        with p.stage('diagnostics'):
            self.f11.target(future, target)
            self.f11.numerics.observe('Q_selected', q, scalar=True)
            self.f11.numerics.observe('TD_target', target, scalar=True)
            self.f11.numerics.observe('TD_error', target - q, scalar=True)
            value = self.f11.loss(loss)
        with p.stage('backward_clip_optimizer', self.device):
            self.optim.zero_grad()
            loss.backward()
            self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip, error_if_nonfinite=True)
            self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        with p.stage('target_update', self.device):
            if self.target_update_tau > 0:
                _polyak_update(self.target, self.online, self.target_update_tau)
            elif self._updates % self.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        return value
from mahdrl.config import VM_CAPACITY
from mahdrl.config import VM_INPUT_DIM
from mahdrl.networks import _polyak_update
from mahdrl.replay import IndexedReplay
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def pack_fixed_batch(batch):
    obs, masks, actions, rewards, successors, next_masks, done = zip(*batch)
    state, legal, following, next_legal = (np.stack(x) for x in (obs, masks, successors, next_masks))
    b = len(batch)
    if state.shape != (b, VM_INPUT_DIM) or following.shape != state.shape or legal.shape != (b, VM_CAPACITY) or (next_legal.shape != legal.shape):
        raise ValueError('Fixed VA requires a batch of 93-vectors and 9-action masks')
    if not all((np.isfinite(x).all() for x in (state, legal, following, next_legal))):
        raise ValueError('Invalid fixed VA state or binary mask')
    if any((np.any((m != 0) & (m != 1)) for m in (legal, next_legal))):
        raise ValueError('Invalid fixed VA state or binary mask')
    ongoing = np.flatnonzero(~np.asarray(done, dtype=bool))
    if np.any(~legal.any(1)) or np.any(~next_legal[ongoing].any(1)):
        raise ValueError('No legal VM at a non-terminal decision')
    return (state, legal, actions, rewards, following[ongoing], next_legal[ongoing], ongoing)

class _IndexedLearner:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = IndexedReplay(self.buffer.maxlen)

class TAAgent(_IndexedLearner, _TaskLearner):
    pass

class HAAgent(_IndexedLearner, _HostLearner):
    pass

class FixedVAAgent(_IndexedLearner, _FixedVMLearner):

    def _update(self):
        p = self.profiler
        with p.stage('replay_pack'):
            batch = random.sample(self.buffer, self.batch_size)
            state, legal, actions, rewards, following, next_legal, ongoing = pack_fixed_batch(batch)
        with p.stage('h2d', self.device):
            state, legal, following, next_legal = [torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in (state, legal, following, next_legal)]
            action = torch.tensor(actions, dtype=torch.int64, device=self.device)
            reward = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            indices = torch.as_tensor(ongoing, dtype=torch.int64, device=self.device)
        with p.stage('online_current_forward', self.device):
            q = self.online(state, legal).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            future = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            if len(ongoing):
                with p.stage('online_next_forward', self.device):
                    best = self.online(following, next_legal).masked_fill(next_legal <= 0.5, -torch.inf).argmax(1)
                with p.stage('target_forward', self.device):
                    future[indices] = self.gamma * self.target(following, next_legal).gather(1, best[:, None]).squeeze(1)
            target = reward + future
        loss = F.smooth_l1_loss(q, target)
        with p.stage('diagnostics'):
            self.f11.target(future, target)
            self.f11.numerics.observe('Q_selected', q, scalar=True)
            self.f11.numerics.observe('TD_target', target, scalar=True)
            self.f11.numerics.observe('TD_error', target - q, scalar=True)
            value = self.f11.loss(loss)
        with p.stage('backward_clip_optimizer', self.device):
            self.optim.zero_grad()
            loss.backward()
            self.f11.numerics.gradients(self.online.parameters(), self.grad_clip)
            if self.grad_clip is not None and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip, error_if_nonfinite=True)
            self.optim.step()
        self._updates += 1
        self._eps_steps += 1
        self.f11.updates += 1
        with p.stage('target_update', self.device):
            if self.target_update_tau > 0:
                _polyak_update(self.target, self.online, self.target_update_tau)
            elif self._updates % self.target_update_freq == 0:
                self.target.load_state_dict(self.online.state_dict())
        return value
