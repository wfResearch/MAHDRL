"""Real-successor transitions, reward settlement, and assignment update barriers."""
from mahdrl.diagnostics import NumericalDiagnostics
import json
import os
from pathlib import Path
import numpy as np
LEARNING_MODEL = 'f11_real_decisions_v1'

def valid_decision(obs, mask):
    if not np.isfinite(obs).all() or not np.isfinite(mask).all():
        raise ValueError('Non-finite decision state/mask')
    if not np.any(np.asarray(mask) > 0.5):
        raise ValueError('No legal action at a non-terminal decision')

def snapshot_decision(state):
    return {**state, 'obs': np.array(state['obs'], copy=True), 'mask': np.array(state['mask'], copy=True)}

def validate_transition(state, mask, action, reward, next_state, next_mask, done):
    valid_decision(state, mask)
    if not 0 <= int(action) < len(mask) or mask[int(action)] <= 0.5:
        raise ValueError('Illegal recorded action')
    if not np.isfinite(reward) or not np.isfinite(next_state).all() or (not np.isfinite(next_mask).all()):
        raise ValueError('Non-finite transition')
    if done not in (0, 1, False, True):
        raise ValueError('Learning stop flag must be boolean')
    if not done:
        valid_decision(next_state, next_mask)

class LearningDiagnostics:

    def __init__(self):
        self.enabled = os.environ.get('F11_DIAGNOSTICS', '1') != '0'
        self.numerics = NumericalDiagnostics()
        self.transitions = self.nonterminal = self.updates = 0
        self.targets = self.bootstrap_nonzero = 0
        self.future_abs_sum = self.target_sum = 0.0
        self.last_loss = self.max_abs_loss = 0.0
        self.last_rollout_value = None
        self.nonterminal_rollouts = self.truncation_cutoffs = 0
        self.last_nonterminal_rollout_value = None

    def transition(self, done):
        if self.enabled:
            self.transitions += 1
            self.nonterminal += int(not done)

    def target(self, future, target):
        if self.enabled:
            f = future.detach().cpu().numpy().reshape(-1)
            t = target.detach().cpu().numpy().reshape(-1)
            self.targets += len(t)
            self.bootstrap_nonzero += int(np.count_nonzero(np.abs(f) > 1e-12))
            self.future_abs_sum += float(np.abs(f).sum())
            self.target_sum += float(t.sum())

    def loss(self, loss):
        self.numerics.observe('loss', loss, scalar=True)
        value = float(loss.detach().item())
        if not np.isfinite(value):
            raise FloatingPointError('Non-finite learning loss')
        if self.enabled:
            self.last_loss = value
            self.max_abs_loss = max(self.max_abs_loss, abs(value))
        return value

    def snapshot(self):
        return dict(transitions=self.transitions, nonterminal=self.nonterminal, nonterminal_ratio=self.nonterminal / max(self.transitions, 1), gradient_updates=self.updates, sampled_targets=self.targets, last_loss=self.last_loss, max_abs_loss=self.max_abs_loss, bootstrap_nonzero_ratio=self.bootstrap_nonzero / max(self.targets, 1), future_abs_mean=self.future_abs_sum / max(self.targets, 1), target_mean=self.target_sum / max(self.targets, 1), last_rollout_value=self.last_rollout_value, nonterminal_rollouts=self.nonterminal_rollouts, last_nonterminal_rollout_value=self.last_nonterminal_rollout_value, truncation_cutoffs=self.truncation_cutoffs, terminal_transitions=self.transitions - self.nonterminal - self.truncation_cutoffs)

class F11Logger:

    def __init__(self, directory):
        self.path = Path(directory) / 'f11_diagnostics.jsonl'

    def log(self, episode, agents, **fields):
        if os.environ.get('F11_DIAGNOSTICS', '1') == '0':
            return
        record = dict(learning_model=LEARNING_MODEL, episode=episode, learners={k: v.f11.snapshot() for k, v in agents.items()}, numerics={k: v.f11.numerics.snapshot() for k, v in agents.items()}, **fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

def protect_training_outputs(save_dir, log_dir):
    """Versioned output names protect legacy artifacts; also refuse reusing a completed run."""
    for directory in (save_dir, log_dir):
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Refusing to overwrite training artifacts: {path}')

class PendingDecision:
    """One learner's consecutive actual decisions, independent of phases or host changes."""

    def __init__(self, emit):
        self.emit = emit
        self.pending = None
        self.recorded_reward = self.emitted_reward = 0.0
        self.cutoffs = 0

    def advance(self, obs, mask, host_index=None):
        valid_decision(obs, mask)
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError('Previous decision has no actual successor yet')
        valid_decision(obs, mask)
        self.pending = dict(obs=np.array(obs, copy=True), mask=np.array(mask, copy=True), action=int(action), reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def add_reward(self, reward):
        if self.pending is None:
            if abs(reward) > 1e-12:
                raise RuntimeError('Reward without a preceding learner decision')
            return
        self.pending['reward'] += float(reward)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, np.array(obs, copy=True), np.array(mask, copy=True), bool(done), next_host_index)
        self.emitted_reward += p['reward']
        self.pending = None

    def finish(self, *, truncated=False):
        if self.pending is not None:
            self.cutoffs += int(truncated)
            self._emit(np.zeros_like(self.pending['obs']), np.zeros_like(self.pending['mask']), True, None)
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError('Unsettled decision rewards')

def _f11_finish_lower_phase(env, host_trace, vm_trace, info):
    """Only energy not already returned by vm_assign belongs to this extra advance."""
    if not info.get('lower_energy_already_emitted', True):
        energy = float(info.get('r_energy_phase', 0.0))
        if host_trace is not None:
            host_trace.add_reward((1 - env.alpha_delay_host) * energy)
        if vm_trace is not None and vm_trace.pending is not None:
            host = vm_trace.pending['host_index']
            local = info.get('r_energy_by_host', {}).get(host, 0.0)
            vm_energy = energy if getattr(env, 'vm_energy_scope', 'selected_host') == 'global' else local
            vm_trace.add_reward((1 - env.alpha_delay_vm) * vm_energy)
    if env.done_flag:
        truncated = not env._episode_should_end()
        for trace in (host_trace, vm_trace):
            if trace is not None:
                trace.finish(truncated=truncated)

class _f11_LowerRewardAudit:
    """Conservation checks for existing team HA and selected-host VA event rewards."""

    def __init__(self, host_trace, vm_trace):
        self.host_start = host_trace.recorded_reward if host_trace else 0.0
        self.vm_start = vm_trace.recorded_reward if vm_trace else 0.0
        self.energy = self.local_energy = self.delay = self.phase_energy = 0.0

    def assignment(self, info):
        self.energy += float(info.get('r_energy_total', 0.0))
        self.local_energy += float(info.get('r_energy_host', 0.0))
        self.delay += float(info.get('r_delay', 0.0))

    def phase(self, info, vm_trace):
        self.phase_energy += float(info.get('phase_energy', 0.0))
        if not info.get('lower_energy_already_emitted', True):
            self.energy += float(info.get('r_energy_phase', 0.0))
            if vm_trace and vm_trace.pending is not None:
                self.local_energy += float(info.get('r_energy_by_host', {}).get(vm_trace.pending['host_index'], 0.0))

    def summary(self, env, host_trace, vm_trace):
        result = dict(physical_energy_balance=self.phase_energy - float(env.total_energy), lower_energy_balance=self.energy + float(env.total_energy) * env.energy_reward_scale, team_energy_reward=self.energy, selected_host_energy_reward=self.local_energy)
        for name, trace, start, alpha, energy in (('HA', host_trace, self.host_start, env.alpha_delay_host, self.energy), ('VA', vm_trace, self.vm_start, env.alpha_delay_vm, self.energy if getattr(env, 'vm_energy_scope', 'selected_host') == 'global' else self.local_energy)):
            if trace is not None:
                result[name + '_reward_formula_balance'] = trace.recorded_reward - start - ((1 - alpha) * energy + alpha * self.delay)
        return result
from mahdrl.diagnostics import NumericalDiagnostics
import numpy as np
import torch

class SampledNumerics(NumericalDiagnostics):

    def __init__(self, every):
        super().__init__()
        self.every, self.active = (every, False)

    def observe(self, name, values, scalar=False):
        if self.active:
            super().observe(name, values, scalar)

    def gradients(self, parameters, threshold=None, name='gradient'):
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if grads and (not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all().item())):
            raise FloatingPointError('Nonfinite ' + name)
        if not self.active or not self.enabled:
            return None
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g).double() for g in grads]))) if grads else 0.0
        if not np.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        self.observe(name, [norm], scalar=True)
        counts = self.clipping.setdefault(name, dict(observations=0, clipped=0, threshold=threshold))
        counts['observations'] += 1
        counts['clipped'] += int(threshold is not None and threshold > 0 and (norm > threshold))
        return norm

    def snapshot(self):
        return dict(super().snapshot(), sample_every=self.every, sample_scope='first_and_every_N_updates_or_policy_calls; validation_disabled')

class FastDiagnostics(LearningDiagnostics):

    def __init__(self, every=100):
        super().__init__()
        if every < 1:
            raise ValueError('diagnostics-every must be positive')
        self.numerics = SampledNumerics(every)
        self.pending_totals = None

    def transition(self, done):
        self.transitions += 1
        self.nonterminal += int(not done)

    def target(self, future, target):
        self.targets += target.numel()
        with torch.no_grad():
            totals = torch.stack(((future.abs() > 1e-12).sum().double(), future.abs().sum().double(), target.sum().double()))
            if self.pending_totals is None:
                self.pending_totals = totals
            else:
                self.pending_totals.add_(totals)

    def snapshot(self):
        if self.pending_totals is not None:
            nonzero, future, target = self.pending_totals.cpu().tolist()
            self.bootstrap_nonzero += int(nonzero)
            self.future_abs_sum += future
            self.target_sum += target
            self.pending_totals = None
        return dict(super().snapshot(), diagnostics_every=self.numerics.every, count_scope='all_transitions_updates_and_sampled_targets')

def _core_reward_finish_lower_phase(env, host_trace, vm_trace, info):
    delta = float(info.get('lower_assignment_energy_unemitted', 0.0))
    if delta:
        for trace, alpha in ((host_trace, env.alpha_delay_host), (vm_trace, env.alpha_delay_vm)):
            if trace is None or trace.pending is None:
                raise RuntimeError('Forced assignment energy has no preceding learner decision')
            trace.add_reward(-(1.0 - alpha) * delta * env.energy_reward_scale)
    if env.done_flag:
        truncated = not env._episode_should_end()
        for trace in (host_trace, vm_trace):
            if trace is not None:
                trace.finish(truncated=truncated)

class _core_reward_LowerRewardAudit:

    def __init__(self, host_trace, vm_trace):
        self.host_start = host_trace.recorded_reward if host_trace else 0.0
        self.vm_start = vm_trace.recorded_reward if vm_trace else 0.0
        self.energy = self.delay = self.phase_energy = 0.0

    def assignment(self, info):
        self.energy += float(info.get('r_energy_total', 0.0))
        self.delay += float(info.get('r_delay', 0.0))

    def phase(self, info, vm_trace):
        self.phase_energy += float(info.get('phase_energy', 0.0))
        self.unreturned_j = getattr(self, 'unreturned_j', 0.0) + float(info.get('lower_assignment_energy_unemitted', 0.0))

    def summary(self, env, host_trace, vm_trace):
        energy = self.energy - getattr(self, 'unreturned_j', 0.0) * env.energy_reward_scale
        physical_gap = env.assignment_energy_total - env.total_energy
        result = dict(physical_energy_balance=self.phase_energy - env.total_energy, lower_energy_balance=energy + env.assignment_energy_total * env.energy_reward_scale, team_energy_reward=energy, selected_host_energy_reward=energy, assignment_energy_total=env.assignment_energy_total, assignment_energy_physical_gap=physical_gap, complete_episode=bool(env._episode_should_end()))
        for name, trace, start, alpha in (('HA', host_trace, self.host_start, env.alpha_delay_host), ('VA', vm_trace, self.vm_start, env.alpha_delay_vm)):
            if trace is not None:
                result[name + '_reward_formula_balance'] = trace.recorded_reward - start - ((1.0 - alpha) * energy + alpha * self.delay)
        return result
from mahdrl.replay import frozen_array
from mahdrl.replay import frozen_observation
import numpy as np

class DecisionTrace(PendingDecision):

    def __init__(self, clock, role):
        self.clock, self.role = (clock, role)
        self.agent = clock.agents[role]

        def emit(p, obs, mask, done, next_host):
            clock.remember(role, p['obs'], p['mask'], p['action'], p['reward'], obs, mask, float(done))
        super().__init__(emit)

    def advance(self, obs, mask, host_index=None):
        if self.pending is not None:
            self._emit(obs, mask, False, host_index)

    def record(self, obs, mask, action, reward, host_index=None):
        if self.pending is not None:
            raise RuntimeError('Previous decision has no actual successor yet')
        copy = frozen_observation if self.agent.is_set else frozen_array
        self.pending = dict(obs=copy(obs), mask=frozen_array(mask), action=int(action), reward=float(reward), host_index=host_index)
        self.recorded_reward += float(reward)

    def _emit(self, obs, mask, done, next_host_index):
        p = self.pending
        self.emit(p, obs, mask, bool(done), next_host_index)
        self.emitted_reward += p['reward']
        self.pending = None

    def finish(self, *, truncated=False):
        if self.pending is not None:
            self.cutoffs += int(truncated)
            if self.clock.training:
                self.agent.f11.truncation_cutoffs += int(truncated)
            obs = self.pending['obs']
            if self.agent.is_set:
                empty = dict(context=np.zeros_like(obs['context']), candidates=np.zeros((0, obs['candidates'].shape[1]), np.float32), exists_mask=np.zeros(0, np.float32))
                mask = np.zeros(0, np.float32)
            else:
                empty, mask = (np.zeros_like(obs), np.zeros_like(self.pending['mask']))
            self._emit(empty, mask, True, None)
        if not np.isclose(self.recorded_reward, self.emitted_reward):
            raise RuntimeError('Unsettled synchronization decision rewards')

class UpdateClock:

    def __init__(self, agents, *, training=False, warmup=0.03):
        if set(agents) != {'TA', 'HA', 'VA'} or not 0 <= warmup <= 1:
            raise ValueError('Invalid agents or warmup')
        self.agents, self.training, self.warmup = (agents, training, warmup)
        self.debt = dict.fromkeys(agents, 0)
        self.inserted = dict.fromkeys(agents, 0)
        self.requested = dict.fromkeys(agents, 0)
        self.executed = dict.fromkeys(agents, 0)

    def remember(self, role, *transition):
        if not self.training:
            return
        agent = self.agents[role]
        agent.remember(*transition)
        self.inserted[role] += 1
        if len(agent.buffer) < max(agent.batch_size, int(agent.buffer.maxlen * self.warmup)):
            return
        if role == 'HA':
            agent.eligible_steps += 1
            if agent.eligible_steps % agent.mechanism.ha_update_period:
                return
        self.debt[role] += 1
        self.requested[role] += 1

    def barrier(self):
        for role, count in self.debt.items():
            for _ in range(count):
                agent = self.agents[role]
                before = agent._updates
                agent.update()
                if agent._updates != before + 1:
                    raise RuntimeError('A scheduled update did not execute')
                self.executed[role] += 1
            self.debt[role] = 0

    def summary(self):
        if any(self.debt.values()) or self.executed != self.requested:
            raise RuntimeError('Unsettled update credits')
        return dict(transitions=self.inserted, update_requests=self.requested, optimizer_steps=self.executed, ha_eligible_steps=self.agents['HA'].eligible_steps)
