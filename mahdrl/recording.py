"""Training and frozen-evaluation measurements."""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from contextvars import ContextVar
from functools import wraps
import csv
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
VERSION = 'review_metrics_v1'
_active_validation = ContextVar('active_validation_recorder', default=None)

def enabled():
    return os.environ.get('EXPERIMENT_METRICS', '1') != '0'

def _sync(agents):
    devices = {str(getattr(a, 'device', 'cpu')) for a in agents.values()}
    for device in devices:
        if device.startswith('cuda'):
            torch.cuda.synchronize(torch.device(device))

def workflow_rows(env):
    finishes = {}
    counts = {}
    completed = {}
    for gid, meta in enumerate(env.task_meta):
        wf_id = int(meta.wf_id if hasattr(meta, 'wf_id') else meta[0])
        counts[wf_id] = counts.get(wf_id, 0) + 1
        if env.task_state[gid] == 'Finished':
            completed[wf_id] = completed.get(wf_id, 0) + 1
            finishes[wf_id] = max(finishes.get(wf_id, 0.0), float(env.task_end_time[gid]))
    rows = []
    for i, wf in enumerate(env.workflows):
        done = counts.get(i, 0) > 0 and counts.get(i) == completed.get(i, 0)
        finish = finishes.get(i) if done else None
        arrival = float(wf.arrival_time)
        deadline = float(wf.deadline)
        late = finish - deadline if done else None
        names = getattr(env, 'wf_dax_name', [])
        template = getattr(wf, 'dax_path', None) or (names[i] if len(names) > i else '')
        seed = int(getattr(env, 'episode_seed', env.random_seed))
        cache = getattr(env, '_deadline_cache', None) or getattr(env, '_deadline_cache_by_seed', None) or {}
        refs = cache.get(seed, {}).get('wf_makespans', [])
        reference = float(refs[i]) if len(refs) > i else None
        alpha = (deadline - arrival) / reference if reference and reference > 0 else None
        rows.append(dict(workflow_id=i, template=Path(template).name, arrival=arrival, reference_makespan=reference, deadline_alpha=alpha, deadline=deadline, finish=finish, response_time=finish - arrival if done else None, lateness=late, tardiness=max(0.0, late) if done else None, completed=int(done), success=int(done and late <= 1e-12), task_count=counts.get(i, 0)))
    return rows

def episode_metrics(env):
    rows = workflow_rows(env)
    n = len(rows)
    task_late = [float(env.task_end_time[i]) - float(env.task_baseline_finish[i]) for i, state in enumerate(env.task_state) if state == 'Finished']
    finished = [r for r in rows if r['completed']]
    arrival = min((r['arrival'] for r in rows), default=0.0)
    final = max((r['finish'] for r in finished), default=arrival)
    tardy = [max(0.0, v) for v in task_late]
    return dict(total_energy=float(env.total_energy), first_arrival_time=arrival, final_completion_time=final, makespan=max(0.0, final - arrival), task_count=len(task_late), task_tardy_count=sum((v > 1e-12 for v in tardy)), task_avg_tardiness=float(np.mean(tardy)) if tardy else 0.0, task_success_rate=sum((v <= 1e-12 for v in tardy)) / max(1, len(tardy)), wf_count=n, wf_completed_count=len(finished), wf_success_count=sum((r['success'] for r in rows)), wf_tardy_count=sum((r['tardiness'] > 1e-12 for r in finished)), wf_success_rate=sum((r['success'] for r in rows)) / max(1, n), wf_avg_tardiness=sum((r['tardiness'] for r in finished)) / max(1, len(finished)), wf_avg_response_time=sum((r['response_time'] for r in finished)) / max(1, len(finished)))

def episode_duration(env):
    arrivals = getattr(env, 'arrival_times', [])
    first = float(arrivals[0]) if len(arrivals) else 0.0
    return max(0.0, float(env.current_time) - first)

def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def _append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')

def _wrap(obj, name, factory, restores):
    if not hasattr(obj, name):
        return
    original = getattr(obj, name)
    had = name in getattr(obj, '__dict__', {})
    setattr(obj, name, factory(original))
    restores.append((obj, name, original, had))

def _restore(restores):
    for obj, name, original, had in reversed(restores):
        if had:
            setattr(obj, name, original)
        else:
            delattr(obj, name)
    restores.clear()

def _agent_metadata(agents):
    result = {}
    for name, agent in agents.items():
        optimizers = {k: v for k, v in vars(agent).items() if isinstance(v, torch.optim.Optimizer)}
        params = {id(p): p for o in optimizers.values() for g in o.param_groups for p in g['params']}
        result[name] = dict(device=str(getattr(agent, 'device', 'cpu')), optimizer_parameters=sum((p.numel() for p in params.values())), settings={k: getattr(agent, k) for k in ('gamma', 'batch_size', 'buffer_size', 'grad_clip', 'target_update_tau', 'target_update_freq', '_eps_start', '_eps_end', '_eps_decay_steps') if hasattr(agent, k)}, config=asdict(agent.cfg) if is_dataclass(getattr(agent, 'cfg', None)) else {}, networks={k: str(v) for k, v in vars(agent).items() if isinstance(v, torch.nn.Module)}, optimizer_groups={k: [{a: v for a, v in g.items() if a != 'params' and isinstance(v, (str, int, float, bool, tuple, type(None)))} for g in o.param_groups] for k, o in optimizers.items()})
    return result

class TrainingRecorder:

    def __init__(self, directory, scenario, algorithm, env, agents):
        self.enabled = enabled()
        self.directory = Path(directory)
        self.scenario = scenario
        self.algorithm = algorithm
        self.env = env
        self.agents = agents
        self.restores = []
        self.active = False
        self.index = 0
        self.stage_times = {'validation': 0.0, 'checkpoint': 0.0}
        self.validation_results = []
        self.optimizer_steps = {}
        self.optimizer_samples = {}
        self.update_step_index = {}
        self.assignments = 0
        self.phase_calls = 0
        self.rollout_samples = {}
        self.policy_calls = {}
        self.cumulative_train = 0.0
        self.cumulative_validation = 0.0
        self.cumulative_checkpoint = 0.0
        self.clock = time.perf_counter()
        self.reset_seconds = 0.0
        if not self.enabled:
            return
        previous = self.directory / 'training_measurements.jsonl'
        if previous.exists():
            last = json.loads(previous.read_text(encoding='utf-8').splitlines()[-1])
            self.index = last['episode'] + 1
            self.cumulative_train = last['cumulative_train_seconds']
            self.cumulative_validation = last['cumulative_validation_seconds']
            self.cumulative_checkpoint = last['cumulative_checkpoint_seconds']
        _write_json(self.directory / 'measurement_metadata.json', dict(version=VERSION, algorithm=algorithm, model_seed=scenario.model_seed, task=scenario.task, resource=scenario.resource, deadline=scenario.deadline, python=platform.python_version(), platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_count=os.cpu_count(), numpy=np.__version__, gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], agents=_agent_metadata(agents), environment={k: getattr(env, k) for k in ('arrival_lambda', 'workflows_per_episode', 'num_hosts', 'num_vms', 'deadline_mode', 'deadline_cache_path', 'energy_reward_scale', 'task_baseline_norm', 'alpha_delay_vm', 'alpha_delay_host', 'manager_alpha_delay', 'alpha_delay', 'deadline_alpha_small', 'deadline_alpha_large', 'deadline_alpha_small_prob') if hasattr(env, k)}, timing='wall seconds with CUDA synchronization at stage boundaries; includes instrumentation overhead and timed resets; excludes initial model/probe setup'))
        for role, agent in agents.items():
            self.policy_calls[role] = 0
            self.rollout_samples[role] = 0
            _wrap(agent, 'select_action', lambda f, role=role: self._count_policy(f, role), self.restores)
            _wrap(agent, 'prepare', lambda f, role=role: self._count_policy(f, role), self.restores)
            _wrap(agent, 'select', lambda f, role=role: self._count_policy(f, role), self.restores)
            _wrap(agent, 'save', lambda f: self._timed_save(f), self.restores)
            _wrap(agent, 'update', lambda f, role=role: self._observe_update(f, role), self.restores)
            for name, opt in vars(agent).items():
                if isinstance(opt, torch.optim.Optimizer):
                    key = role + '.' + name
                    self.optimizer_steps[key] = 0
                    self.optimizer_samples[key] = 0
                    _wrap(opt, 'step', lambda f, key=key, role=role: self._count_step(f, key, role), self.restores)
        assignment = next((name for name in ('paper_step', 'pair_step', 'vm_assign_global', 'vm_assign', 'step') if hasattr(env, name)))
        _wrap(env, assignment, lambda f: self._assignment(f), self.restores)
        _wrap(env, 'finish_phase_and_advance', lambda f: self._phase(f), self.restores)
        _wrap(env, 'reset', lambda f: self._reset(f), self.restores)
        self.start()

    def _count_policy(self, f, role):

        @wraps(f)
        def call(*a, **kw):
            result = f(*a, **kw)
            if self.active:
                self.policy_calls[role] += 1
            return result
        return call

    def _count_step(self, f, key, role):

        @wraps(f)
        def call(*a, **kw):
            result = f(*a, **kw)
            if self.active:
                self.optimizer_steps[key] += 1
                agent = self.agents[role]
                cfg = getattr(agent, 'cfg', None)
                if hasattr(agent, 'batch_size'):
                    size = int(agent.batch_size)
                elif hasattr(cfg, 'minibatch_size') and hasattr(agent, 'buf'):
                    count = agent.buf.size()
                    batch = int(cfg.minibatch_size)
                    batches = (count + batch - 1) // batch
                    offset = self.update_step_index.get(role, 0) % max(1, batches)
                    size = min(batch, count - offset * batch)
                elif hasattr(agent, 'buffer') and hasattr(cfg, 'update_rounds'):
                    size = len(agent.buffer)
                elif hasattr(agent, 'f11'):
                    size = 1
                else:
                    size = None
                self.update_step_index[role] = self.update_step_index.get(role, 0) + 1
                if size is None:
                    self.optimizer_samples[key] = None
                elif self.optimizer_samples[key] is not None:
                    self.optimizer_samples[key] += size
            return result
        return call

    def _observe_update(self, f, role):

        @wraps(f)
        def call(*a, **kw):
            self.update_step_index[role] = 0
            result = f(*a, **kw)
            if self.active and isinstance(result, dict) and ('transitions' in result):
                rounds = getattr(getattr(self.agents[role], 'cfg', None), 'update_rounds', 1)
                self.rollout_samples[role] += int(result['transitions']) * rounds
            return result
        return call

    def _phase(self, f):

        @wraps(f)
        def call(*a, **kw):
            result = f(*a, **kw)
            if self.active:
                self.phase_calls += 1
            return result
        return call

    def _assignment(self, f):

        @wraps(f)
        def call(*a, **kw):
            result = f(*a, **kw)
            if self.active:
                self.assignments += 1
            return result
        return call

    def checkpoint(self, function):
        return self._timed_save(function) if self.enabled else function

    def _timed_save(self, f):

        @wraps(f)
        def call(*a, **kw):
            with self.stage('checkpoint'):
                return f(*a, **kw)
        return call

    def _reset(self, f):

        @wraps(f)
        def call(*a, **kw):
            if self.active and self.assignments:
                self.finish()
            _sync(self.agents)
            started = time.perf_counter()
            result = f(*a, **kw)
            _sync(self.agents)
            reset_seconds = time.perf_counter() - started
            self.start()
            self.reset_seconds = reset_seconds
            return result
        return call

    def start(self, episode=None):
        if not self.enabled:
            return
        if episode is not None:
            self.index = episode
        _sync(self.agents)
        self.clock = time.perf_counter()
        self.active = True
        self.reset_seconds = 0.0
        self.stage_times = {'validation': 0.0, 'checkpoint': 0.0}
        self.validation_results = []
        self.assignments = 0
        self.phase_calls = 0
        self.policy_calls = dict.fromkeys(self.policy_calls, 0)
        self.optimizer_steps = dict.fromkeys(self.optimizer_steps, 0)
        self.optimizer_samples = dict.fromkeys(self.optimizer_samples, 0)
        self.rollout_samples = dict.fromkeys(self.rollout_samples, 0)
        self.baseline = {k: getattr(a, 'f11', None).snapshot() if hasattr(a, 'f11') else {} for k, a in self.agents.items()}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        _sync(self.agents)
        started = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.agents)
            self.stage_times[name] += time.perf_counter() - started

    def validation(self, function, *args, **kwargs):
        token = _active_validation.set(self)
        was_active = self.active
        self.active = False
        try:
            with self.stage('validation'):
                return function(*args, **kwargs)
        finally:
            self.active = was_active
            _active_validation.reset(token)

    def finish(self):
        if not self.enabled or not self.active:
            return
        _sync(self.agents)
        wall = time.perf_counter() - self.clock
        train = max(0.0, wall - sum(self.stage_times.values())) + self.reset_seconds
        wall += self.reset_seconds
        self.cumulative_train += train
        self.cumulative_validation += self.stage_times['validation']
        self.cumulative_checkpoint += self.stage_times['checkpoint']
        samples = {}
        for role, a in self.agents.items():
            if hasattr(a, 'f11'):
                snap = a.f11.snapshot()
                base = self.baseline.get(role, {})
                samples[role] = {k: snap.get(k, 0) - base.get(k, 0) for k in ('transitions', 'gradient_updates', 'sampled_targets')}
        record = dict(version=VERSION, algorithm=self.algorithm, episode=self.index, model_seed=self.scenario.model_seed, instance_seed=int(getattr(self.env, 'episode_seed', self.env.random_seed)), train_seconds=train, reset_seconds=self.reset_seconds, validation_seconds=self.stage_times['validation'], checkpoint_seconds=self.stage_times['checkpoint'], episode_wall_seconds=wall, cumulative_train_seconds=self.cumulative_train, cumulative_validation_seconds=self.cumulative_validation, cumulative_checkpoint_seconds=self.cumulative_checkpoint, assignment_calls=self.assignments, phase_advance_calls=self.phase_calls, policy_calls=self.policy_calls.copy(), optimizer_steps=self.optimizer_steps.copy(), optimizer_sample_presentations=self.optimizer_samples.copy(), learning_samples=samples, rollout_transition_presentations=self.rollout_samples.copy(), metrics=episode_metrics(self.env), validation=self.validation_results.copy())
        _append(self.directory / 'training_measurements.jsonl', record)
        self.active = False
        self.index += 1

    def close(self):
        self.finish()
        _restore(self.restores)

def collect_validation(env):
    recorder = _active_validation.get()
    if recorder is not None and recorder.enabled:
        recorder.validation_results.append(dict(seed=int(getattr(env, 'episode_seed', env.random_seed)), **episode_metrics(env)))

class EvaluationRecorder:
    """Synchronized allocation control-path timing, excluding simulator advancement."""

    def __init__(self, scenario, algorithm, env, agents, directory=None, checkpoint_directory=None, checkpoint_paths=None):
        self.enabled = enabled()
        self.env = env
        self.scenario = scenario
        self.algorithm = algorithm
        self.agents = agents
        self.restores = []
        self.depth = 0
        self.pending_seconds = 0.0
        self.samples = []
        self.by_role = {}
        self.state_seconds = 0.0
        self.policy_seconds = 0.0
        self.excluded_seconds = 0.0
        self.cached_state_seconds = 0.0
        self.in_environment = False
        self.api_samples = []
        root = Path(__file__).resolve().parents[1]
        self.directory = Path(directory) if directory else root / 'outputs/run' / VERSION / algorithm / scenario.output_version / (scenario.task + scenario.resource + '_' + scenario.deadline + '_model' + str(scenario.model_seed))
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_paths = checkpoint_paths
        if not self.enabled:
            return
        if checkpoint_paths is not None:
            seed = int(getattr(env, 'episode_seed', env.random_seed))
            for path in (self.directory / 'episodes' / f'seed{seed}.json', self.directory / 'workflows' / f'seed{seed}.csv', self.directory / 'decisions' / f'seed{seed}.json'):
                if path.exists():
                    raise FileExistsError(f'{algorithm}: refusing to overwrite evaluation artifact: {path}')
        for name in ('get_manager_state', 'get_manager_action_mask', 'get_host_state_for_next_assignment', 'get_vm_state_for_current_task', 'get_vm_state_for_next_assignment', 'get_action_mask', 'get_global_state', 'get_local_state', 'get_ready_task_candidates', 'get_vm_obs_and_mask', 'graph_state', '_build_obs', '_build_global_obs', '_build_local_obs', '_build_vm_obs_for_task', '_build_host_obs_for_task', '_build_vm_obs_for_task_host'):
            _wrap(env, name, lambda f: self._timed(f, 'state'), self.restores)
        for role, agent in agents.items():
            for name in ('select_action', 'prepare', 'act', 'select'):
                _wrap(agent, name, lambda f, role=role: self._timed(f, role), self.restores)
        assignment = next((name for name in ('paper_step', 'pair_step', 'vm_assign_global', 'vm_assign', 'step') if hasattr(env, name)))
        _wrap(env, assignment, lambda f: self._allocation(f), self.restores)
        _wrap(env, 'finish_phase_and_advance', lambda f: self._exclude_phase(f), self.restores)
        _sync(self.agents)
        self.boundary_clock = time.perf_counter()

    def _timed(self, f, role):

        @wraps(f)
        def call(*a, **kw):
            outer = self.depth == 0
            if outer:
                _sync(self.agents)
                started = time.perf_counter()
            self.depth += 1
            try:
                return f(*a, **kw)
            finally:
                self.depth -= 1
                if outer:
                    _sync(self.agents)
                    duration = time.perf_counter() - started
                    self.pending_seconds += duration
                    if role == 'state':
                        self.state_seconds += duration
                        if self.in_environment:
                            self.cached_state_seconds += duration
                    else:
                        self.policy_seconds += duration
                        self.by_role.setdefault(role, []).append(duration)
        return call

    def measure(self, function, role):
        return self._timed(function, role) if self.enabled else function

    def _exclude_phase(self, f):

        @wraps(f)
        def call(*a, **kw):
            started = time.perf_counter()
            self.in_environment = True
            try:
                return f(*a, **kw)
            finally:
                self.excluded_seconds += time.perf_counter() - started
                self.in_environment = False
        return call

    def _allocation(self, f):

        @wraps(f)
        def call(*a, **kw):
            _sync(self.agents)
            duration = time.perf_counter() - self.boundary_clock - self.excluded_seconds + self.cached_state_seconds
            self.samples.append(max(0.0, duration))
            self.api_samples.append(self.pending_seconds)
            self.pending_seconds = 0.0
            self.cached_state_seconds = 0.0
            self.excluded_seconds = 0.0
            self.in_environment = True
            try:
                return f(*a, **kw)
            finally:
                _sync(self.agents)
                self.in_environment = False
                self.boundary_clock = time.perf_counter()
        return call

    def finish(self):
        _restore(self.restores)
        if not self.enabled:
            return None
        metrics = episode_metrics(self.env)
        old_metrics = self.env.get_episode_metrics() if hasattr(self.env, 'get_episode_metrics') else {}
        metrics['legacy_environment_makespan'] = float(old_metrics.get('makespan', self.env.current_time))
        seed = int(getattr(self.env, 'episode_seed', self.env.random_seed))
        timing = dict(scope='control path between allocation calls, excluding assignment/event advancement; cached state-encoding API time inside excluded calls is attributed to the next allocation; first reset/probe excluded; includes measurement overhead', cuda_synchronized=True, allocation_count=len(self.samples), decision_mean_ms=1000 * float(np.mean(self.samples)) if self.samples else 0.0, decision_p95_ms=1000 * float(np.percentile(self.samples, 95)) if self.samples else 0.0, decision_p99_ms=1000 * float(np.percentile(self.samples, 99)) if self.samples else 0.0, measured_state_seconds=self.state_seconds, measured_policy_seconds=self.policy_seconds, policy_call_counts={k: len(v) for k, v in self.by_role.items()})
        from mahdrl.protocol import artifact_info
        checkpoints = [artifact_info(p) for p in sorted(Path(self.checkpoint_directory).glob('best*.pth'))] if self.checkpoint_directory else []
        if self.checkpoint_paths is not None:
            checkpoints = [artifact_info(p) for p in self.checkpoint_paths]
        record = dict(version=VERSION, algorithm=self.algorithm, model_seed=self.scenario.model_seed, seed=seed, provenance=self.scenario.reproducibility_metadata(), checkpoints=checkpoints, task=self.scenario.task, resource=self.scenario.resource, deadline=self.scenario.deadline, source_train_scene=self.scenario.normalization_kwargs()['source_train_scene'], checkpoint_directory=self.checkpoint_directory, metrics=metrics, timing=timing)
        _write_json(self.directory / 'measurement_metadata.json', dict(version=VERSION, agents=_agent_metadata(self.agents), python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), timing_scope=timing['scope']))
        _write_json(self.directory / 'episodes' / f'seed{seed}.json', record)
        path = self.directory / 'workflows' / f'seed{seed}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(seed=seed, model_seed=self.scenario.model_seed, **r) for r in workflow_rows(self.env)]
        if rows:
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        _write_json(self.directory / 'decisions' / f'seed{seed}.json', dict(seed=seed, allocation_seconds=self.samples, state_and_policy_api_seconds=self.api_samples, policy_seconds=self.by_role))
        return record
