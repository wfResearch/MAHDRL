"""Frozen source-training observation normalization."""
from mahdrl.diagnostics import NumericalDiagnostics
import hashlib
import json
from pathlib import Path
import numpy as np
VERSION = 'f18_shared_v1'
CURRENT_VERSION = 'f19_shared_v1'
SHARED_VERSIONS = (VERSION, CURRENT_VERSION)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'data/calibration/f18_shared_normalization_v1.json'
SCALE_UNITS = dict(R_ref='tasks', T_ref='s', workload_mi_ref='MI', input_bits_ref='bit', output_bits_ref='bit', up_rank_ref='s', down_rank_ref='s', vm_pc_ref='MI/s', vm_bw_ref='Mb/s', host_pc_ref='MI/s')

def scene_id(task, resource, deadline):
    return task + resource + '_' + {'Loose': 'L', 'Medium': 'M', 'Tight': 'T'}[deadline]

class StateNormalizer:

    def __init__(self, path, source_scene, version=VERSION):
        if version not in SHARED_VERSIONS:
            raise ValueError('Unknown shared observation version')
        path = Path(path).resolve()
        raw = path.read_bytes()
        obj = json.loads(raw)
        if obj.get('normalization_version') not in SHARED_VERSIONS:
            raise ValueError('F18 normalization version mismatch')
        if obj.get('units') != SCALE_UNITS:
            raise ValueError('F18 physical units mismatch')
        if source_scene not in obj.get('scenes', {}):
            raise ValueError('F18 source training scene has no completed calibration: ' + source_scene)
        self.scales = dict(obj['scenes'][source_scene])
        for key in SCALE_UNITS:
            value = self.scales.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (not np.isfinite(value)) or (value <= 0):
                raise ValueError('Invalid F18 positive reference: ' + key)
        self.numerics = NumericalDiagnostics()
        self.version = version
        self.metadata = dict(normalization_version=version, source_train_scene=source_scene, normalization_config_path=str(path), normalization_config_hash=hashlib.sha256(raw).hexdigest(), R_ref=self.scales['R_ref'], T_ref=self.scales['T_ref'])
        if version == CURRENT_VERSION:
            self.io_ratio_ref = (self.scales['input_bits_ref'] + self.scales['output_bits_ref']) / self.scales['workload_mi_ref']
            if not np.isfinite(self.io_ratio_ref) or self.io_ratio_ref <= 0:
                raise ValueError('Invalid I/O ratio reference')
            self.metadata.update(io_ratio_encoding='bits_per_MI_divided_by_reference_v1', io_ratio_ref=self.io_ratio_ref, scale_config_version=obj['normalization_version'])

    def io_ratio(self, raw_ratio):
        if self.version == VERSION:
            return raw_ratio
        reference = self.io_ratio_ref
        normalized = float(raw_ratio / reference)
        self.numerics.observe('io_ratio_raw_bits_per_MI', [raw_ratio], scalar=True)
        self.numerics.observe('io_ratio_dimensionless', [normalized], scalar=True)
        return normalized

    def scale(self, value, key):
        result = np.asarray(value, dtype=np.float64) / self.scales[key]
        if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
            raise ValueError('Nonfinite F18 observation: ' + key)
        return float(result) if result.ndim == 0 else result

    def rr(self, count):
        return min(1.0, max(0.0, self.scale(count, 'R_ref')))

def load_normalizer(mode='legacy', path=None, source_scene=None):
    if mode == 'legacy':
        return None
    if mode not in SHARED_VERSIONS or not source_scene:
        raise ValueError('Specify f19_shared_v1/f18_shared_v1 and a source training scene, or explicit legacy')
    return StateNormalizer(path or DEFAULT_CONFIG, source_scene, version=mode)

def validate_calibration(path, source_scene, protocol_path):
    from mahdrl.protocol import ExperimentProtocol
    from mahdrl.protocol import protocol_binding
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    generation = obj.get('generation', {})
    for key, value in protocol_binding(protocol_path).items():
        if generation.get(key) != value:
            raise ValueError('Calibration protocol missing or mismatched; regenerate with the selected protocol: ' + key)
    seeds = generation.get('calibration_seeds', {}).get(source_scene, [])
    protocol = ExperimentProtocol.load(protocol_path)
    if not seeds or len(set(seeds)) != len(seeds) or (not set(seeds) <= set(protocol.train)):
        raise ValueError('Calibration seeds must belong to the selected training partition: ' + source_scene)

def counts(env):
    """Arrays use unique global task IDs and contain arrived tasks only."""
    states = env.task_state
    items = states.items() if isinstance(states, dict) else enumerate(states)
    unfinished = {int(gid): state for gid, state in items if state != 'Finished'}
    ready = {gid for gid, state in unfinished.items() if state == 'Ready'}
    blocked = {gid for gid, state in unfinished.items() if state == 'unReady'}
    running = {gid for gid, state in unfinished.items() if state == 'Running'}
    return dict(ready_count=len(ready), blocked_count=len(blocked), running_count=len(running), unfinished_count=len(unfinished))

def queue_metrics(env):
    c = counts(env)
    now = float(env.current_time)
    tids = [i for i in set(env.ready_task_ids) if env.task_state[i] == 'Ready']
    waits = [max(0.0, now - env.task_ready_time[i]) for i in tids]
    slacks = []
    for tid in tids:
        wf_id, _ = env.task_meta[tid]
        deadline = getattr(env.workflows[wf_id], 'deadline', None)
        if deadline is not None:
            duration = np.mean([env._task_duration(tid, j) for j in range(env.num_vms)])
            slacks.append(float(deadline) - now - float(duration))
    return dict(c, waiting_time_mean=float(np.mean(waits)) if waits else 0.0, waiting_time_max=max(waits, default=0.0), min_workflow_slack=min(slacks, default=0.0), avg_workflow_slack=float(np.mean(slacks)) if slacks else 0.0, urgent_ratio=sum((s < 0 for s in slacks)) / max(1, len(tids)))

def queue_features(env):
    m = queue_metrics(env)
    n = env.state_normalizer
    observe_rr(env, m['ready_count'])
    return [n.rr(m['ready_count']), m['blocked_count'] / max(1, m['unfinished_count']), m['running_count'] / max(1, env.num_vms), n.scale(m['waiting_time_mean'], 'T_ref'), n.scale(m['waiting_time_max'], 'T_ref'), n.scale(m['min_workflow_slack'], 'T_ref'), n.scale(m['avg_workflow_slack'], 'T_ref'), m['urgent_ratio']]

def checkpoint_metadata(agent):
    return dict(getattr(agent, 'normalization_metadata', {'normalization_version': 'legacy'}))

def check_checkpoint(actual, expected):
    actual = actual or {'normalization_version': 'legacy'}
    expected = expected or {'normalization_version': 'legacy'}
    for key in ('normalization_version', 'observation_layout', 'source_train_scene', 'normalization_config_hash', 'R_ref', 'T_ref', 'experiment_protocol_hash', 'io_ratio_encoding', 'io_ratio_ref', 'scale_config_version', 'energy_accounting'):
        if actual.get(key) != expected.get(key):
            raise ValueError('Checkpoint normalization mismatch: ' + key)

def observe_rr(env, ready_count):
    n = getattr(env, 'state_normalizer', None)
    if n is None or ready_count <= 0:
        return
    episode = (env.random_seed, id(env.task_state))
    if getattr(env, '_f18_rr_episode', None) != episode:
        env._f18_rr_episode = episode
        env._f18_rr_seen = set()
        env._f18_rr_total = env._f18_rr_saturated = 0
    key = (float(env.current_time), tuple(env.task_state))
    if key not in env._f18_rr_seen:
        env._f18_rr_seen.add(key)
        env._f18_rr_total += 1
        env._f18_rr_saturated += int(n.rr(ready_count) >= 1.0)

def normalization_diagnostics(env):
    n = getattr(env, 'state_normalizer', None)
    if n is None:
        return {'normalization_version': 'legacy'}
    total = getattr(env, '_f18_rr_total', 0)
    return dict(n.metadata, rr_observations=total, rr_saturated=getattr(env, '_f18_rr_saturated', 0), rr_saturation_ratio=getattr(env, '_f18_rr_saturated', 0) / max(1, total), numerics=n.numerics.snapshot())

def checked_observation(env, obs):
    if getattr(env, 'state_normalizer', None) and (not np.isfinite(obs).all()):
        raise ValueError('F18 observation exceeds the finite network-input range')
    return obs
