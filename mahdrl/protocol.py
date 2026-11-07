"""Experiment partitions, cache auditing, instance provenance, and shared scene settings."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / 'configs/f22_split_v1.json'

@dataclass(frozen=True)
class ExperimentProtocol:
    train: tuple
    validation: tuple
    test: tuple
    episodes: int = 600
    selection: str = 'minimum_validation_total_energy'
    version: str = 'f22_split_v1'

    def __post_init__(self):
        if self.version == 'f22_split_v1' and (self.train != tuple(range(600)) or self.validation != (600,) or self.test != tuple(range(700, 730)) or (self.episodes != 600)):
            raise ValueError('f22_split_v1 requires train=0-599, validation=600, test=700-729 and 600 episodes')
        if len(self.validation) != 1:
            raise ValueError('Exactly one validation instance seed is required')
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any((not g for g in groups)) or any((len(g) != len(seq) for g, seq in zip(groups, (self.train, self.validation, self.test)))):
            raise ValueError('Empty or duplicate experiment seeds')
        if any((groups[i] & groups[j] for i in range(3) for j in range(i))):
            raise ValueError('Training, validation and test seeds must be disjoint')
        if min((min(g) for g in groups)) < 0 or len(self.train) < self.episodes:
            raise ValueError('Insufficient training seeds or invalid seeds')
        if self.selection != 'minimum_validation_total_energy':
            raise ValueError('Selection must match the existing energy-only validation rule')
        if self.train != tuple(range(self.train[0], self.train[0] + len(self.train))):
            raise ValueError('Shared existing training loops require contiguous explicit training seeds')

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(tuple(d['train']), tuple(d['validation']), tuple(d['test']), d['episodes'], d['selection'], d['version'])

    def metadata(self):
        d = dict(version=self.version, train=self.train, validation=self.validation, test=self.test, episodes=self.episodes, selection=self.selection)
        return json.loads(json.dumps(d))

def protocol_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protocol_binding(path=DEFAULT_PROTOCOL):
    return dict(experiment_protocol=ExperimentProtocol.load(path).metadata(), experiment_protocol_hash=protocol_hash(path))
from mahdrl.workflow import ENERGY_ACCOUNTING
from mahdrl.workflow import data_model_metadata
from mahdrl.workflow import dax_template_id
from mahdrl.workflow import workflow_instance_hash
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@lru_cache(maxsize=32)
def file_hash(path, size, modified):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def artifact_info(path):
    path = Path(path)
    if not path.is_file():
        return dict(path=str(path), status='missing')
    stat = path.stat()
    return dict(path=str(path), sha256=file_hash(str(path), stat.st_size, stat.st_mtime_ns))

@lru_cache(maxsize=1)
def git_commit():
    root = Path(__file__).resolve().parents[1]
    if (root / '.git').exists():
        try:
            return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return 'unversioned'

def _instance_metadata(scenario):
    protocol = scenario.protocol()
    return dict(data_split=protocol.metadata() if protocol else 'Legacy', model_seed=scenario.model_seed, energy_accounting=ENERGY_ACCOUNTING, source_train_scene=scenario.normalization_kwargs()['source_train_scene'], git_commit=git_commit(), deadline_cache=artifact_info(scenario.cache_path), instance_generator=data_model_metadata(), instance_identity_version='f22_instance_v1')

def instance_identity(env, scenario, instance_seed):
    """Call after all arrivals. Ignore scheduling order, observations and task subdeadlines."""
    if len(env.workflows) != env.workflows_per_episode:
        raise ValueError('Instance hash requires every workflow to have arrived')
    workflows = []
    for i, wf in enumerate(env.workflows):
        name = getattr(wf, 'dax_path', None) or env.wf_dax_name[i]
        workflows.append(dict(template=dax_template_id(name), arrival=float(wf.arrival_time), deadline=float(wf.deadline), payload=workflow_instance_hash(wf.graph, wf.tasks)))
    resources = dict(hosts=[dict(id=k, vm_ids=h.vm_ids, total_pc=h.total_pc, total_bw=h.total_bw, power=[h.power(i / 10) for i in range(11)]) for k, h in sorted(env.hosts.items())], vms=[dict(id=k, **vars(v)) for k, v in sorted(env.vms.items())])
    contents = dict(workflows=workflows, resources=resources)
    identifier = dict(version='f22_instance_v1', task=scenario.task, resource=scenario.resource, deadline=scenario.deadline, instance_seed=int(instance_seed))
    return dict(**identifier, content_hash=stable_hash(contents), instance_hash=stable_hash(dict(**identifier, **contents)))

def write_run_metadata(directory, scenario):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'data_split.json').write_text(json.dumps(scenario.reproducibility_metadata(), indent=2), encoding='utf-8')

def episode_identity(env, scenario):
    seed = getattr(env, 'episode_seed', env.random_seed)
    if len(env.workflows) != env.workflows_per_episode:
        return dict(instance_seed=int(seed), status='incomplete_arrivals')
    return instance_identity(env, scenario, seed)

def _reference_env_kwargs(scenario):
    return dict(dax_paths=scenario.dax_paths(), horizon=1000000000.0, arrival_lambda=0.03, random_seed=0, max_ready_tasks='auto', normalize=True, workflows_per_episode=50, num_hosts=scenario.num_hosts, vms_per_host=scenario.vms_per_host, pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0), bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0), deadline_mode='none', deadline_cache_path=None, deadline_alpha=1.5, deadline_cache_strict=True)
from mahdrl.workflow import DAX_TEMPLATE_ID_SCHEME
from mahdrl.workflow import dax_template_id
from mahdrl.workflow import validate_deadline_cache
import json
from pathlib import Path

def inspect_cache(scenario, path=None):
    path = Path(path or scenario.cache_path)
    protocol = scenario.protocol() or ExperimentProtocol.load(DEFAULT_PROTOCOL)
    required = set(protocol.train) | set(protocol.validation) | set(protocol.test)
    if scenario.mode == 'eval':
        required.update(scenario.seeds)
    if not path.exists():
        return (dict(path=str(path), status='missing', missing=sorted(required)), None)
    obj = json.loads(path.read_text(encoding='utf-8'))
    validate_deadline_cache(obj, path)
    if obj['meta'].get('dax_template_id_scheme') != DAX_TEMPLATE_ID_SCHEME:
        raise ValueError('Deadline cache template identifier version mismatch')
    build_env_kwargs = _reference_env_kwargs
    expected = build_env_kwargs(scenario)
    for key in ('arrival_lambda', 'workflows_per_episode', 'num_hosts', 'vms_per_host', 'pc_tiers', 'bw_tiers'):
        value = list(expected[key]) if isinstance(expected[key], tuple) else expected[key]
        if obj['meta'].get(key) != value:
            raise ValueError('Deadline cache configuration mismatch: ' + key)
    for key in ('task', 'resource'):
        if obj['meta'].get(key) != getattr(scenario, key):
            raise ValueError('Deadline cache scenario mismatch: ' + key)
    if [dax_template_id(x) for x in obj['meta'].get('dax_paths', [])] != [dax_template_id(x) for x in scenario.dax_paths()]:
        raise ValueError('Deadline cache DAX mixture mismatch')
    if obj['meta'].get('policy') != 'FCFS_task + VM_first_idle':
        raise ValueError('Deadline reference makespan policy mismatch')
    rows = obj.get('data')
    if rows is None:
        rows = [dict(v, seed=int(k)) for k, v in obj['seed_to_record'].items()]
    seeds = [int(r['seed']) for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate deadline cache seeds')
    for row in rows:
        if int(row['seed']) not in required:
            continue
        for key in ('arrival_times', 'wf_dax_names', 'wf_makespans', 'wf_instance_hashes'):
            if len(row.get(key, [])) != expected['workflows_per_episode']:
                raise ValueError(f"Deadline cache incomplete {key} for seed={row['seed']}")
    missing = sorted(required - set(seeds))
    return (dict(path=str(path), status='incomplete' if missing else 'ok', missing=missing, required_count=len(required), available_count=len(seeds)), dict(meta=obj['meta'], data=rows))

def check_cache(scenario, path=None):
    report, _ = inspect_cache(scenario, path)
    if report['missing']:
        raise ValueError(f"F22 deadline cache {report['status']}: {report['path']}; missing seeds={seed_ranges(report['missing'])}. Use a complete matching reference cache; no seed fallback is allowed.")
    return report

def seed_ranges(seeds):
    groups = []
    for seed in sorted(set(seeds)):
        if groups and seed == groups[-1][-1] + 1:
            groups[-1].append(seed)
        else:
            groups.append([seed])
    return ','.join((str(g[0]) if len(g) == 1 else f'{g[0]}-{g[-1]}' for g in groups))
from mahdrl.workflow import ENERGY_ACCOUNTING
import hashlib
import json
from pathlib import Path

def source_scene(scenario):
    if scenario.mode != 'eval':
        return f'{scenario.task}{scenario.resource}_{scenario.deadline}'
    return f'{scenario.train_task}{scenario.train_resource or scenario.resource}_{scenario.train_deadline or scenario.deadline}'

def training_name(scenario, model_version, settings):
    calibration = Path(scenario.normalization_config)
    scales = hashlib.sha256(calibration.read_bytes()).hexdigest() if scenario.normalization_mode != 'legacy' and calibration.is_file() else None
    identity = dict(model_version=model_version, settings=settings, normalization=scenario.normalization_mode, calibration_sha256=scales, energy_accounting=ENERGY_ACCOUNTING)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8]
    return f'{source_scene(scenario)}_m{scenario.model_seed}_c{digest}'
from mahdrl.normalization import CURRENT_VERSION
from mahdrl.normalization import DEFAULT_CONFIG
from mahdrl.normalization import SHARED_VERSIONS
from mahdrl.normalization import load_normalizer
from mahdrl.normalization import scene_id
from mahdrl.workflow import DATA_MODEL
from mahdrl.workflow import ENERGY_ACCOUNTING
from dataclasses import dataclass
from pathlib import Path
import argparse
from collections import Counter
import json
ROOT = Path(__file__).resolve().parents[1]
VM_COUNTS = {'S': (9, 8, 8), 'M': (9, 9, 8, 8, 8, 8), 'L': (9, 9, 9, 8, 8, 8, 8, 8, 8)}
SMALL_PROBABILITY = {'Tight': 0.8, 'Medium': 0.5, 'Loose': 0.2}
DAX_GROUPS = (('CyberShake_30.xml', 'Epigenomics_24.xml', 'Ligo_30.xml', 'Montage_25.xml', 'Sipht_29.xml'), ('CyberShake_50.xml', 'Epigenomics_47.xml', 'Ligo_50.xml', 'Montage_50.xml', 'Sipht_58.xml'), ('CyberShake_100.xml', 'Epigenomics_100.xml', 'Ligo_100.xml', 'Montage_100.xml', 'Sipht_97.xml'))

def parse_seeds(value):
    try:
        if '-' in value:
            first, last = map(int, value.split('-'))
            if first < 0 or last < first:
                raise ValueError
            seeds = tuple(range(first, last + 1))
        else:
            seeds = tuple((int(item) for item in value.split(',')))
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError
        return seeds
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('Use an inclusive range such as 700-729, or distinct seeds such as 700,702,705') from None

@dataclass(frozen=True)
class Scenario:
    task: str = 'S'
    resource: str = 'S'
    deadline: str = 'Tight'
    mode: str = 'eval'
    train_task: str = 'S'
    episodes: int = 600
    seed_spec: str = '700-729'
    checkpoint_override: str | None = None
    cache_output: str | None = None
    experiment_protocol: str | None = str(DEFAULT_PROTOCOL)
    model_seed: int = 1

    def protocol(self):
        return ExperimentProtocol.load(self.experiment_protocol)

    def first_training_seed(self):
        return self.protocol().train[0]

    def training_episode_limit(self, protocol):
        return protocol.episodes

    def training_seed(self, episode):
        if not 0 <= episode < self.episodes:
            raise ValueError('Episode outside the configured training budget')
        return self.protocol().train[episode]

    def validation_seeds(self):
        return list(self.protocol().validation)
    normalization_mode: str = CURRENT_VERSION
    normalization_config: str = str(DEFAULT_CONFIG)
    train_resource: str | None = None
    train_deadline: str | None = None

    def arrival_kwargs(self):
        return {}

    @property
    def arrival_lambda(self):
        return 0.03

    @property
    def deadline_choices(self):
        return (2.0, 3.0)

    @property
    def workload_suffix(self):
        return ''

    def with_evaluation_workload_metadata(self, metadata):
        return metadata

    def with_workload_metadata(self, metadata):
        return metadata

    def normalization_kwargs(self):
        source = scene_id(self.task if self.mode == 'train' else self.train_task, self.resource if self.mode == 'train' else self.train_resource or self.resource, self.deadline if self.mode == 'train' else self.train_deadline or self.deadline)
        return dict(normalization_mode=self.normalization_mode, normalization_config=self.normalization_config, source_train_scene=source)

    def normalization_metadata(self):
        kw = self.normalization_kwargs()
        self.validate_calibration()
        n = load_normalizer(kw['normalization_mode'], kw['normalization_config'], kw['source_train_scene'])
        metadata = dict(n.metadata) if n else {'normalization_version': 'legacy'}
        if self.protocol():
            metadata.update(experiment_protocol_hash=protocol_hash(self.experiment_protocol), experiment_protocol=self.protocol().metadata())
        metadata['reproducibility'] = self.reproducibility_metadata()
        metadata['energy_accounting'] = ENERGY_ACCOUNTING
        return self.with_evaluation_workload_metadata(self.with_workload_metadata(metadata))

    def validate_calibration(self):
        if self.protocol() and self.normalization_mode in SHARED_VERSIONS:
            from mahdrl.normalization import validate_calibration
            validate_calibration(self.normalization_config, self.normalization_kwargs()['source_train_scene'], self.experiment_protocol)

    def reproducibility_metadata(self):
        reproducibility_metadata = _instance_metadata
        return reproducibility_metadata(self)

    def __post_init__(self):
        if not 0 <= self.model_seed <= 4294967295:
            raise ValueError('Model seed must be an unsigned 32-bit integer')
        if self.task not in VM_COUNTS or self.resource not in VM_COUNTS or self.train_task not in VM_COUNTS:
            raise ValueError('Task and resource sizes must be S, M or L')
        if self.deadline not in SMALL_PROBABILITY or self.mode not in {'train', 'eval', 'precompute'} or self.episodes < 1:
            raise ValueError('Invalid deadline, mode or episode count')
        if self.train_resource is not None and self.train_resource not in VM_COUNTS:
            raise ValueError('Invalid source resource')
        if self.train_deadline is not None and self.train_deadline not in SMALL_PROBABILITY:
            raise ValueError('Invalid source deadline')
        if self.normalization_mode != CURRENT_VERSION:
            raise ValueError('The main experiment requires f19_shared_v1 normalization')
        if self.protocol():
            protocol = self.protocol()
            if self.mode == 'train' and self.episodes > self.training_episode_limit(protocol):
                raise ValueError('Episode budget exceeds the shared protocol')
            if self.mode == 'eval' and (not set(parse_seeds(self.seed_spec)) <= set(protocol.test)):
                raise ValueError('Evaluation seeds must belong to the selected protocol test partition')
        parse_seeds(self.seed_spec)

    @property
    def vms_per_host(self):
        return VM_COUNTS[self.resource]

    @property
    def num_hosts(self):
        return len(self.vms_per_host)

    @property
    def small_probability(self):
        return SMALL_PROBABILITY[self.deadline]

    @property
    def deadline_probs(self):
        return {'Tight': (0.8, 0.2), 'Medium': (0.5, 0.5), 'Loose': (0.2, 0.8)}[self.deadline]

    @property
    def seeds(self):
        return parse_seeds(self.seed_spec)

    @property
    def cache_name(self):
        return f'ddl_fcfs_task{self.task}_res{self.resource}_seed0-1000.json'

    @property
    def cache_path(self):
        return ROOT / 'data' / 'deadlines' / DATA_MODEL / self.cache_name

    @property
    def precompute_path(self):
        if self.cache_output:
            return Path(self.cache_output).resolve()
        if self.seeds == tuple(range(1001)):
            return self.cache_path
        tag = f'{self.seeds[0]}-{self.seeds[-1]}' if self.seeds == tuple(range(self.seeds[0], self.seeds[-1] + 1)) else '_'.join(map(str, self.seeds))
        return self.cache_path.with_name(f'ddl_fcfs_task{self.task}_res{self.resource}_seed{tag}.json')

    def dax_paths(self, directory=None):
        directory = Path(directory) if directory is not None else ROOT / 'data' / 'dax'
        groups = [[str(directory / name) for name in group] for group in DAX_GROUPS]
        weights = {'S': (1, 0, 0), 'M': (3, 7, 0), 'L': (2, 3, 5)}[self.task]
        return [path for group, weight in zip(groups, weights) for path in group * weight]

    @property
    def output_version(self):
        return self.normalization_mode + '_protocol_' + protocol_hash(self.experiment_protocol)[:12] + '_' + ENERGY_ACCOUNTING

    def describe(self, algorithm=None):
        result = dict(task=self.task, resource=self.resource, deadline=self.deadline, mode=self.mode, experiment_protocol=self.protocol().metadata() if self.protocol() else 'legacy_overlapping_seeds', model_seed=self.model_seed, normalization=self.normalization_kwargs() if self.mode != 'precompute' else None, num_hosts=self.num_hosts, vms_per_host=self.vms_per_host, dax_weights=dict(Counter((Path(p).name for p in self.dax_paths()))), cache=str(self.cache_path), deadline_choices=self.deadline_choices, deadline_probs=self.deadline_probs, episodes=self.episodes if self.mode == 'train' else None, seeds=self.seeds if self.mode != 'train' else None)
        if self.mode == 'precompute':
            result['output'] = str(self.precompute_path)
            result['deadline_mode'] = 'none'
            for key in ('deadline', 'deadline_choices', 'deadline_probs', 'cache'):
                result.pop(key, None)
        return self.with_evaluation_workload_metadata(self.with_workload_metadata(result))
