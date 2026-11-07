"""Main-experiment scene arguments and output paths."""
from mahdrl.config import ALGORITHM
from mahdrl.config import DISPLAY_NAME
from mahdrl.config import MODEL_VERSION
from mahdrl.config import Mechanism
from mahdrl.config import OUTPUT_TAG
from mahdrl.config import SELECTION_TAG
from mahdrl.config import VM_CAPACITY
from mahdrl.config import implementation_metadata
from mahdrl.config import selection_metadata
from mahdrl.config import variant_metadata
from mahdrl.normalization import DEFAULT_CONFIG
from mahdrl.protocol import ROOT
from mahdrl.protocol import Scenario
from mahdrl.protocol import training_name
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

@dataclass(frozen=True)
class ScoreScenario(Scenario):
    algorithm = ALGORITHM
    display_name = DISPLAY_NAME
    model_version = MODEL_VERSION
    output_tag = OUTPUT_TAG
    va_preview: bool = field(default=True, init=False)
    ha_update_period: int = field(default=4, init=False)

    @property
    def mechanism(self):
        return Mechanism(self.va_preview, self.ha_update_period)

    def variant_metadata(self):
        return variant_metadata(self.mechanism)

    def implementation_metadata(self):
        return implementation_metadata(self.mechanism)

    def deadline_reference_scenario(self):
        return self
    cache_file: str | None = None
    output_root: str = str(ROOT)
    device: str | None = None
    diagnostics_every: int = 100
    profile_mode: str = 'sampled'
    profile_every: int = 100

    def __post_init__(self):
        super().__post_init__()
        if self.diagnostics_every < 1 or self.profile_every < 1 or self.profile_mode not in ('off', 'sampled'):
            raise ValueError('Invalid diagnostics or profiling settings')
        if self.mode == 'train' and self.task != 'S':
            raise ValueError('The main experiment trains source task mixture S')
        if self.mode == 'eval' and self.train_task != 'S':
            raise ValueError('The main experiment evaluates source task mixture S')
        if max(self.vms_per_host) > VM_CAPACITY:
            raise ValueError('The fixed VA capacity is nine local VM slots')

    @property
    def cache_path(self):
        return Path(self.cache_file).resolve() if self.cache_file else super().cache_path

    def model_name(self, algorithm=ALGORITHM):
        original = training_name(self, self.model_version, dict(protocol=self.selection_protocol(), variant=self.variant_metadata(), implementation=self.implementation_metadata()))
        return self.source_tag + original[original.index('_m'):]

    @staticmethod
    def scene_tag(task, resource, deadline):
        return task + resource + '_' + {'Tight': 'T', 'Medium': 'M', 'Loose': 'L'}[deadline]

    @property
    def source_tag(self):
        if self.mode != 'eval':
            return self.scene_tag(self.task, self.resource, self.deadline)
        return self.scene_tag(self.train_task, self.train_resource or self.resource, self.train_deadline or self.deadline)

    @property
    def target_tag(self):
        return self.scene_tag(self.task, self.resource, self.deadline)

    def checkpoint_dir(self, algorithm=ALGORITHM):
        if self.mode == 'eval' and self.checkpoint_override:
            return Path(self.checkpoint_override).resolve()
        return Path(self.output_root) / 'outputs/ckpts' / self.output_tag / self.model_name()

    def log_dir(self, algorithm=ALGORITHM):
        return Path(self.output_root) / 'outputs/logs' / self.output_tag / self.model_name()

    @property
    def evaluation_root(self):
        return Path(self.output_root) / 'outputs/run' / self.output_tag

    def evaluation_csv(self):
        seeds = sorted(self.seeds)
        seed_tag = str(seeds[0]) if len(seeds) == 1 else f'{seeds[0]}-{seeds[-1]}'
        if seeds != list(range(seeds[0], seeds[-1] + 1)):
            seed_tag += '_' + hashlib.sha256(json.dumps(seeds).encode()).hexdigest()[:8]
        config_tag = self.model_name().rsplit('_', 1)[-1]
        mechanism_tag = f'p{int(self.va_preview)}h{self.ha_update_period}'
        return self.evaluation_root / f'eval_{self.output_tag}_noapf_srgt092_{mechanism_tag}_{self.target_tag}_from_{self.source_tag}_m{self.model_seed}_s{seed_tag}_{config_tag}{self.workload_suffix}.csv'

    def evaluation_identity(self):
        return dict(target_scene=self.target_tag, source_scene=self.source_tag, task=self.task, resource=self.resource, deadline=self.deadline, train_task=self.train_task, train_resource=self.train_resource or self.resource, train_deadline=self.train_deadline or self.deadline, vm_capacity=VM_CAPACITY, vm_input_dim=93, vm_selection_scope='host_local_fixed_slots')

    def evaluation_dir(self):
        key = hashlib.sha256(self.evaluation_csv().name.encode()).hexdigest()[:16]
        return self.evaluation_root / '_details' / key

    @property
    def output_version(self):
        return self.model_version + '_' + super().output_version + '_' + SELECTION_TAG

    def selection_protocol(self):
        protocol = self.protocol()
        metadata = protocol.metadata() if protocol else dict(version='legacy', validation=self.validation_seeds())
        metadata.update(selection=SELECTION_TAG, checkpoint_selection=selection_metadata(), mechanism=self.variant_metadata()['mechanism'])
        return self.with_workload_metadata(metadata)

    def reproducibility_metadata(self):
        metadata = super().reproducibility_metadata()
        metadata.update(algorithm=self.algorithm, model_version=self.model_version, data_split=self.selection_protocol(), checkpoint_selection=selection_metadata())
        return metadata

    def normalization_metadata(self):
        metadata = super().normalization_metadata()
        metadata.update(model_version=self.model_version, experiment_protocol=self.selection_protocol(), checkpoint_selection=selection_metadata())
        return metadata

    def describe(self, algorithm=ALGORITHM):
        result = super().describe(None)
        result.update(algorithm=self.algorithm, display_name=self.display_name, variant=self.variant_metadata(), implementation=self.implementation_metadata(), diagnostics_every=self.diagnostics_every, profile_mode=self.profile_mode, profile_every=self.profile_every, model_version=self.model_version, checkpoint_selection=selection_metadata(), experiment_protocol=self.selection_protocol(), checkpoint_dir=str(self.checkpoint_dir()), log_dir=str(self.log_dir()), evaluation_dir=str(self.evaluation_dir()), evaluation_csv=str(self.evaluation_csv()), device=self.device or 'auto')
        return result

def parse_scenario(mode, argv=None):
    p = argparse.ArgumentParser(description=DISPLAY_NAME + ': ' + mode)
    p.add_argument('--task', choices=('S',) if mode == 'train' else tuple('SML'), default='S')
    p.add_argument('--resource', choices='SML', default='S')
    p.add_argument('--deadline', choices=('Tight', 'Medium', 'Loose'), default='Tight')
    p.add_argument('--episodes', type=int, choices=(600,), default=600)
    if mode != 'train':
        p.add_argument('--seeds', default='0-1000' if mode == 'precompute' else '700-729')
    if mode == 'eval':
        p.add_argument('--train-task', choices=('S',), default='S')
        p.add_argument('--train-resource', choices='SML')
        p.add_argument('--train-deadline', choices=('Tight', 'Medium', 'Loose'))
        p.add_argument('--checkpoint-dir')
    p.add_argument('--deadline-cache')
    if mode == 'precompute':
        p.add_argument('--out', help='Reference-cache output; never overwrites an existing file')
    p.add_argument('--model-seed', type=int, default=1)
    p.add_argument('--normalization-config', default=str(DEFAULT_CONFIG))
    p.add_argument('--device', choices=('cpu', 'cuda'))
    p.add_argument('--diagnostics-every', type=int, default=100)
    p.add_argument('--profile-mode', choices=('off', 'sampled'), default='sampled')
    p.add_argument('--profile-every', type=int, default=100)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    try:
        scene = ScoreScenario(task=a.task, resource=a.resource, deadline=a.deadline, mode=mode, episodes=a.episodes, seed_spec=getattr(a, 'seeds', '700-729'), train_task=getattr(a, 'train_task', 'S'), train_resource=getattr(a, 'train_resource', None), train_deadline=getattr(a, 'train_deadline', None), checkpoint_override=getattr(a, 'checkpoint_dir', None), cache_output=getattr(a, 'out', None), cache_file=a.deadline_cache, model_seed=a.model_seed, normalization_config=a.normalization_config, device=a.device, diagnostics_every=a.diagnostics_every, profile_mode=a.profile_mode, profile_every=a.profile_every)
    except (ValueError, OSError, TypeError) as exc:
        p.error(str(exc))
    if a.dry_run:
        print(json.dumps(scene.describe(), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    return scene
