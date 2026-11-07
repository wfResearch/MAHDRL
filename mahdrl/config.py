"""Main-experiment settings, model identities, and checkpoint selection."""
from dataclasses import asdict, dataclass
_core_set_config_ALGORITHM = 'mahdrl_oe_ge_apf_score'
_core_set_config_DISPLAY_NAME = 'MAHDRL-AS (OE + GE + APF + shared scoring)'
ARCHITECTURE = 'shared_candidate_dueling_v1'
OBSERVATION = 'task23_host10_taskhost30_vm7_v1'
_core_set_config_MODEL_VERSION = 'mahdrl_as_v1'

@dataclass(frozen=True)
class APFConfig:
    scheme: str = 'adaptive_pf_energy_v1'
    host_base: float = 1.5
    vm_base: float = 1.25
    spare_weight: float = 0.5
    urgency_weight: float = 0.25
    lower: float = 1.25
    upper: float = 2.0
    guided_probability: float = 0.5
    time_atol: float = 1e-09
    energy_rtol: float = 1e-09
    energy_atol: float = 1e-09
    energy_retention: str = 'all_tied_minimum_marginal_energy_among_ontime_raw_legal'
APF = APFConfig()
TRUNK = (1024, 1024, 512, 512, 256)
ENTITY = (128, 128)
TA_HIDDEN = (512, 512, 256, 128)

def _core_set_config_variant_metadata():
    return dict(algorithm=_core_set_config_ALGORITHM, scheme=_core_set_config_MODEL_VERSION, model_version=_core_set_config_MODEL_VERSION, architecture=ARCHITECTURE, observation_layout=OBSERVATION, components=['OE', 'GE', 'APF', 'shared_scoring'], oe_base_scheme='mahdrl_oe_1235_v1', apf=asdict(APF), host_summary_scope='final_apf_candidates', deadline_action_filter=False, pooling='exists_mask_mean_max_log1p_real_and_legal_counts')

def learning_kwargs(role, device):
    return dict(lr=0.0003, gamma=0.95, batch_size=256, buffer_size=60000 if role == 'TA' else 120000, eps_start=1.0, eps_end=0.05, eps_decay_steps=40000 if role == 'TA' else 80000, target_update_tau=0.01 if role == 'TA' else 0.005, grad_clip=10.0, device=device)
import math
SELECTION_TAG = 'select_srgt092_minenergy_v1'
_selection_WORKFLOW_SUCCESS_THRESHOLD = 0.92

def selection_metadata():
    return dict(version=SELECTION_TAG, rule='validation_feasible_then_minimum_energy', workflow_success_threshold=_selection_WORKFLOW_SUCCESS_THRESHOLD, workflow_success_comparison='strictly_greater', feasibility_metric='total_successful_workflows_divided_by_total_workflows', energy_metric='mean_validation_instance_total_energy', success_metric='total_successful_workflows_divided_by_total_workflows', fallback='maximum_workflow_success_then_minimum_energy', tie_break='higher_workflow_success_then_earlier_checkpoint', population_probability_guaranteed=False)

def validation_selection(rows):
    if not rows:
        raise ValueError('Checkpoint selection requires validation instances')
    successes = total = 0
    energies = []
    for row in rows:
        count, success = (row['wf_count'], row['wf_success_count'])
        energy = float(row['total_energy'])
        if not math.isfinite(count) or not math.isfinite(success) or count <= 0 or (int(count) != count) or (int(success) != success) or (not 0 <= success <= count) or (not math.isfinite(energy)) or (energy < 0):
            raise ValueError('Invalid validation workflow counts or energy')
        if not math.isclose(float(row['wf_success_rate']), success / count, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Validation workflow success rate/count mismatch')
        successes += int(success)
        total += int(count)
        energies.append(energy)
    return dict(workflow_success_rate=successes / total, workflow_success_count=successes, workflow_count=total, mean_total_energy=sum(energies) / len(rows), validation_instances=len(rows), feasible=successes / total > _selection_WORKFLOW_SUCCESS_THRESHOLD)

def selection_key(candidate):
    if candidate['feasible']:
        return (0, candidate['mean_total_energy'], -candidate['workflow_success_rate'])
    return (1, -candidate['workflow_success_rate'], candidate['mean_total_energy'])

def validate_checkpoint_selection(record, expected):
    """Recompute saved evidence before accepting any new-rule checkpoint."""
    if not isinstance(record, dict):
        raise ValueError('Missing main-experiment checkpoint selection metadata; retrain under the new rule')
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('MAHDRL-AS-Fast checkpoint selection mismatch: ' + key)
    episode = record.get('episode')
    selected = record.get('selected_episode')
    if type(episode) is not int or episode < 0 or type(selected) is not int or (not 0 <= selected <= episode) or (record.get('completed_episodes') != episode + 1):
        raise ValueError('Invalid checkpoint selection episode')
    train_seeds = expected['protocol'].get('train')
    budget = len(train_seeds) if train_seeds is not None else None
    if budget is not None and episode >= budget:
        raise ValueError('Checkpoint episode outside training protocol')
    summaries = []
    for name in ('validation_rows', 'best_validation_rows'):
        rows = record.get(name)
        if not isinstance(rows, list) or any((not isinstance(row, dict) for row in rows)) or [row.get('seed') for row in rows] != expected['validation_seeds']:
            raise ValueError('Checkpoint validation seeds mismatch: ' + name)
        try:
            summaries.append(validation_selection(rows))
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Invalid checkpoint validation evidence: ' + name) from error
    candidate, best_metrics = summaries
    best = dict(best_metrics, episode=selected, completed_episodes=selected + 1)
    if record.get('validation_selection') != candidate or record.get('best_validation_selection') != best:
        raise ValueError('Checkpoint validation metrics mismatch')
    kind = record.get('checkpoint_kind')
    if kind not in ('best', 'final') or selection_key(best) > selection_key(candidate) or (selected == episode and best_metrics != candidate) or (kind == 'best' and selected != episode):
        raise ValueError('Checkpoint best/final selection mismatch')
    return record

def no_apf_metadata(metadata):
    """Declare the physically legal idle action set for the main algorithm."""
    metadata = dict(metadata)
    metadata.setdefault('guided_probability', metadata.get('apf', {}).get('guided_probability', 0.5))
    metadata.update(host_apf='disabled', vm_apf='disabled', host_apf_enabled=False, vm_apf_enabled=False, performance_action_filter=False, candidate_filter='physical_legality_only', apf=dict(scheme='physical_legality_only_v1', host_enabled=False, vm_enabled=False), host_summary_scope='not_used' if metadata.get('host_summary_scope') == 'not_used' else 'all_physically_legal_idle_vms')
    metadata['components'] = [value for value in metadata.get('components', []) if 'APF' not in value and value != 'PF']
    return metadata
import math
_core_selection_legacy_SELECTION_TAG = 'select_sr092_minenergy_v1'
_core_selection_legacy_WORKFLOW_SUCCESS_THRESHOLD = 0.92

def _core_selection_legacy_selection_metadata():
    return dict(version=_core_selection_legacy_SELECTION_TAG, rule='validation_feasible_then_minimum_energy', workflow_success_threshold=_core_selection_legacy_WORKFLOW_SUCCESS_THRESHOLD, feasibility_metric='total_successful_workflows_divided_by_total_workflows', energy_metric='mean_validation_instance_total_energy', success_metric='total_successful_workflows_divided_by_total_workflows', fallback='maximum_workflow_success_then_minimum_energy', tie_break='higher_workflow_success_then_earlier_checkpoint', population_probability_guaranteed=False)

def _core_selection_legacy_validation_selection(rows):
    if not rows:
        raise ValueError('Checkpoint selection requires validation instances')
    successes = total = 0
    energies = []
    for row in rows:
        count, success = (row['wf_count'], row['wf_success_count'])
        energy = float(row['total_energy'])
        if not math.isfinite(count) or not math.isfinite(success) or count <= 0 or (int(count) != count) or (int(success) != success) or (not 0 <= success <= count) or (not math.isfinite(energy)) or (energy < 0):
            raise ValueError('Invalid validation workflow counts or energy')
        if not math.isclose(float(row['wf_success_rate']), success / count, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError('Validation workflow success rate/count mismatch')
        successes += int(success)
        total += int(count)
        energies.append(energy)
    return dict(workflow_success_rate=successes / total, workflow_success_count=successes, workflow_count=total, mean_total_energy=sum(energies) / len(rows), validation_instances=len(rows), feasible=successes / total >= _core_selection_legacy_WORKFLOW_SUCCESS_THRESHOLD)

def _core_selection_legacy_selection_key(candidate):
    if candidate['feasible']:
        return (0, candidate['mean_total_energy'], -candidate['workflow_success_rate'])
    return (1, -candidate['workflow_success_rate'], candidate['mean_total_energy'])

def _core_selection_legacy_validate_checkpoint_selection(record, expected, row_selector=_core_selection_legacy_validation_selection):
    """Recompute saved evidence before accepting any new-rule checkpoint."""
    if not isinstance(record, dict):
        raise ValueError('Missing MAHDRL-AS-Fast checkpoint selection metadata; use matching historical code for legacy weights')
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('MAHDRL-AS-Fast checkpoint selection mismatch: ' + key)
    episode = record.get('episode')
    selected = record.get('selected_episode')
    if type(episode) is not int or episode < 0 or type(selected) is not int or (not 0 <= selected <= episode) or (record.get('completed_episodes') != episode + 1):
        raise ValueError('Invalid checkpoint selection episode')
    train_seeds = expected['protocol'].get('train')
    if train_seeds is not None and episode >= len(train_seeds):
        raise ValueError('Checkpoint episode outside training protocol')
    summaries = []
    for name in ('validation_rows', 'best_validation_rows'):
        rows = record.get(name)
        if not isinstance(rows, list) or any((not isinstance(row, dict) for row in rows)) or [row.get('seed') for row in rows] != expected['validation_seeds']:
            raise ValueError('Checkpoint validation seeds mismatch: ' + name)
        try:
            summaries.append(row_selector(rows))
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Invalid checkpoint validation evidence: ' + name) from error
    candidate, best_metrics = summaries
    best = dict(best_metrics, episode=selected, completed_episodes=selected + 1)
    if record.get('validation_selection') != candidate or record.get('best_validation_selection') != best:
        raise ValueError('Checkpoint validation metrics mismatch')
    kind = record.get('checkpoint_kind')
    if kind not in ('best', 'final') or _core_selection_legacy_selection_key(best) > _core_selection_legacy_selection_key(candidate) or (selected == episode and best_metrics != candidate) or (kind == 'best' and selected != episode):
        raise ValueError('Checkpoint best/final selection mismatch')
    return record
_core_fast_config_ALGORITHM = 'mahdrl_oe_ge_apf_score_fast'
_core_fast_config_DISPLAY_NAME = 'MAHDRL-AS-Fast-NoAPF'
_core_fast_config_MODEL_VERSION = 'mahdrl_as_fast_noapf_v3'

def _core_fast_config_variant_metadata():
    return no_apf_metadata(dict(_core_set_config_variant_metadata(), algorithm=_core_fast_config_ALGORITHM, scheme=_core_fast_config_MODEL_VERSION, model_version=_core_fast_config_MODEL_VERSION))

def _core_fast_config_implementation_metadata():
    return dict(algorithm=_core_fast_config_ALGORITHM, implementation_version=_core_fast_config_MODEL_VERSION, checkpoint_selection=_core_selection_legacy_selection_metadata(), scientific_variant=_core_fast_config_variant_metadata(), replay='validate_on_insert_owned_readonly_arrays_grouped_numpy_pack', environment='one_decision_cache_raw_legal_noapf_float64', diagnostics='sampled_details_full_update_counts_and_nonfinite_guards')
from dataclasses import asdict, dataclass
_core_mechanism_ALGORITHM = 'mahdrl_as_fast_vaaware'
_core_mechanism_DISPLAY_NAME = 'MAHDRL-AS-Fast-VAware'
_core_mechanism_MODEL_VERSION = 'mahdrl_as_fast_vaaware_noapf_v1'
_core_mechanism_OUTPUT_TAG = 'mahdrl_vaaware'
_core_mechanism_CHECKPOINT_METADATA_KEY = 'mahdrl_vaaware'

@dataclass(frozen=True)
class Mechanism:
    va_preview: bool = True
    ha_update_period: int = 4

    def __post_init__(self):
        if self.va_preview is not True or self.ha_update_period != 4:
            raise ValueError('The main experiment requires VA preview and HA update period four')

    @property
    def host_features(self):
        return 14

def _core_mechanism_variant_metadata(mechanism=Mechanism()):
    metadata = _core_fast_config_variant_metadata()
    metadata.update(algorithm=_core_mechanism_ALGORITHM, scheme=_core_mechanism_MODEL_VERSION, model_version=_core_mechanism_MODEL_VERSION, base_implementation='mahdrl_as_fast_noapf_v3', mechanism=asdict(mechanism), observation_layout=f'host23x{mechanism.host_features}_vm30x7_vaaware_v1', learned_roles=['TA', 'HA', 'VA'], va_preview=dict(version='behavior_expectation_v1', network='online', fields=['expected_duration', 'expected_marginal_energy', 'expected_delay_penalty', 'expected_Q'], duration_scale='source_T_ref', energy_scale='100_times_source_T_ref', delay_scale=300.0, q_scale='R_ref_divided_by_1_minus_gamma', R_ref='0.25*0.001*100*T_ref+0.75', evaluation_epsilon=0.0), update_schedule='lower_updates_after_assignment_TA_at_phase_v1', ha_epsilon_clock='eligible_real_transitions', target_clock='actual_optimizer_steps')
    metadata['components'] += ['VA_behavior_preview']
    metadata['components'] += ['HA_low_frequency_updates']
    return metadata

def _core_mechanism_implementation_metadata(mechanism=Mechanism()):
    return dict(algorithm=_core_mechanism_ALGORITHM, implementation_version=_core_mechanism_MODEL_VERSION, scientific_variant=_core_mechanism_variant_metadata(mechanism), replay='frozen_historical_augmented_observations_fast_pack', environment='raw_legal_noapf_float64', preview='batched_no_grad_no_rng', checkpoint_selection='select_sr092_minenergy_v1')
_core_config_ALGORITHM = 'mahdrl_as_fast_hscore'
_core_config_DISPLAY_NAME = 'MAHDRL-AS-Fast-VAware-HScore-FixedVA'
_core_config_MODEL_VERSION = 'mahdrl_hscore_fixedva_srgt092_v2'
_core_config_OUTPUT_TAG = 'mahdrl_hscore'
CHECKPOINT_METADATA_KEY = 'mahdrl_hscore'
VM_CAPACITY = 9
VM_CONTEXT_DIM = 30
VM_FEATURE_DIM = 7
VM_INPUT_DIM = VM_CONTEXT_DIM + VM_CAPACITY * VM_FEATURE_DIM

def _core_config_variant_metadata(mechanism=Mechanism()):
    metadata = _core_mechanism_variant_metadata(mechanism)
    metadata.update(algorithm=_core_config_ALGORITHM, scheme=_core_config_MODEL_VERSION, model_version=_core_config_MODEL_VERSION, base_implementation='mahdrl_as_fast_vaaware_noapf_v1', architecture='ha_shared_scoring_va_fixed_slot_dueling_v1', observation_layout=f'host23x{mechanism.host_features}_vm93_fixed9_v1', pooling='HA_exists_mean_max_counts_VA_none', fixed_va=dict(input_dim=VM_INPUT_DIM, output_dim=VM_CAPACITY, context_dim=VM_CONTEXT_DIM, slot_features=VM_FEATURE_DIM, capacity=VM_CAPACITY, padding='zero_right_pad_masked', slot_order='existing_host_to_vm_indices_no_reordering', dueling_center='legal_actions_only', shared_across_hosts=True, host_identity='physical_features_without_host_one_hot'))
    metadata['components'] = [c for c in metadata['components'] if c != 'shared_scoring']
    metadata['components'] += ['HA_shared_scoring', 'VA_fixed_slot_D3QN']
    return metadata

def _core_config_implementation_metadata(mechanism=Mechanism()):
    return dict(algorithm=_core_config_ALGORITHM, implementation_version=_core_config_MODEL_VERSION, scientific_variant=_core_config_variant_metadata(mechanism), replay='HA_frozen_set_VA_frozen_93_vector_9_mask', environment='raw_legal_noapf_float64_fixed9_adapter', preview='batched_fixed93_no_grad_no_rng_cached9', checkpoint_selection='select_srgt092_minenergy_v1')
_accel_config_ALGORITHM = 'mahdrl_backend'
_accel_config_DISPLAY_NAME = 'MAHDRL'
_accel_config_MODEL_VERSION = 'mahdrl_backend_v1'
_accel_config_OUTPUT_TAG = 'mahdrl_backend'

def _accel_config_variant_metadata(mechanism):
    return dict(_core_config_variant_metadata(mechanism), algorithm=_accel_config_ALGORITHM, scheme=_accel_config_MODEL_VERSION, model_version=_accel_config_MODEL_VERSION, base_implementation=_core_config_MODEL_VERSION)

def _accel_config_implementation_metadata(mechanism):
    return dict(_core_config_implementation_metadata(mechanism), algorithm=_accel_config_ALGORITHM, implementation_version=_accel_config_MODEL_VERSION, scientific_variant=_accel_config_variant_metadata(mechanism), replay='chronological_indexed_ring_same_python_random_sample', va_batch='vectorized_validation_cpu_nonterminal_indices')
ALGORITHM = 'mahdrl'
DISPLAY_NAME = 'MAHDRL'
MODEL_VERSION = 'mahdrl_v1'
OUTPUT_TAG = 'mahdrl'

def variant_metadata(mechanism):
    return dict(_accel_config_variant_metadata(mechanism), algorithm=ALGORITHM, scheme=MODEL_VERSION, model_version=MODEL_VERSION, base_implementation=_accel_config_MODEL_VERSION)

def implementation_metadata(mechanism):
    return dict(_accel_config_implementation_metadata(mechanism), algorithm=ALGORITHM, implementation_version=MODEL_VERSION, scientific_variant=variant_metadata(mechanism), environment='raw_legal_fixed9_settled_energy_prefix_ordered_float64_cumsum_direct_normalized_queue_v1')
