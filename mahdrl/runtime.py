"""Main-experiment construction, training, checkpoint loading, and frozen evaluation."""
from mahdrl.agents import _FixedVMLearner
from mahdrl.agents import _HostLearner
from mahdrl.agents import _TaskLearner
from mahdrl.config import TA_HIDDEN
from mahdrl.config import learning_kwargs
from mahdrl.config import selection_key
from mahdrl.config import selection_metadata
from mahdrl.config import validate_checkpoint_selection
from mahdrl.config import validation_selection
from mahdrl.environment import _FixedSlotEnvironment
from mahdrl.learning import DecisionTrace
from mahdrl.learning import F11Logger
from mahdrl.learning import UpdateClock
from mahdrl.learning import _core_reward_LowerRewardAudit
from mahdrl.learning import _core_reward_finish_lower_phase
from mahdrl.learning import protect_training_outputs
from mahdrl.normalization import normalization_diagnostics
from mahdrl.protocol import check_cache
from mahdrl.protocol import episode_identity
from mahdrl.protocol import write_run_metadata
from mahdrl.recording import EvaluationRecorder
from mahdrl.recording import TrainingRecorder
from mahdrl.recording import collect_validation
from mahdrl.recording import episode_metrics
from mahdrl.replay import prepare_decision
from mahdrl.simulator import MANAGER_ACTION_TABLE
import json
import csv
from pathlib import Path
import random
import uuid
from copy import deepcopy
import numpy as np
import torch
FILES = {'TA': 'manager', 'HA': 'host', 'VA': 'vm'}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_env_kwargs(scene, seed):
    return dict(dax_paths=scene.dax_paths(), horizon=1000000000.0, arrival_lambda=scene.arrival_lambda, random_seed=int(seed), max_ready_tasks='auto', normalize=True, workflows_per_episode=50, num_hosts=scene.num_hosts, vms_per_host=scene.vms_per_host, pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0), bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0), deadline_mode='cache_fcfs', deadline_cache_path=str(scene.cache_path), deadline_cache_strict=True, deadline_alpha_small=scene.deadline_choices[0], deadline_alpha_large=scene.deadline_choices[1], deadline_alpha_small_prob=scene.small_probability, manager_alpha_delay=0.75, manager_delay_mode='tardiness', **scene.normalization_kwargs(), **scene.arrival_kwargs())

def _core_runtime_build_env(scene, seed):
    env = _FixedSlotEnvironment(**build_env_kwargs(scene, seed), mechanism=scene.mechanism, experiment_metadata=scene.variant_metadata(), profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    env.energy_reward_scale = 0.001
    env.task_baseline_norm = 300.0
    env.energy_norm_per_mi_ref = 20.0
    env.alpha_delay_host = env.alpha_delay_vm = 0.75
    return env

def validate_inputs(scene):
    scene.validate_calibration()
    reference = scene.deadline_reference_scenario()
    if scene.protocol():
        check_cache(reference, path=scene.cache_path)
    else:
        obj = json.loads(scene.cache_path.read_text(encoding='utf-8'))
        expected = dict(num_hosts=reference.num_hosts, vms_per_host=list(reference.vms_per_host), task=reference.task, resource=reference.resource)
        for key, value in expected.items():
            if obj.get('meta', {}).get(key) != value:
                raise ValueError('Deadline cache configuration mismatch: ' + key)

def _core_runtime_build_agents(scene):
    device = torch.device(scene.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    options = dict(diagnostics_every=scene.diagnostics_every, profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    agents = {'TA': _TaskLearner(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN, **options, **learning_kwargs('TA', device)), 'HA': _HostLearner(23, scene.mechanism.host_features, **options, **learning_kwargs('HA', device)), 'VA': _FixedVMLearner(**options, **learning_kwargs('VA', device))}
    for role, agent in agents.items():
        agent.bind(scene, role)
    return agents

def save_agents(agents, directory, kind, *, episode, candidate, best, rows, best_rows):
    records = {role: dict(deepcopy(agent.selection_expectation), checkpoint_kind=kind, episode=episode, completed_episodes=episode + 1, selected_episode=best['episode'], validation_selection=deepcopy(candidate), best_validation_selection=deepcopy(best), validation_rows=deepcopy(rows), best_validation_rows=deepcopy(best_rows)) for role, agent in agents.items()}
    for role, agent in agents.items():
        validate_checkpoint_selection(records[role], agent.selection_expectation)
    group = uuid.uuid4().hex
    for role, agent in agents.items():
        agent.checkpoint_group = group
        agent.normalization_metadata[agent.checkpoint_key] = records[role]
        name = f'best_{FILES[role]}.pth' if kind == 'best' else f'{FILES[role]}_{kind}.pth'
        agent.save(str(Path(directory) / name))

def load_agents(scene, *, agent_builder=None):
    agents = (agent_builder or _core_runtime_build_agents)(scene)
    for role, agent in agents.items():
        agent.load(str(scene.checkpoint_dir() / f'best_{FILES[role]}.pth'), require_learning_model=True)
    if len({a.loaded_group for a in agents.values()}) != 1:
        raise ValueError('TA/HA/VA checkpoints are from different saved groups')
    records = [{key: value for key, value in a.loaded_selection.items() if key != 'learner_role'} for a in agents.values()]
    if any((record['checkpoint_kind'] != 'best' for record in records)):
        raise ValueError('Evaluation requires best checkpoints')
    if any((record != records[0] for record in records[1:])):
        raise ValueError('TA/HA/VA checkpoints have different selection evidence or episodes')
    return agents

def run_episode(env, agents, *, training=False, warmup=0.03, phase_log=None, assignment_log=None):
    if set(agents) != set(FILES) or any((a.mechanism != env.mechanism for a in agents.values())):
        raise ValueError('Mismatched HScore roles or mechanisms')
    clock = UpdateClock(agents, training=training, warmup=warmup)
    traces = {r: DecisionTrace(clock, r) for r in ('HA', 'VA')}
    audit = _core_reward_LowerRewardAudit(traces['HA'], traces['VA'])
    phases, returns, diagnostics = (0, dict(TA=0.0, HA=0.0, VA=0.0), [])
    state, mask = (env.get_manager_state(), env.get_manager_action_mask())
    action = agents['TA'].select_action(state, mask, deterministic=not training, count_step=training)
    env.apply_manager_delta(MANAGER_ACTION_TABLE[action])
    while not env.done_flag:
        rewards = {'HA': [], 'VA': []}
        while True:
            host, ok = env.get_host_state_for_next_assignment()
            if not ok:
                break
            host, previews, extra = prepare_decision(env, agents['VA'], host, training=training)
            traces['HA'].advance(host['obs'], host['mask'])
            ah = agents['HA'].select_action(host['obs'], host['mask'], deterministic=not training, count_step=training, performance_times=host['performance_times'])
            hid = env.host_ids[ah]
            env.host_select(ah)
            vm = previews[hid]
            traces['VA'].advance(vm['obs'], vm['mask'])
            av = agents['VA'].select_preview_action(vm, training=training)
            expected_energy = float(vm['probabilities'] @ vm['rows'][:, 1])
            selected = vm['rows'][av]
            legal = vm['mask'] > 0.5
            ontime = legal & (vm['rows'][:, 2] <= 1e-09)
            all_ontime = [s['rows'][i, 1] for s in previews.values() for i in np.flatnonzero((s['mask'] > 0.5) & (s['rows'][:, 2] <= 1e-09))]
            host_gap = vm_gap = None
            if ontime[av]:
                local_min = float(vm['rows'][ontime, 1].min())
                host_gap = local_min - float(min(all_ontime))
                vm_gap = float(selected[1]) - local_min
            greedy = int(np.flatnonzero(legal)[np.argmax(vm['q'][legal])])
            rh, rv, info = env.vm_assign(av)
            audit.assignment(info)
            rewards['HA'].append(float(rh))
            rewards['VA'].append(float(rv))
            traces['HA'].record(host['obs'], host['mask'], ah, rh, host_index=hid)
            traces['VA'].record(vm['obs'], vm['mask'], av, rv, host_index=hid)
            row = dict(task_id=int(info['task_id']), host_id=int(hid), vm_id=int(env.vm_ids[info['vm_global_idx']]), va_version=int(vm['va_version']), va_epsilon=float(vm['epsilon']), legal_host_count=int(host['mask'].sum()), legal_vm_count=int(vm['mask'].sum()), global_legal_vm_count=int(sum((s['mask'].sum() for s in previews.values()))), preview_features=extra[ah].tolist(), expected_assignment_energy=expected_energy, assignment_energy=float(info['assignment_energy']), energy_prediction_error=float(info['assignment_energy']) - expected_energy, greedy_match=bool(av == greedy), host_selection_gap=host_gap, vm_selection_gap=vm_gap)
            diagnostics.append(row)
            if assignment_log:
                assignment_log(row)
            clock.barrier()
        rm, info = env.finish_phase_and_advance()
        audit.phase(info, traces['VA'])
        _core_reward_finish_lower_phase(env, traces['HA'], traces['VA'], info)
        unreturned = float(info.get('lower_assignment_energy_unemitted', 0.0))
        for role, alpha in (('HA', env.alpha_delay_host), ('VA', env.alpha_delay_vm)):
            if unreturned:
                rewards[role].append(-(1 - alpha) * unreturned * env.energy_reward_scale)
        next_state, next_mask = (env.get_manager_state(), env.get_manager_action_mask())
        if training and info.get('assign_cnt', 0) > 0:
            agents['TA'].f11.truncation_cutoffs += int(info.get('truncated', False))
            clock.remember('TA', state, mask, action, rm, next_state, next_mask, float(env.done_flag))
        clock.barrier()
        returns['TA'] += float(rm)
        for role in rewards:
            returns[role] += float(np.mean(rewards[role])) if rewards[role] else 0.0
        phases += 1
        if phase_log:
            phase_log(dict(phase=phases, manager_reward=float(rm), **info))
        state, mask = (next_state, next_mask)
        if not env.done_flag:
            action = agents['TA'].select_action(state, mask, deterministic=not training, count_step=training)
            env.apply_manager_delta(MANAGER_ACTION_TABLE[action])
    for trace in traces.values():
        trace.finish(truncated=not env._episode_should_end())
    clock.barrier()
    mean = lambda key: float(np.mean([r[key] for r in diagnostics if r[key] is not None])) if any((r[key] is not None for r in diagnostics)) else None
    return dict(phase_count=phases, rewards={r: v / max(1, phases) for r, v in returns.items()}, reward_audit=audit.summary(env, traces['HA'], traces['VA']), update_clock=clock.summary(), vaaware_diagnostics=dict(assignments=len(diagnostics), mean_host_selection_gap=mean('host_selection_gap'), mean_global_legal_vm_count=mean('global_legal_vm_count'), mean_local_legal_vm_count=mean('legal_vm_count'), mean_legal_host_count=mean('legal_host_count'), mean_vm_selection_gap=mean('vm_selection_gap'), greedy_match_rate=mean('greedy_match'), mean_energy_prediction_error=mean('energy_prediction_error')), **episode_metrics(env))

def apf_report(env):
    windows = {}
    for r in env._records:
        if r.server_id not in windows:
            windows[r.server_id] = [float(r.start_time), float(r.end_time)]
        else:
            windows[r.server_id][0] = min(windows[r.server_id][0], float(r.start_time))
            windows[r.server_id][1] = max(windows[r.server_id][1], float(r.end_time))
    return dict(variant=env.variant_metadata(), decisions=env.apf_decisions, host_windows={str(h): dict(start=w[0], end=w[1], duration=w[1] - w[0]) for h, w in windows.items()})

def _core_runtime_train(scene, *, env_builder=None, agent_builder=None, rollout=None):
    make_env, make_agents, episode_runner = (env_builder or _core_runtime_build_env, agent_builder or _core_runtime_build_agents, rollout or run_episode)
    protect_training_outputs(scene.checkpoint_dir(), scene.log_dir())
    validate_inputs(scene)
    set_seed(scene.model_seed)
    env, agents = (make_env(scene, scene.training_seed(0)), make_agents(scene))
    device = next(iter(agents.values())).device
    env.reset()
    scene.checkpoint_dir().mkdir(parents=True)
    scene.log_dir().mkdir(parents=True)
    for directory in (scene.checkpoint_dir(), scene.log_dir()):
        write_run_metadata(directory, scene)
        write_json(directory / 'algorithm_variant.json', scene.variant_metadata())
        write_json(directory / 'implementation.json', scene.implementation_metadata())
    recorder = TrainingRecorder(scene.log_dir(), scene, scene.algorithm, env, agents)
    f11 = F11Logger(scene.log_dir())
    best, best_rows = (None, None)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for episode in range(scene.episodes):
            if episode:
                env.reset(seed=scene.training_seed(episode))
            result = episode_runner(env, agents, training=True, phase_log=lambda row: append_json(scene.log_dir() / 'phases.jsonl', dict(episode=episode, **row)), assignment_log=lambda row: append_json(scene.log_dir() / 'assignments.jsonl', dict(episode=episode, **row)))
            append_json(scene.log_dir() / 'stage_timings.jsonl', dict(episode=episode, scope='training', environment=env.profiler.report(), learners={r: a.profiler.report() for r, a in agents.items()}))
            f11.log(episode, agents, normalization=normalization_diagnostics(env), instance=episode_identity(env, scene), reward_audit=result['reward_audit'], pending=0)

            def validate():
                values = []
                for seed in scene.validation_seeds():
                    validation = make_env(scene, seed)
                    validation.reset()
                    result = episode_runner(validation, agents)
                    if result['wf_count'] != validation.workflows_per_episode:
                        raise RuntimeError('Validation ended before all workflows arrived')
                    values.append(dict(seed=int(seed), **{key: result[key] for key in ('total_energy', 'wf_count', 'wf_success_count', 'wf_success_rate')}))
                    append_json(scene.log_dir() / 'stage_timings.jsonl', dict(episode=episode, scope='validation', seed=seed, environment=validation.profiler.report()))
                    collect_validation(validation)
                return values
            rows = recorder.validation(validate)
            candidate = validation_selection(rows)
            selected = best is None or selection_key(candidate) < selection_key(best)
            if selected:
                best = dict(candidate, episode=episode, completed_episodes=episode + 1)
                best_rows = deepcopy(rows)
                save_agents(agents, scene.checkpoint_dir(), 'best', episode=episode, candidate=candidate, best=best, rows=rows, best_rows=best_rows)
            energy = candidate['mean_total_energy']
            append_json(scene.log_dir() / 'episodes.jsonl', dict(episode=episode, instance_seed=scene.training_seed(episode), validation_energy=energy, validation_selection=candidate, selected_as_best=selected, best_episode=best['episode'], **result))
            append_json(scene.log_dir() / 'validation_selection.jsonl', dict(algorithm=scene.algorithm, model_version=scene.model_version, checkpoint_selection=selection_metadata(), episode=episode, instance_seed=scene.training_seed(episode), validation_rows=rows, validation_selection=candidate, best_validation_selection=best, selected_as_best=selected, selected_episode=best['episode']))
            write_json(scene.log_dir() / f'candidates_episode{episode}.json', apf_report(env))
            print(f"[{scene.display_name}] episode={episode + 1}/{scene.episodes} energy={result['total_energy']:.6f} validation_energy={energy:.6f} SR={candidate['workflow_success_rate']:.4f} feasible={candidate['feasible']} selected={selected} best_episode={best['episode']}", flush=True)
        save_agents(agents, scene.checkpoint_dir(), 'final', episode=episode, candidate=candidate, best=best, rows=rows, best_rows=best_rows)
        write_json(scene.log_dir() / 'training_summary.json', dict(algorithm=scene.algorithm, model_version=scene.model_version, variant=scene.variant_metadata(), episodes=scene.episodes, checkpoint_selection=selection_metadata(), protocol=scene.selection_protocol(), selected_episode=best['episode'], best_validation_selection=best, final_validation_selection=candidate, feasible_model_found=best['feasible'], resume_supported=False, best_validation_energy=best['mean_total_energy'], peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None, parameter_counts={r: sum((p.numel() for p in a.online.parameters())) for r, a in agents.items()}))
    finally:
        recorder.close()
    print(f'[{scene.display_name} train completed] models={scene.checkpoint_dir().resolve()} logs={scene.log_dir().resolve()}')
    return agents

def _core_runtime_evaluate(scene, *, env_builder=None, agent_builder=None, rollout=None):
    make_env, episode_runner = (env_builder or _core_runtime_build_env, rollout or run_episode)
    directory = scene.evaluation_dir()
    csv_path = scene.evaluation_csv()
    if csv_path.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation CSV: {csv_path}')
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation artifacts: {directory}')
    validate_inputs(scene)
    set_seed(scene.model_seed)
    agents = load_agents(scene, agent_builder=agent_builder)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'algorithm_variant.json', scene.variant_metadata())
    write_json(directory / 'implementation.json', scene.implementation_metadata())
    write_run_metadata(directory, scene)
    values = []
    for seed in scene.seeds:
        set_seed(scene.model_seed)
        env = make_env(scene, seed)
        env.reset()
        recorder = EvaluationRecorder(scene, scene.algorithm, env, agents, directory=directory / 'measurements', checkpoint_directory=str(scene.checkpoint_dir()))
        try:
            result = episode_runner(env, agents, assignment_log=lambda row: append_json(directory / 'assignments.jsonl', dict(seed=seed, **row)))
        finally:
            recorder.finish()
        append_json(directory / 'stage_timings.jsonl', dict(seed=seed, scope='evaluation', environment=env.profiler.report()))
        write_json(directory / f'candidates_seed{seed}.json', apf_report(env))
        append_json(directory / 'evaluation.jsonl', dict(seed=seed, **result))
        row = dict(**scene.evaluation_identity(), seed=seed, model_seed=scene.model_seed, model_version=scene.model_version, va_preview=scene.va_preview, ha_update_period=scene.ha_update_period, host_apf_enabled=False, vm_apf_enabled=False, **episode_metrics(env))
        with csv_path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        values.append(result)
    summary = {key: dict(mean=float(np.mean([v[key] for v in values])), std=float(np.std([v[key] for v in values], ddof=1)) if len(values) > 1 else 0.0) for key in ('total_energy', 'wf_success_rate', 'wf_avg_tardiness', 'makespan', 'wf_avg_response_time')}
    write_json(directory / 'summary.json', dict(algorithm=scene.algorithm, model_version=scene.model_version, variant=scene.variant_metadata(), checkpoint_selection=selection_metadata(), selected_episode=next(iter(agents.values())).loaded_selection['selected_episode'], best_validation_selection=next(iter(agents.values())).loaded_selection['best_validation_selection'], seeds=list(scene.seeds), metrics=summary, provenance=scene.reproducibility_metadata()))
    print(f'[{scene.display_name} eval completed] csv={csv_path.resolve()} details={directory.resolve()}')
    return summary
from mahdrl.agents import FixedVAAgent
from mahdrl.agents import HAAgent
from mahdrl.agents import TAAgent
from mahdrl.config import TA_HIDDEN
from mahdrl.config import learning_kwargs
from mahdrl.simulator import MANAGER_ACTION_TABLE
import torch

def build_agents(scene):
    device = torch.device(scene.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    options = dict(diagnostics_every=scene.diagnostics_every, profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    agents = {'TA': TAAgent(15, len(MANAGER_ACTION_TABLE), hidden_dims=TA_HIDDEN, **options, **learning_kwargs('TA', device)), 'HA': HAAgent(23, scene.mechanism.host_features, **options, **learning_kwargs('HA', device)), 'VA': FixedVAAgent(**options, **learning_kwargs('VA', device))}
    return {role: agent.bind(scene, role) for role, agent in agents.items()}
from mahdrl.environment import CloudWorkflowEnv_VMAgents

def build_env(scene, seed):
    env = CloudWorkflowEnv_VMAgents(**build_env_kwargs(scene, seed), mechanism=scene.mechanism, experiment_metadata=scene.variant_metadata(), profile_mode=scene.profile_mode, profile_every=scene.profile_every)
    env.energy_reward_scale = 0.001
    env.task_baseline_norm = 300.0
    env.energy_norm_per_mi_ref = 20.0
    env.alpha_delay_host = env.alpha_delay_vm = 0.75
    return env

def train(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    return _core_runtime_train(scene, **kwargs)

def evaluate(scene, **kwargs):
    kwargs.setdefault('env_builder', build_env)
    kwargs.setdefault('agent_builder', build_agents)
    return _core_runtime_evaluate(scene, **kwargs)
