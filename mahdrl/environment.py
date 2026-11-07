"""The main MAHDRL environment with legal actions and settled-prefix energy accounting."""
from mahdrl.config import APF
from mahdrl.config import _core_set_config_ALGORITHM
from mahdrl.config import _core_set_config_variant_metadata
from mahdrl.simulator import POWER_REF_W
from mahdrl.simulator import _EnergySimulator
from mahdrl.simulator import _WorkflowSimulator
from mahdrl.simulator import spaces
import numpy as np

class VariableArraySpace(spaces.Space):
    """Variable first dimension, compatible with both installed Gym and Gymnasium."""

    def __init__(self, tail=(), binary=False):
        super().__init__(shape=None, dtype=np.float32)
        self.tail, self.binary = (tuple(tail), binary)

    def sample(self, mask=None):
        return np.zeros((0, *self.tail), np.float32)

    def contains(self, x):
        return isinstance(x, np.ndarray) and x.dtype == np.float32 and (x.ndim == len(self.tail) + 1) and (x.shape[1:] == self.tail) and np.isfinite(x).all() and (not self.binary or np.isin(x, (0, 1)).all())

def _candidates_apf_candidates(rows, host_rows, ready_count, now, deadline):
    """rows: global VM index -> (completion delay, marginal joules, tardiness)."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0.0, u=0.0, r_H=1.25, r_V=1.25, energy_hosts=set(), energy_vms={}, raw_count=0)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError('APF needs a current Ready task and finite times')
    if not all((np.isfinite(row).all() and row[0] >= 0 for row in rows.values())):
        raise ValueError('Non-finite or negative APF duration')
    g = len(rows) / (len(rows) + ready_count)
    fastest = min((row[0] for row in rows.values()))
    u = float(np.clip(fastest / max(deadline - now, APF.time_atol), 0.0, 1.0))
    rh = float(np.clip(APF.host_base + APF.spare_weight * g - APF.urgency_weight * u, APF.lower, APF.upper))
    rv = float(np.clip(APF.vm_base + APF.spare_weight * g - APF.urgency_weight * u, APF.lower, APF.upper))

    def energy_choices(indices):
        ontime = [j for j in indices if now + rows[j][0] <= deadline + APF.time_atol]
        if not ontime:
            return set()
        minimum = min((rows[j][1] for j in ontime))
        return {j for j in ontime if np.isclose(rows[j][1], minimum, rtol=APF.energy_rtol, atol=APF.energy_atol)}
    global_energy = energy_choices(rows)
    hosts, kept, energy_vms, energy_hosts = (set(), {}, {}, set())
    for hid, indices in host_rows.items():
        raw = set(indices).intersection(rows)
        if not raw:
            continue
        th = min((rows[j][0] for j in raw))
        energy_vms[hid] = energy_choices(raw)
        kept[hid] = {j for j in raw if rows[j][0] <= rv * th + APF.time_atol} | energy_vms[hid]
        if raw.intersection(global_energy):
            energy_hosts.add(hid)
        if th <= rh * fastest + APF.time_atol:
            hosts.add(hid)
    return dict(hosts=hosts | energy_hosts, vms=kept, g=g, u=u, r_H=rh, r_V=rv, energy_hosts=energy_hosts, energy_vms=energy_vms, raw_count=len(rows))

class _CandidateEnvironment(_EnergySimulator):
    algorithm = _core_set_config_ALGORITHM

    def __init__(self, *args, **kwargs):
        self._apf_cache = None
        super().__init__(*args, **kwargs)

        def obs_space(c, f):
            return spaces.Dict({'context': spaces.Box(-np.inf, np.inf, (c,), np.float32), 'candidates': VariableArraySpace((f,)), 'exists_mask': VariableArraySpace(binary=True)})
        self.observation_space = spaces.Dict({'host_obs': obs_space(23, 10), 'host_mask': VariableArraySpace(binary=True), 'vm_obs': obs_space(30, 7), 'vm_mask': VariableArraySpace(binary=True)})
        self.apf_decisions = []

    def variant_metadata(self):
        return dict(super().variant_metadata(), **_core_set_config_variant_metadata())

    @staticmethod
    def _empty(c, f):
        return {'obs': {'context': np.zeros(c, np.float32), 'candidates': np.zeros((0, f), np.float32), 'exists_mask': np.zeros(0, np.float32)}, 'mask': np.zeros(0, np.float32), 'candidate_ids': [], 'performance_times': np.zeros(0, np.float64)}

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self._apf_cache = None
        self.apf_decisions = []
        h, v = (self._empty(23, 10), self._empty(30, 7))
        return (dict(host_obs=h['obs'], host_mask=h['mask'], vm_obs=v['obs'], vm_mask=v['mask']), info)

    def _apf_snapshot(self, tid):
        key = (int(tid), float(self.current_time), len(self._records), float(self.task_baseline_finish[tid]), tuple(self.vm_available_at), tuple(self.ready_task_ids))
        if self._apf_cache is not None and self._apf_cache[0] == key:
            return self._apf_cache[1]
        raw = _EnergySimulator._candidate_rows(self, tid)
        rows = {}
        for j, (_, energy, _) in raw.items():
            if self.vm_available_at[j] <= self.current_time + 1e-12:
                duration = max(0.0, self.vm_available_at[j] - self.current_time) + self._task_duration(tid, j)
                rows[j] = (float(duration), energy, max(0.0, self.current_time + duration - self.task_baseline_finish[tid]))
        ready = len({j for j in self.ready_task_ids if self.task_state[j] == 'Ready'} | {int(tid)})
        result = _candidates_apf_candidates(rows, self.host_to_vm_indices, ready, self.current_time, self.task_baseline_finish[tid])
        result['rows'] = rows
        self._apf_cache = (key, result)
        return result

    def _candidate_rows(self, tid):
        s = self._apf_snapshot(tid)
        return {j: s['rows'][j] for h in s['hosts'] for j in s['vms'][h]}

    def _build_host_obs_for_task(self, tid):
        original, original_mask = _WorkflowSimulator._build_host_obs_for_task(self, tid)
        s = self._apf_snapshot(tid)
        time_ref = self._obs_ref('T_ref', self.task_baseline_norm)
        summary = np.zeros((self.num_hosts, 5), np.float32)
        mask = np.zeros_like(original_mask)
        for hi, hid in enumerate(self.host_ids):
            if hid not in s['hosts'] or original_mask[hi] <= 0.5:
                continue
            mask[hi] = 1.0
            values = [s['rows'][j] for j in s['vms'][hid]]
            fast_t, cheap_e = (min((x[0] for x in values)), min((x[1] for x in values)))
            fast_e = min((x[1] for x in values if abs(x[0] - fast_t) <= APF.time_atol))
            cheap_t = min((x[0] for x in values if np.isclose(x[1], cheap_e, rtol=APF.energy_rtol, atol=APF.energy_atol)))
            summary[hi] = [fast_t / time_ref, fast_e / (POWER_REF_W * time_ref), min((x[2] for x in values)) / time_ref, cheap_e / (POWER_REF_W * time_ref), cheap_t / time_ref]
        obs = np.concatenate((original, [(self.task_baseline_finish[tid] - self.current_time) / time_ref], summary.ravel())).astype(np.float32)
        return (obs, mask)

    def _build_vm_obs_for_task_host(self, tid, host_id):
        obs, original_mask = _EnergySimulator._build_vm_obs_for_task_host(self, tid, host_id)
        s = self._apf_snapshot(tid)
        mask = np.zeros_like(original_mask)
        if host_id in s['hosts']:
            for slot, j in enumerate(self.host_to_vm_indices[host_id]):
                mask[slot] = float(original_mask[slot] > 0.5 and j in s['vms'][host_id])
        start = self.base_vm_obs_dim + 1 + self.num_hosts + 2
        obs[start:].reshape(self.max_vms_per_host, 2)[mask <= 0.5] = 0.0
        return (obs, mask)

    def get_host_state_for_next_assignment(self):
        state, ok = super().get_host_state_for_next_assignment()
        if not ok:
            return (self._empty(23, 10), False)
        x, s = (state['obs'], self._apf_snapshot(self._cur_tid))
        context = np.concatenate((x[:22], x[self.base_host_obs_dim:self.base_host_obs_dim + 1]))
        candidates = np.concatenate((x[22:self.base_host_obs_dim].reshape(self.num_hosts, 5), x[self.base_host_obs_dim + 1:].reshape(self.num_hosts, 5)), 1)
        times = np.array([min((s['rows'][j][0] for j in indices if j in s['rows']), default=np.inf) for indices in (self.host_to_vm_indices[h] for h in self.host_ids)])
        return (dict(obs=dict(context=context, candidates=candidates, exists_mask=np.ones(self.num_hosts, np.float32)), mask=state['mask'], candidate_ids=list(self.host_ids), performance_times=times), True)

    def get_vm_state_for_current_task(self):
        state, ok = super().get_vm_state_for_current_task()
        if not ok:
            return (self._empty(30, 7), False)
        x, b, h = (state['obs'], self.base_vm_obs_dim, self._cur_host_id)
        indices = self.host_to_vm_indices[h]
        k = len(indices)
        context = np.concatenate((x[:27], x[b:b + 1], x[b + 1 + self.num_hosts:b + 3 + self.num_hosts]))
        candidates = np.concatenate((x[27:b].reshape(self.max_vms_per_host, 5)[:k], x[b + 3 + self.num_hosts:].reshape(self.max_vms_per_host, 2)[:k]), 1)
        s = self._apf_snapshot(self._cur_tid)
        return (dict(obs=dict(context=context, candidates=candidates, exists_mask=np.ones(k, np.float32)), mask=state['mask'][:k], candidate_ids=[self.vm_ids[j] for j in indices], performance_times=np.array([s['rows'].get(j, (np.inf,))[0] for j in indices])), True)

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= 0.5:
            raise ValueError('Host excluded by APF')
        return super().host_select(host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = (self._cur_tid, self._cur_host_id)
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= 0.5:
            raise ValueError('VM excluded by APF')
        s = self._apf_snapshot(tid)
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        th = min((s['rows'][i][0] for i in self.host_to_vm_indices[hid] if i in s['rows']))
        speed_hosts = {h for h in s['hosts'] if min((s['rows'][i][0] for i in self.host_to_vm_indices[h] if i in s['rows'])) <= s['r_H'] * min((x[0] for x in s['rows'].values())) + APF.time_atol}
        speed_vms = {i for i in s['vms'][hid] if s['rows'][i][0] <= s['r_V'] * th + APF.time_atol}
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]), g=s['g'], u=s['u'], r_H=s['r_H'], r_V=s['r_V'], raw_vm_count=s['raw_count'], raw_host_count=len(s['vms']), retained_host_count=len(s['hosts']), local_raw_vm_count=sum((i in s['rows'] for i in self.host_to_vm_indices[hid])), local_retained_vm_count=len(s['vms'][hid]), energy_rescued_host_count=len(s['hosts'] - speed_hosts), local_energy_rescued_vm_count=len(s['vms'][hid] - speed_vms), selected_energy_only_host=hid not in speed_hosts, selected_energy_only_vm=j not in speed_vms, selected_time_ratio=s['rows'][j][0] / max(min((x[0] for x in s['rows'].values())), 1e-12), predicted_ontime=bool(self.current_time + s['rows'][j][0] <= self.task_baseline_finish[tid] + APF.time_atol)))
        result = super().vm_assign(vm_slot_index)
        self._apf_cache = None
        return result
from mahdrl.config import APF
from mahdrl.config import _core_fast_config_ALGORITHM
from mahdrl.diagnostics import StageProfiler
from mahdrl.simulator import POWER_REF_W
from mahdrl.simulator import _EnergySimulator
from mahdrl.simulator import _WorkflowSimulator
import numpy as np

def _cache_apf_candidates(rows, host_rows, ready_count, now, deadline):
    """Float64 grouped reductions preserve original legality, tolerances and all ties."""
    if not rows:
        return dict(hosts=set(), vms={}, g=0.0, u=0.0, r_H=1.25, r_V=1.25, energy_hosts=set(), energy_vms={}, raw_count=0)
    values = np.asarray(list(rows.values()), dtype=np.float64)
    if ready_count < 1 or not np.isfinite([now, deadline]).all():
        raise ValueError('APF needs a current Ready task and finite times')
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
        raise ValueError('Non-finite or negative APF duration')
    ids = np.asarray(list(rows), dtype=np.int64)
    hosts = list(host_rows)
    owner = {j: hi for hi, h in enumerate(hosts) for j in host_rows[h]}
    group = np.asarray([owner[j] for j in ids], dtype=np.int64)
    t, energy = (values[:, 0], values[:, 1])
    g = len(rows) / (len(rows) + ready_count)
    fastest = float(t.min())
    u = float(np.clip(fastest / max(deadline - now, APF.time_atol), 0.0, 1.0))
    rh = float(np.clip(APF.host_base + APF.spare_weight * g - APF.urgency_weight * u, APF.lower, APF.upper))
    rv = float(np.clip(APF.vm_base + APF.spare_weight * g - APF.urgency_weight * u, APF.lower, APF.upper))
    minimum = np.full(len(hosts), np.inf)
    np.minimum.at(minimum, group, t)
    ontime = now + t <= deadline + APF.time_atol
    local_energy = np.full(len(hosts), np.inf)
    np.minimum.at(local_energy, group[ontime], energy[ontime])
    energy_vm = ontime & np.isclose(energy, local_energy[group], rtol=APF.energy_rtol, atol=APF.energy_atol)
    global_energy = ontime & np.isclose(energy, energy[ontime].min() if ontime.any() else np.inf, rtol=APF.energy_rtol, atol=APF.energy_atol)
    energy_host = np.zeros(len(hosts), dtype=bool)
    np.logical_or.at(energy_host, group, global_energy)
    present = np.isfinite(minimum)
    speed_host = present & (minimum <= rh * fastest + APF.time_atol)
    speed_vm = t <= rv * minimum[group] + APF.time_atol
    kept = speed_vm | energy_vm
    result = dict(hosts={hosts[i] for i in np.flatnonzero(speed_host | energy_host)}, vms={h: set(ids[(group == i) & kept].tolist()) for i, h in enumerate(hosts) if present[i]}, energy_hosts={hosts[i] for i in np.flatnonzero(energy_host)}, energy_vms={h: set(ids[(group == i) & energy_vm].tolist()) for i, h in enumerate(hosts) if present[i]}, g=g, u=u, r_H=rh, r_V=rv, raw_count=len(rows))
    result['arrays'] = dict(ids=ids, group=group, values=values, minimum=minimum, speed_host=speed_host, speed_vm=speed_vm, kept=kept)
    return result

class _CachedEnvironment(_CandidateEnvironment):
    algorithm = _core_fast_config_ALGORITHM

    def __init__(self, *args, profile_mode='sampled', profile_every=100, **kwargs):
        self.profiler = StageProfiler(profile_mode, profile_every)
        self._decision_index = 0
        self._invalidate_decision()
        super().__init__(*args, **kwargs)

    def _invalidate_decision(self):
        self._fast_key = None
        self._fast_cache = {}
        self._apf_cache = None

    def _ensure_decision(self, tid):
        key = (int(tid), float(self.current_time), len(self._records), float(self.task_baseline_finish[tid]), tuple(self.vm_available_at), tuple(self.ready_task_ids), len(self.task_state))
        if key != self._fast_key:
            self._invalidate_decision()
            self._fast_key = key
            self.profiler.select(self._decision_index)
            self._decision_index += 1

    def reset(self, **kwargs):
        self._invalidate_decision()
        self._decision_index = 0
        result = super().reset(**kwargs)
        self._invalidate_decision()
        return result

    def _compute_block1(self, tid):
        key = ('block1', tid)
        if key not in self._fast_cache:
            self._fast_cache[key] = super()._compute_block1(tid)
        return self._fast_cache[key]

    def _compute_block2(self):
        if 'block2' not in self._fast_cache:
            self._fast_cache['block2'] = super()._compute_block2()
        return self._fast_cache['block2']

    def _compute_host_block(self):
        if 'host_block' not in self._fast_cache:
            self._fast_cache['host_block'] = super()._compute_host_block()
        return self._fast_cache['host_block']

    def _task_duration(self, task_id, vm_index):
        if self._fast_key is None:
            return super()._task_duration(task_id, vm_index)
        key = ('duration', int(task_id), int(vm_index))
        if key not in self._fast_cache:
            self._fast_cache[key] = super()._task_duration(task_id, vm_index)
        return self._fast_cache[key]

    def _apf_snapshot(self, tid):
        self._ensure_decision(tid)
        if self._apf_cache is not None:
            return self._apf_cache
        with self.profiler.stage('apf_preview_and_filter'):
            rows = {}
            for j in np.flatnonzero(self._vm_idle_mask() & (np.asarray(self.vm_available_at) <= self.current_time + 1e-12)):
                j = int(j)
                vm = self.vms[self.vm_ids[j]]
                duration = float(self._task_duration(tid, j))
                start = max(float(self.current_time), float(self.vm_available_at[j]))
                energy = self.assignment_ledger.preview(vm.host_id, start, start + duration, vm.pc)
                delay = max(0.0, self.vm_available_at[j] - self.current_time) + duration
                rows[j] = (float(delay), energy, max(0.0, self.current_time + delay - self.task_baseline_finish[tid]))
            ready = len({j for j in self.ready_task_ids if self.task_state[j] == 'Ready'} | {int(tid)})
            s = _cache_apf_candidates(rows, self.host_to_vm_indices, ready, self.current_time, self.task_baseline_finish[tid])
            s['rows'] = rows
            self._apf_cache = s
        return s

    def _build_host_obs_for_task(self, tid):
        self._ensure_decision(tid)
        if 'host_obs' in self._fast_cache:
            return self._fast_cache['host_obs']
        with self.profiler.stage('host_observation'):
            original, original_mask = _WorkflowSimulator._build_host_obs_for_task(self, tid)
            s = self._apf_snapshot(tid)
            ref = self._obs_ref('T_ref', self.task_baseline_norm)
            summary = np.zeros((self.num_hosts, 5), np.float32)
            mask = np.asarray([h in s['hosts'] for h in self.host_ids], np.float32) * original_mask
            if s['rows']:
                a = s['arrays']
                group, values = (a['group'], a['values'])
                kept = a['kept'] & (mask[group] > 0.5)
                t, e, late = (values[:, 0], values[:, 1], values[:, 2])
                fast_t, cheap_e, min_late = (np.full(self.num_hosts, np.inf) for _ in range(3))
                for dest, data in ((fast_t, t), (cheap_e, e), (min_late, late)):
                    np.minimum.at(dest, group[kept], data[kept])
                fast_e, cheap_t = (np.full(self.num_hosts, np.inf), np.full(self.num_hosts, np.inf))
                fastest = kept & (np.abs(t - fast_t[group]) <= APF.time_atol)
                cheapest = kept & np.isclose(e, cheap_e[group], rtol=APF.energy_rtol, atol=APF.energy_atol)
                np.minimum.at(fast_e, group[fastest], e[fastest])
                np.minimum.at(cheap_t, group[cheapest], t[cheapest])
                legal = mask > 0.5
                summary[legal] = np.column_stack((fast_t / ref, fast_e / (POWER_REF_W * ref), min_late / ref, cheap_e / (POWER_REF_W * ref), cheap_t / ref))[legal]
            obs = np.concatenate((original, [(self.task_baseline_finish[tid] - self.current_time) / ref], summary.ravel())).astype(np.float32)
            obs.flags.writeable = mask.flags.writeable = False
            self._fast_cache['host_obs'] = (obs, mask)
        return (obs, mask)

    def _build_vm_obs_for_task_host(self, tid, host_id):
        self._ensure_decision(tid)
        key = ('vm_obs', host_id)
        if key not in self._fast_cache:
            with self.profiler.stage('vm_observation'):
                obs, original_mask = _EnergySimulator._build_vm_obs_for_task_host(self, tid, host_id)
                s = self._apf_snapshot(tid)
                mask = np.zeros_like(original_mask)
                if host_id in s['hosts']:
                    indices = self.host_to_vm_indices[host_id]
                    mask[:len(indices)] = ((original_mask[:len(indices)] > 0.5) & np.isin(indices, list(s['vms'][host_id]))).astype(np.float32)
                start = self.base_vm_obs_dim + 1 + self.num_hosts + 2
                obs[start:].reshape(self.max_vms_per_host, 2)[mask <= 0.5] = 0.0
                obs.flags.writeable = mask.flags.writeable = False
                self._fast_cache[key] = (obs, mask)
        return self._fast_cache[key]

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= 0.5:
            raise ValueError('Host excluded by APF')
        return _WorkflowSimulator.host_select(self, host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = (self._cur_tid, self._cur_host_id)
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= 0.5:
            raise ValueError('VM excluded by APF')
        s = self._apf_snapshot(tid)
        a = s['arrays']
        hi = self.host_ids.index(hid)
        local = a['group'] == hi
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        chosen = int(np.flatnonzero(a['ids'] == j)[0])
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]), g=s['g'], u=s['u'], r_H=s['r_H'], r_V=s['r_V'], raw_vm_count=s['raw_count'], raw_host_count=len(s['vms']), retained_host_count=len(s['hosts']), local_raw_vm_count=int(local.sum()), local_retained_vm_count=len(s['vms'][hid]), energy_rescued_host_count=len(s['hosts']) - int(a['speed_host'].sum()), local_energy_rescued_vm_count=int((local & a['kept'] & ~a['speed_vm']).sum()), selected_energy_only_host=not bool(a['speed_host'][hi]), selected_energy_only_vm=not bool(a['speed_vm'][chosen]), selected_time_ratio=s['rows'][j][0] / max(float(a['values'][:, 0].min()), 1e-12), predicted_ontime=bool(self.current_time + s['rows'][j][0] <= self.task_baseline_finish[tid] + APF.time_atol)))
        try:
            with self.profiler.stage('environment_assignment'):
                return _EnergySimulator.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        try:
            return super()._assign_task_to_specific_vm(task_id, vm_index)
        finally:
            self._invalidate_decision()

    def finish_phase_and_advance(self):
        self._invalidate_decision()
        try:
            with self.profiler.stage('environment_phase_advance'):
                return super().finish_phase_and_advance()
        finally:
            self._invalidate_decision()

    def apply_manager_delta(self, delta_vec):
        self._invalidate_decision()
        return super().apply_manager_delta(delta_vec)

    def set_manager_combo(self, weights):
        self._invalidate_decision()
        return super().set_manager_combo(weights)

    def _add_workflow_if_arrived(self):
        self._invalidate_decision()
        return super()._add_workflow_if_arrived()

    def _advance_until_decision_energy_only(self, return_breakdown=False):
        self._invalidate_decision()
        try:
            return super()._advance_until_decision_energy_only(return_breakdown=return_breakdown)
        finally:
            self._invalidate_decision()

    def _process_finish_events_at_current_time(self):
        self._invalidate_decision()
        try:
            return super()._process_finish_events_at_current_time()
        finally:
            self._invalidate_decision()
from mahdrl.config import no_apf_metadata
from mahdrl.simulator import _EnergySimulator
from mahdrl.simulator import _WorkflowSimulator
import numpy as np

class _LegalEnvironment(_CachedEnvironment):

    def variant_metadata(self):
        return no_apf_metadata(dict(super().variant_metadata(), algorithm=self.algorithm))

    def _apf_snapshot(self, tid):
        self._ensure_decision(tid)
        if self._apf_cache is not None:
            return self._apf_cache
        with self.profiler.stage('raw_legal_candidate_preview'):
            rows = {}
            idle = self._vm_idle_mask() & (np.asarray(self.vm_available_at) <= self.current_time + 1e-12)
            for j in np.flatnonzero(idle):
                j = int(j)
                vm = self.vms[self.vm_ids[j]]
                duration = float(self._task_duration(tid, j))
                start = max(float(self.current_time), float(self.vm_available_at[j]))
                energy = self.assignment_ledger.preview(vm.host_id, start, start + duration, vm.pc)
                delay = max(0.0, self.vm_available_at[j] - self.current_time) + duration
                rows[j] = (float(delay), energy, max(0.0, self.current_time + delay - self.task_baseline_finish[tid]))
            ids = np.asarray(list(rows), dtype=np.int64)
            values = np.asarray(list(rows.values()), dtype=np.float64).reshape(-1, 3)
            if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
                raise ValueError('Non-finite or negative raw candidate prediction')
            owner = {j: hi for hi, hid in enumerate(self.host_ids) for j in self.host_to_vm_indices[hid]}
            group = np.asarray([owner[int(j)] for j in ids], dtype=np.int64)
            vms = {hid: set(indices).intersection(rows) for hid, indices in self.host_to_vm_indices.items()}
            vms = {hid: indices for hid, indices in vms.items() if indices}
            self._apf_cache = dict(hosts=set(vms), vms=vms, rows=rows, raw_count=len(rows), r_H=None, r_V=None, host_apf_enabled=False, vm_apf_enabled=False, energy_hosts=set(), energy_vms={}, arrays=dict(ids=ids, group=group, values=values, kept=np.ones(len(ids), dtype=bool)))
        return self._apf_cache

    def host_select(self, host_index):
        _, mask = self._build_host_obs_for_task(self._cur_tid)
        if not 0 <= int(host_index) < len(mask) or mask[int(host_index)] <= 0.5:
            raise ValueError('Host has no physically legal idle VM')
        return _WorkflowSimulator.host_select(self, host_index)

    def vm_assign(self, vm_slot_index):
        tid, hid = (self._cur_tid, self._cur_host_id)
        _, mask = self._build_vm_obs_for_task_host(tid, hid)
        if not 0 <= int(vm_slot_index) < len(mask) or mask[int(vm_slot_index)] <= 0.5:
            raise ValueError('VM is not a physically legal idle slot')
        snapshot = self._apf_snapshot(tid)
        j = self.host_to_vm_indices[hid][int(vm_slot_index)]
        self.apf_decisions.append(dict(task_id=int(tid), host_id=int(hid), vm_id=int(self.vm_ids[j]), host_apf_enabled=False, vm_apf_enabled=False, r_H=None, r_V=None, raw_vm_count=len(snapshot['rows']), raw_host_count=len(snapshot['hosts']), retained_host_count=len(snapshot['hosts']), local_raw_vm_count=len(snapshot['vms'][hid]), local_retained_vm_count=len(snapshot['vms'][hid]), selected_energy_only_host=None, selected_energy_only_vm=None, predicted_ontime=bool(self.current_time + snapshot['rows'][j][0] <= self.task_baseline_finish[tid] + 1e-09)))
        try:
            with self.profiler.stage('environment_assignment'):
                return _EnergySimulator.vm_assign(self, vm_slot_index)
        finally:
            self._invalidate_decision()
from mahdrl.config import Mechanism
from mahdrl.config import _core_mechanism_ALGORITHM
from mahdrl.config import _core_mechanism_variant_metadata
import numpy as np

class _PreviewEnvironment(_LegalEnvironment):
    algorithm = _core_mechanism_ALGORITHM

    def __init__(self, *args, mechanism=Mechanism(), experiment_metadata=None, **kwargs):
        self.mechanism = mechanism
        self._experiment_metadata = experiment_metadata
        if experiment_metadata is not None:
            self.algorithm = experiment_metadata['algorithm']
        super().__init__(*args, **kwargs)

    def variant_metadata(self):
        return dict(super().variant_metadata(), **self._experiment_metadata if self._experiment_metadata is not None else _core_mechanism_variant_metadata(self.mechanism))

    def preview_vm_states(self):
        if self._cur_tid is None or self._cur_host_id is not None:
            raise RuntimeError('Preview requires a current task before host selection')
        tid, states = (self._cur_tid, {})
        snapshot = self._apf_snapshot(tid)
        for hid in self.host_ids:
            if hid not in snapshot['hosts']:
                continue
            x, mask = self._build_vm_obs_for_task_host(tid, hid)
            b, indices = (self.base_vm_obs_dim, self.host_to_vm_indices[hid])
            k = len(indices)
            context = np.concatenate((x[:27], x[b:b + 1], x[b + 1 + self.num_hosts:b + 3 + self.num_hosts]))
            candidates = np.concatenate((x[27:b].reshape(self.max_vms_per_host, 5)[:k], x[b + 3 + self.num_hosts:].reshape(self.max_vms_per_host, 2)[:k]), axis=1)
            rows = np.array([snapshot['rows'].get(j, (0.0, 0.0, 0.0)) for j in indices], dtype=np.float64)
            states[hid] = dict(obs=dict(context=context, candidates=candidates, exists_mask=np.ones(k, np.float32)), mask=mask[:k], rows=rows, performance_times=rows[:, 0], candidate_ids=[self.vm_ids[j] for j in indices])
        return states
from mahdrl.config import VM_CAPACITY
from mahdrl.config import VM_INPUT_DIM
from mahdrl.config import _core_config_ALGORITHM
from mahdrl.config import _core_config_variant_metadata
from mahdrl.replay import fixed_state
from mahdrl.simulator import spaces
import numpy as np

class _FixedSlotEnvironment(_PreviewEnvironment):
    algorithm = _core_config_ALGORITHM

    def __init__(self, *args, **kwargs):
        counts = kwargs.get('vms_per_host', ())
        if counts and max(counts) > VM_CAPACITY:
            raise ValueError('Fixed VA capacity is 9 VMs per host')
        super().__init__(*args, **kwargs)
        if self.max_vms_per_host > VM_CAPACITY:
            raise ValueError('Fixed VA capacity is 9 VMs per host')
        self.observation_space = spaces.Dict(dict(self.observation_space.spaces, vm_obs=spaces.Box(-np.inf, np.inf, (VM_INPUT_DIM,), np.float32), vm_mask=spaces.Box(0.0, 1.0, (VM_CAPACITY,), np.float32)))

    def variant_metadata(self):
        metadata = dict(super().variant_metadata(), **_core_config_variant_metadata(self.mechanism))
        metadata.update(self._experiment_metadata or {})
        return metadata

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        state.update(vm_obs=np.zeros(VM_INPUT_DIM, np.float32), vm_mask=np.zeros(VM_CAPACITY, np.float32))
        return (state, info)

    def preview_vm_states(self):
        return {hid: fixed_state(state) for hid, state in super().preview_vm_states().items()}

    def get_vm_state_for_current_task(self):
        state, ok = super().get_vm_state_for_current_task()
        return (fixed_state(state), ok)
import heapq
import numpy as np

class _HostPrefix:

    def __init__(self, host):
        self.host = host
        self.pending, self.events, self.ends = ([], {}, {})
        self.terms = []
        self.load = self.energy = 0.0
        self.previous = None
        self.right = -float('inf')

    def event(self, when, delta):
        if when not in self.events:
            self.events[when] = 0.0
            heapq.heappush(self.pending, when)
        self.events[when] += delta

    def add(self, record, origin):
        start, end, pc = (float(record.start_time), float(record.end_time), float(record.vm_pc))
        if pc <= 0 or end <= start or end <= 0:
            return
        start = max(origin, start)
        self.ends[start] = max(self.ends.get(start, -float('inf')), end)
        self.event(start, pc if end > start else 0.0)
        if end > start:
            self.event(end, -pc)

    def power(self):
        ratio = max(0.0, min(1.0, self.load / max(float(self.host.total_pc), 1e-09)))
        return float(self.host.power(ratio))

    def at(self, now):
        while self.pending and self.pending[0] < now:
            point = heapq.heappop(self.pending)
            self.right = max(self.right, self.ends.pop(point, -float('inf')))
            if self.previous is None:
                self.previous = point
            term = self.power() * (point - self.previous)
            self.terms.append(term)
            self.energy += term
            self.load += self.events.pop(point)
            self.previous = point
        if self.previous is None:
            return (0.0, 0.0)
        tail = self.power() * max(0.0, min(now, self.right) - self.previous)
        return (self.energy + tail, tail)

class PrefixEnergy:

    def __init__(self, hosts, records, origin):
        self.hosts, self.records, self.origin = (hosts, records, float(origin))
        self.prefix = {hid: _HostPrefix(host) for hid, host in hosts.items()}
        self.count = 0
        self.time = -float('inf')

    def at(self, now):
        now = float(now)
        if now < self.time or len(self.records) < self.count or any((r.start_time < self.time for r in self.records[self.count:])):
            self.__init__(self.hosts, self.records, self.origin)
        for record in self.records[self.count:]:
            self.prefix[record.server_id].add(record, self.origin)
        self.count, self.time = (len(self.records), now)
        terms, per_host = ([], {})
        for hid, prefix in self.prefix.items():
            energy, tail = prefix.at(now)
            per_host[hid] = energy
            terms.extend(prefix.terms)
            terms.append(tail)
        total = float(np.cumsum(terms, dtype=np.float64)[-1]) if terms else 0.0
        return (total, per_host)
from mahdrl.normalization import checked_observation
from mahdrl.normalization import queue_features
import numpy as np

class CloudWorkflowEnv_VMAgents(_FixedSlotEnvironment):

    def _compute_block2(self):
        if 'block2' in self._fast_cache:
            return self._fast_cache['block2']
        if not getattr(self, 'state_normalizer', None):
            return super()._compute_block2()
        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if self.ready_task_ids else 0.0
        block = np.zeros(10, np.float32)
        block[[0, 1, 2, 3, 4, 7, 8, 6]] = queue_features(self)
        block[5] = self.state_normalizer.scale(sum_mi, 'workload_mi_ref')
        block[9] = float(len({self.task_meta[x][0] for x in self.ready_task_ids})) / max(1.0, len(self.workflows))
        self._fast_cache['block2'] = checked_observation(self, block)
        return block

    def _energy_reward_to_current_time_with_breakdown(self):
        if not self.workflows:
            return (0.0, {int(h): 0.0 for h in self.host_ids})
        ledger = getattr(self, '_physical_prefix', None)
        if ledger is None or ledger.records is not self._records or ledger.origin != self.first_arrival_time:
            ledger = self._physical_prefix = PrefixEnergy(self.hosts, self._records, self.first_arrival_time)
        total, per_host = ledger.at(self.current_time)
        delta = max(0.0, total - self._energy_cache)
        self._energy_cache = float(total)
        rewards = {}
        for hid in self.host_ids:
            value = per_host[hid]
            previous = float(self._energy_cache_by_host.get(int(hid), 0.0))
            change = max(0.0, value - previous)
            self._energy_cache_by_host[int(hid)] = float(value)
            rewards[int(hid)] = -(float(change) * float(self.energy_reward_scale))
        self.total_energy += float(delta)
        self.lifetime_energy += float(delta)
        return (-(float(delta) * float(self.energy_reward_scale)), rewards)
