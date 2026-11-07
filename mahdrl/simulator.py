"""Workflow execution, task deadlines, and assignment energy settlement."""
from mahdrl.normalization import checked_observation
from mahdrl.normalization import load_normalizer
from mahdrl.normalization import queue_features
from mahdrl.workflow import LoadRecord
from mahdrl.workflow import Workflow
from mahdrl.workflow import create_cluster
from mahdrl.workflow import energy_from_records
from mahdrl.workflow import exec_time_components
from mahdrl.workflow import load_workflow_from_dax
from mahdrl.workflow import normalize_initial_weights
from mahdrl.workflow import poisson_arrival_times
from mahdrl.workflow import weight_candidates
import heapq
import numpy as np
import os
import re
from functools import lru_cache
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    import gym
    from gym import spaces

def _safe_div(a, b, eps=1e-09):
    return float(a) / float(b if abs(b) > eps else eps if b >= 0 else -eps)

def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
MANAGER_STEP = 0.1

def _build_manager_action_table(step: float=0.1) -> np.ndarray:
    vals = (-step, 0.0, step)
    table = []
    for d0 in vals:
        for d1 in vals:
            for d2 in vals:
                for d3 in vals:
                    for d4 in vals:
                        table.append([d0, d1, d2, d3, d4])
    return np.asarray(table, dtype=np.float32)
MANAGER_ACTION_TABLE = _build_manager_action_table(MANAGER_STEP)

@lru_cache(maxsize=None)
def _infer_task_count_from_dax_path(dax_path: str) -> int:
    base = os.path.basename(dax_path)
    m = re.search('_(\\d+)\\.xml$', base)
    if m:
        return int(m.group(1))
    try:
        G, tasks = load_workflow_from_dax(dax_path, seed=0, randomize_payloads=False)
        return int(len(tasks))
    except Exception:
        return 0

def _auto_max_ready_tasks(dax_paths, workflows_per_episode: int) -> int:
    nmax = 0
    for p in dax_paths:
        nmax = max(nmax, _infer_task_count_from_dax_path(str(p)))
    if nmax <= 0:
        nmax = 100
    if workflows_per_episode is None:
        return int(nmax)
    return int(workflows_per_episode * nmax)

def _topo_sort_local(tasks):
    n = len(tasks)
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for t in tasks:
        i = int(t.task_id)
        for c in getattr(t, 'children', []):
            succ[i].append(int(c))
        indeg[i] = len(getattr(t, 'parents', []))
    q = [i for i in range(n) if indeg[i] == 0]
    topo = []
    while q:
        u = q.pop(0)
        topo.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != n:
        topo = list(range(n))
    return (topo, succ)

def _heft_makespan_and_finish_times(tasks, vm_pc, vm_bw_bps):
    n = len(tasks)
    if n == 0:
        return (0.0, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64))
    topo, succ = _topo_sort_local(tasks)
    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array([float((getattr(t, 'ext_in_bits', 0.0) or 0.0) + (getattr(t, 'from_parents_bits', 0.0) or 0.0)) for t in tasks], dtype=np.float64)
    out_bits = np.array([float(getattr(t, 'out_file_size_sum_bits', 0.0) or 0.0) for t in tasks], dtype=np.float64)
    bw = vm_bw_bps.reshape(1, -1)
    pc = vm_pc.reshape(1, -1)
    dur = in_bits.reshape(-1, 1) / np.maximum(bw, 1e-12) + mi.reshape(-1, 1) / np.maximum(pc, 1e-12) + out_bits.reshape(-1, 1) / np.maximum(bw, 1e-12)
    avg_dur = dur.mean(axis=1)
    rank_u = np.zeros((n,), dtype=np.float64)
    for u in reversed(topo):
        if len(succ[u]) == 0:
            rank_u[u] = avg_dur[u]
        else:
            rank_u[u] = avg_dur[u] + float(np.max(rank_u[succ[u]]))
    indeg = np.array([len(getattr(tasks[i], 'parents', [])) for i in range(n)], dtype=np.int32)
    ready = []
    for i in range(n):
        if indeg[i] == 0:
            heapq.heappush(ready, (-rank_u[i], i))
    Pn = vm_pc.size
    vm_ready = np.zeros((Pn,), dtype=np.float64)
    finish = np.zeros((n,), dtype=np.float64)
    start = np.zeros((n,), dtype=np.float64)
    parents = [list(getattr(tasks[i], 'parents', [])) for i in range(n)]
    placement = {}
    scheduled = 0
    while ready:
        _neg_r, u = heapq.heappop(ready)
        if len(parents[u]) == 0:
            ready_t = 0.0
        else:
            ready_t = float(np.max(finish[np.array(parents[u], dtype=np.int32)]))
        st = np.maximum(vm_ready, ready_t)
        candidate_durations = np.array([exec_time_components(tasks[u], float(vm_pc[j]), float(vm_bw_bps[j]) / 1000000.0, parents_loc=placement, child_vm_id=j, zero_if_same_host=False)[-1] for j in range(Pn)])
        ft = st + candidate_durations
        p_star = int(np.argmin(ft))
        start[u] = float(st[p_star])
        finish[u] = float(ft[p_star])
        vm_ready[p_star] = finish[u]
        placement[u] = (p_star, -1)
        scheduled += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (-rank_u[v], v))
    if scheduled != n:
        ms = float(np.max(avg_dur)) if n > 0 else 0.0
        return (ms, finish, avg_dur)
    makespan = float(np.max(finish))
    return (makespan, finish, avg_dur)

class _WorkflowSimulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dax_paths, horizon=36000.0, arrival_lambda=0.1, random_seed=0, max_ready_tasks=32, normalize=True, num_hosts=3, vms_per_host=(9, 8, 8), pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0), bw_tiers=(1000.0, 2000.0, 4000.0, 6000.0, 8000.0), dax_probs=None, state_include_vm=True, state_include_host=True, state_include_queue=True, combo_weights=None, workflows_per_episode=8, normalization_mode='legacy', normalization_config=None, source_train_scene=None):
        super().__init__()
        self.state_normalizer = load_normalizer(normalization_mode, normalization_config, source_train_scene)
        self.rng = np.random.RandomState(random_seed)
        self.horizon = float(horizon)
        self.arrival_lambda = float(arrival_lambda)
        self.dax_paths = list(dax_paths)
        self.random_seed = int(random_seed)
        self.normalize = bool(normalize)
        self.energy_reward_scale = 0.001
        self.task_baseline_norm = 300.0
        self.energy_norm_per_mi_ref = 20.0
        self.alpha_delay_host = 0.35
        self.alpha_delay_vm = 0.6
        self.workflows_per_episode = int(workflows_per_episode) if workflows_per_episode is not None else None
        self.dax_probs = None
        if dax_probs is not None:
            probs = np.asarray(dax_probs, dtype=float).copy()
            assert len(probs) == len(self.dax_paths), 'dax_probs and dax_paths must have matching lengths'
            probs = np.clip(probs, 0.0, None)
            if probs.sum() <= 0:
                probs[:] = 1.0
            self.dax_probs = probs / probs.sum()
        self.hosts, self.vms = create_cluster(num_hosts=num_hosts, vms_per_host=vms_per_host, pc_tiers=pc_tiers, bw_tiers=bw_tiers)
        self.host_ids = sorted(self.hosts.keys())
        self.vm_ids = sorted(self.vms.keys())
        self.num_hosts = len(self.hosts)
        self.num_vms = len(self.vm_ids)
        self.vm_host = np.array([self.vms[vid].host_id for vid in self.vm_ids], dtype=np.int32)
        self._heft_vm_pc = np.array([float(self.vms[vid].pc) for vid in self.vm_ids], dtype=np.float64)
        self._heft_vm_bw_bps = np.array([float(self.vms[vid].bw) * 1000000.0 for vid in self.vm_ids], dtype=np.float64)
        self.host_to_vm_indices = {h: [] for h in self.host_ids}
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            self.host_to_vm_indices[h].append(j)
        self.max_vms_per_host = max((len(vs) for vs in self.host_to_vm_indices.values())) if self.num_hosts > 0 else 0
        if max_ready_tasks is None or (isinstance(max_ready_tasks, (int, float)) and max_ready_tasks <= 0) or (isinstance(max_ready_tasks, str) and max_ready_tasks.lower() == 'auto'):
            self.max_ready = _auto_max_ready_tasks(self.dax_paths, self.workflows_per_episode)
        else:
            self.max_ready = int(max_ready_tasks)
        self.max_ready = max(1, int(self.max_ready))
        self.task_heur_dim = 5
        self.state_include_vm = bool(state_include_vm)
        self.state_include_host = bool(state_include_host)
        self.state_include_queue = bool(state_include_queue)
        initial = normalize_initial_weights([0.2] * 5 if combo_weights is None else combo_weights)
        self._manager_initial_weights = tuple((float(value) for value in initial))
        self.combo_weights = initial.copy()
        self.block1_dim = 12
        self.block2_dim = 10
        self.vm_feat_dim = 5
        self.host_feat_dim = 5
        self.host_obs_dim = self.block1_dim + self.block2_dim + self.host_feat_dim * self.num_hosts
        self.host_act_dim = self.num_hosts
        self.vm_obs_dim = self.block1_dim + self.block2_dim + self.host_feat_dim + self.vm_feat_dim * self.max_vms_per_host
        self.vm_act_dim = self.max_vms_per_host
        self.observation_space = spaces.Dict({'host_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(self.host_obs_dim,), dtype=np.float32), 'host_mask': spaces.Box(low=0.0, high=1.0, shape=(self.host_act_dim,), dtype=np.float32), 'vm_obs': spaces.Box(low=-np.inf, high=np.inf, shape=(self.vm_obs_dim,), dtype=np.float32), 'vm_mask': spaces.Box(low=0.0, high=1.0, shape=(self.vm_act_dim,), dtype=np.float32)})
        self.total_energy = 0.0
        self.lifetime_energy = 0.0
        self._records = []
        self.parents_loc_by_wf = {}
        self._energy_cache = 0.0
        self._energy_cache_by_host = {int(h): 0.0 for h in self.host_ids}
        self._fuse_zero_assign_limit = 50
        self._fuse_no_event_limit = 10
        self._reset_internal_buffers()

    def set_manager_combo(self, weights):
        candidate = normalize_initial_weights(weights)
        self.combo_weights = candidate
        self._phase_started = False
        self._phase_tasks = []

    def apply_manager_delta(self, delta_vec):
        delta = np.asarray(delta_vec, dtype=np.float64)
        if delta.shape != (self.task_heur_dim,):
            raise ValueError('TA increment must have five dimensions')
        candidates, valid = weight_candidates(self.combo_weights, delta[None, :])
        if not valid[0]:
            raise ValueError('Illegal TA increment: candidate weights must be nonnegative with positive total')
        self.combo_weights = candidates[0].copy()
        self._phase_started = False
        self._phase_tasks = []

    def get_manager_action_mask(self) -> np.ndarray:
        _, valid = weight_candidates(self.combo_weights, MANAGER_ACTION_TABLE)
        return valid.astype(np.float32)

    def get_manager_state(self):
        now = getattr(self, 'current_time', 0.0)
        ready_cnt = float(len(self.ready_task_ids))
        waiting_cnt = float(sum((1 for s in self.task_state if s == 'unReady')))
        running_cnt = float(sum((1 for s in self.task_state if s == 'Running')))
        waits = [max(0.0, now - self.task_ready_time[tid]) for tid in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0
        ready_ratio = ready_cnt / max(1.0, self.max_ready)
        waiting_ratio = waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0)
        running_ratio = running_cnt / max(1.0, self.num_vms)
        idle_vms = float(np.sum(self._vm_idle_mask()))
        idle_vm_ratio = idle_vms / max(1.0, self.num_vms)
        host_load_list, host_running_ratio_list = ([], [])
        for h in self.host_ids:
            active_pc = 0.0
            running = 0
            cnt = 0
            soonest = None
            for j, vid in enumerate(self.vm_ids):
                if self.vms[vid].host_id == h:
                    cnt += 1
                    if self.vm_available_at[j] > now + 1e-12:
                        active_pc += self.vms[vid].pc
                        running += 1
                    if soonest is None or self.vm_available_at[j] < soonest:
                        soonest = self.vm_available_at[j]
            host_load_list.append(_safe_div(active_pc, max(self.hosts[h].total_pc, 1e-09)))
            host_running_ratio_list.append(_safe_div(running, max(cnt, 1)))
        host_load_mean = float(np.mean(host_load_list)) if host_load_list else 0.0
        host_running_ratio_mean = float(np.mean(host_running_ratio_list)) if host_running_ratio_list else 0.0
        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_mean = float(np.mean([self.vms[v].bw for v in self.vms])) if len(self.vms) > 0 else 1.0
        bw_bps = bw_mean * 1000000.0
        slacks = []
        for tid in self.ready_task_ids:
            exp_t = self.task_in_bits[tid] / max(bw_bps, 1e-09) + self.task_mi[tid] / max(pc_mean, 1.0) + self.task_out_bits[tid] / max(bw_bps, 1e-09)
            wf_idx, _ = self.task_meta[tid]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, 'deadline', None)
            if dl is None:
                continue
            slacks.append(dl - (now + exp_t))
        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
            min_slack_norm = min_slack / self._obs_ref('T_ref', max(self.horizon, 1.0))
            avg_slack_norm = avg_slack / self._obs_ref('T_ref', max(self.horizon, 1.0))
        else:
            min_slack_norm = 0.0
            avg_slack_norm = 0.0
        vec = np.array([ready_ratio, waiting_ratio, running_ratio, avg_wait / self._obs_ref('T_ref', max(self.horizon, 1.0)), max_wait / self._obs_ref('T_ref', max(self.horizon, 1.0)), idle_vm_ratio, host_load_mean, host_running_ratio_mean, min_slack_norm, avg_slack_norm], dtype=np.float32)
        if getattr(self, 'state_normalizer', None):
            q = queue_features(self)
            vec[[0, 1, 2, 3, 4, 8, 9]] = q[:7]
        w_norm = self.combo_weights.astype(np.float32)
        return checked_observation(self, np.concatenate([vec, w_norm], axis=0).astype(np.float32))

    def get_host_state_for_next_assignment(self):
        if self.done_flag:
            return ({'obs': np.zeros(self.host_obs_dim, np.float32), 'mask': np.zeros(self.host_act_dim, np.float32)}, False)
        if not self._phase_started:
            self._phase_prepare_tasks()
            self._phase_started = True
            self._phase_assign_cnt = 0
            self._phase_assigned_tids = []
            self._phase_size_sum_mi = 0.0
            self._phase_preadvanced = False
            self._phase_preadvance_r_energy_total = 0.0
            self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        if not self._has_decision_point():
            return ({'obs': np.zeros(self.host_obs_dim, np.float32), 'mask': np.zeros(self.host_act_dim, np.float32)}, False)
        if len(self._phase_tasks) == 0:
            return ({'obs': np.zeros(self.host_obs_dim, np.float32), 'mask': np.zeros(self.host_act_dim, np.float32)}, False)
        self._cur_tid = int(self._phase_tasks.pop(0))
        self._cur_host_id = None
        obs, mask = self._build_host_obs_for_task(self._cur_tid)
        if np.sum(mask) <= 0.0:
            self._phase_tasks.insert(0, self._cur_tid)
            self._cur_tid = None
            return ({'obs': np.zeros(self.host_obs_dim, np.float32), 'mask': np.zeros(self.host_act_dim, np.float32)}, False)
        return ({'obs': obs.astype(np.float32), 'mask': mask.astype(np.float32)}, True)

    def host_select(self, host_index: int):
        assert hasattr(self, '_cur_tid') and self._cur_tid is not None, 'Call get_host_state_for_next_assignment() first'
        hi = int(host_index)
        if hi < 0 or hi >= self.num_hosts:
            raise ValueError(f'host_index out of range: {hi}')
        host_id = self.host_ids[hi]
        has_idle = False
        now = self.current_time
        for j in self.host_to_vm_indices[host_id]:
            if self.vm_available_at[j] <= now + 1e-09:
                has_idle = True
                break
        if not has_idle:
            raise ValueError(f'Selected host {host_id} has no idle VM at time {now:.6f}')
        self._cur_host_id = host_id

    def get_vm_state_for_current_task(self):
        if self.done_flag:
            return ({'obs': np.zeros(self.vm_obs_dim, np.float32), 'mask': np.zeros(self.vm_act_dim, np.float32)}, False)
        if not hasattr(self, '_cur_tid') or self._cur_tid is None:
            return ({'obs': np.zeros(self.vm_obs_dim, np.float32), 'mask': np.zeros(self.vm_act_dim, np.float32)}, False)
        if self._cur_host_id is None:
            return ({'obs': np.zeros(self.vm_obs_dim, np.float32), 'mask': np.zeros(self.vm_act_dim, np.float32)}, False)
        obs, mask = self._build_vm_obs_for_task_host(self._cur_tid, self._cur_host_id)
        if np.sum(mask) <= 0.0:
            return ({'obs': np.zeros(self.vm_obs_dim, np.float32), 'mask': np.zeros(self.vm_act_dim, np.float32)}, False)
        return ({'obs': obs.astype(np.float32), 'mask': mask.astype(np.float32)}, True)

    def _phase_has_immediate_next_assignment(self) -> bool:
        if self.done_flag:
            return False
        if len(self._phase_tasks) == 0:
            return False
        return bool(self._has_decision_point())

    def vm_assign(self, vm_slot_index: int):
        assert hasattr(self, '_cur_tid') and self._cur_tid is not None, 'Call get_host_state_for_next_assignment() first'
        assert self._cur_host_id is not None, 'Call host_select() first'
        tid = int(self._cur_tid)
        host_id = int(self._cur_host_id)
        slot = int(vm_slot_index)
        if slot < 0 or slot >= self.max_vms_per_host:
            r_host = -1.0
            r_vm = -1.0
            info = {'invalid': 1, 'reason': 'vm_slot_out_of_range'}
            self._cur_tid = None
            self._cur_host_id = None
            return (float(r_host), float(r_vm), info)
        vm_list = self.host_to_vm_indices[host_id]
        if slot >= len(vm_list):
            r_host = -1.0
            r_vm = -1.0
            info = {'invalid': 1, 'reason': 'vm_slot_padding'}
            self._cur_tid = None
            self._cur_host_id = None
            return (float(r_host), float(r_vm), info)
        vm_global_idx = int(vm_list[slot])
        now = self.current_time
        if self.vm_available_at[vm_global_idx] > now + 1e-09:
            r_host = -1.0
            r_vm = -1.0
            info = {'invalid': 1, 'reason': 'vm_not_idle'}
            self._cur_tid = None
            self._cur_host_id = None
            return (float(r_host), float(r_vm), info)
        mi = self.task_mi[tid]
        duration = self._task_duration(tid, vm_global_idx)
        start_time = max(now, float(self.vm_available_at[vm_global_idx]))
        finish_time = start_time + duration
        if tid < len(self.task_baseline_finish):
            deadline = float(self.task_baseline_finish[tid])
        else:
            deadline = float(finish_time)
        lateness = max(0.0, float(finish_time - deadline))
        delay_norm = _clip01(lateness / max(self.task_baseline_norm, 1e-09))
        r_delay = -float(delay_norm)
        self._assign_task_to_specific_vm(tid, vm_global_idx)
        self._phase_assign_cnt += 1
        self._phase_assigned_tids.append(tid)
        self._phase_size_sum_mi += float(mi)
        if self._phase_has_immediate_next_assignment():
            r_energy_total = 0.0
            r_energy_host = 0.0
        else:
            r_energy_total, r_energy_by_host = self._advance_until_decision_energy_only(return_breakdown=True)
            r_energy_host = float(r_energy_by_host.get(host_id, 0.0))
            self._phase_preadvanced = True
            self._phase_preadvance_r_energy_total = float(r_energy_total)
            self._phase_preadvance_r_energy_by_host = {int(h): float(r_energy_by_host.get(h, 0.0)) for h in self.host_ids}
        r_host = (1.0 - float(self.alpha_delay_host)) * float(r_energy_total) + float(self.alpha_delay_host) * float(r_delay)
        r_vm = (1.0 - float(self.alpha_delay_vm)) * float(r_energy_host) + float(self.alpha_delay_vm) * float(r_delay)
        info = {'invalid': 0, 'task_id': tid, 'host_id': host_id, 'vm_global_idx': vm_global_idx, 'start_time': float(start_time), 'finish_time': float(finish_time), 'deadline': float(deadline), 'lateness': float(lateness), 'delay_norm': float(delay_norm), 'r_delay': float(r_delay), 'r_energy_total': float(r_energy_total), 'r_energy_host': float(r_energy_host), 'lower_reward_mode': 'marl_consistent'}
        self._cur_tid = None
        self._cur_host_id = None
        return (float(r_host), float(r_vm), info)

    def finish_phase_and_advance(self):
        if self._phase_assign_cnt == 0 and self._has_decision_point():
            self._fuse_zero_assign_streak += 1
            if self._fuse_zero_assign_streak >= self._fuse_zero_assign_limit:
                self._fuse_zero_assign_streak = 0
                if len(self.ready_task_ids) > 0 and np.any(self._vm_idle_mask()):
                    forced_tid = int(self.ready_task_ids[0])
                    forced_vm = int(self._vm_serial_order()[0])
                    if self._vm_idle_mask()[forced_vm]:
                        self._assign_task_to_specific_vm(forced_tid, forced_vm)
                        self._phase_assign_cnt += 1
                        self._phase_assigned_tids.append(forced_tid)
                        self._phase_size_sum_mi += float(self.task_mi[forced_tid])
        else:
            self._fuse_zero_assign_streak = 0
        lower_energy_already_emitted = bool(self._phase_preadvanced)
        if self._phase_preadvanced:
            r_energy_phase = float(self._phase_preadvance_r_energy_total)
            r_energy_by_host = dict(self._phase_preadvance_r_energy_by_host)
        else:
            r_energy_phase, r_energy_by_host = self._advance_until_decision_energy_only(return_breakdown=True)
        if self._phase_assign_cnt > 0 and self._phase_size_sum_mi > 0:
            r_manager_raw = r_energy_phase / float(self._phase_size_sum_mi)
        else:
            r_manager_raw = 0.0
        info = {'current_time': self.current_time, 'assign_cnt': int(self._phase_assign_cnt), 'phase_size_sum_mi': float(self._phase_size_sum_mi), 'r_energy_phase': float(r_energy_phase), 'r_energy_by_host': dict(r_energy_by_host), 'lower_energy_already_emitted': lower_energy_already_emitted, 'terminated': bool(self.done_flag and self._episode_should_end()), 'truncated': bool(self.done_flag and (not self._episode_should_end())), 'phase_assigned_tids': list(self._phase_assigned_tids)}
        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None
        self._phase_preadvanced = False
        self._phase_preadvance_r_energy_total = 0.0
        self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        return (float(r_manager_raw), info)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_seed = int(seed)
        super().reset(seed=seed)
        self._reset_internal_buffers()
        self.combo_weights = np.array(self._manager_initial_weights, dtype=np.float32)
        self.arrival_times = poisson_arrival_times(self.arrival_lambda, n_max=self.workflows_per_episode, seed=self.random_seed)
        if self.workflows_per_episode is not None:
            n = int(self.workflows_per_episode)
            if n > 0 and len(self.arrival_times) > n:
                self.arrival_times = self.arrival_times[:n]
        self.first_arrival_time = float(self.arrival_times[0]) if self.arrival_times else 0.0
        self.next_arrival_idx = 0
        _ = self._advance_until_decision_energy_only()
        self._phase_started = False
        self._phase_tasks = []
        self._phase_assigned_tids = []
        self._phase_assign_cnt = 0
        self._phase_size_sum_mi = 0.0
        self._cur_tid = None
        self._cur_host_id = None
        self._phase_preadvanced = False
        self._phase_preadvance_r_energy_total = 0.0
        self._phase_preadvance_r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        self._fuse_zero_assign_streak = 0
        self._fuse_no_event_streak = 0
        info = {'current_time': getattr(self, 'current_time', 0.0), 'first_arrival_time': self.first_arrival_time, 'wf_target': self.workflows_per_episode if self.workflows_per_episode is not None else None}
        obs = {'host_obs': np.zeros(self.host_obs_dim, np.float32), 'host_mask': np.zeros(self.host_act_dim, np.float32), 'vm_obs': np.zeros(self.vm_obs_dim, np.float32), 'vm_mask': np.zeros(self.vm_act_dim, np.float32)}
        return (obs, info)

    def render(self):
        idle_vms = int(np.sum(self._vm_idle_mask()))
        print(f'[t={self.current_time:.2f}] ready={len(self.ready_task_ids)} idleVM={idle_vms} done={self.done_flag}')

    def _reset_internal_buffers(self):
        self.current_time = 0.0
        self.done_flag = False
        self.workflows = []
        self.arrival_times = []
        self.next_arrival_idx = 0
        self.task_meta = []
        self.task_state = []
        self.task_parents = []
        self.task_children = []
        self.task_mi = []
        self.task_in_bits = []
        self.task_out_bits = []
        self.task_up_rank = []
        self.task_down_rank = []
        self.task_ready_time = []
        self.task_end_time = []
        self.ready_task_ids = []
        self.event_heap = []
        self.vm_available_at = np.zeros(self.num_vms, dtype=np.float64)
        self.completed_workflows = 0
        self.wf_remaining_tasks = {}
        self.task_baseline_finish = []
        self.wf_finish_time = {}
        self.wf_lateness = {}
        self.wf_timeout = {}
        self.ep_wf_lateness_sum = 0.0
        self.total_energy = 0.0
        self.lifetime_energy = 0.0
        self._records = []
        self.parents_loc_by_wf = {}
        self._energy_cache = 0.0
        self._energy_cache_by_host = {int(h): 0.0 for h in self.host_ids}
        self._cur_tid = None
        self._cur_host_id = None

    def _no_more_arrivals(self):
        return self.next_arrival_idx >= len(self.arrival_times)

    def _add_workflow_if_arrived(self):
        added = False
        while self.next_arrival_idx < len(self.arrival_times) and self.arrival_times[self.next_arrival_idx] <= self.current_time:
            arr_t = self.arrival_times[self.next_arrival_idx]
            if self.dax_probs is None:
                idx = self.rng.randint(len(self.dax_paths))
            else:
                idx = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
            path = self.dax_paths[idx]
            G, tasks = load_workflow_from_dax(path, seed=self.random_seed + self.next_arrival_idx, randomize_payloads=True)
            wf = Workflow(workflow_id=len(self.workflows), graph=G, tasks=tasks, arrival_time=arr_t, source_device_id=0)
            wf.is_arrived = True
            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)
            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)
            heft_ms, heft_finish_rel, avg_dur = _heft_makespan_and_finish_times(tasks, self._heft_vm_pc, self._heft_vm_bw_bps)
            alpha = 1.5
            wf_deadline_abs = float(arr_t + alpha * float(heft_ms))
            wf.deadline = wf_deadline_abs
            wf.heft_makespan = float(heft_ms)
            n_local = len(tasks)
            topo, succ = _topo_sort_local(tasks)
            P = avg_dur.reshape(-1).astype(np.float64)
            D_wf_rel = float(alpha * float(heft_ms))
            LF = np.zeros((n_local,), dtype=np.float64)
            LS = np.zeros((n_local,), dtype=np.float64)
            is_sink = np.array([len(succ[i]) == 0 for i in range(n_local)], dtype=bool)
            for i in range(n_local):
                if is_sink[i]:
                    LF[i] = D_wf_rel
                    LS[i] = LF[i] - P[i]
            for u in reversed(topo):
                if is_sink[u]:
                    continue
                if len(succ[u]) > 0:
                    min_LS_child = float(np.min(LS[np.array(succ[u], dtype=np.int32)]))
                    LF[u] = min_LS_child
                else:
                    LF[u] = D_wf_rel
                LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                LS[u] = LF[u] - P[u]
            for u in range(n_local):
                if is_sink[u]:
                    LF[u] = max(float(LF[u]), float(heft_finish_rel[u]))
                    LS[u] = LF[u] - P[u]
            base = len(self.task_meta)
            for local_id in range(n_local):
                abs_deadline = float(arr_t + LF[local_id])
                self.task_baseline_finish.append(abs_deadline)
            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append('unReady')
                self.task_parents.append(list(t.parents))
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)
            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = 'Ready'
                    self.task_ready_time[gid] = self.current_time
                    self.ready_task_ids.append(gid)
            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True
        return added

    def _vm_idle_mask(self):
        return self.vm_available_at <= self.current_time + 1e-09

    def _has_decision_point(self):
        return len(self.ready_task_ids) > 0 and np.any(self._vm_idle_mask())

    def _episode_should_end(self):
        if self.workflows_per_episode is None:
            return self.current_time >= self.horizon and len(self.event_heap) == 0 and self._no_more_arrivals()
        else:
            all_arrived = self._no_more_arrivals()
            all_done = self.completed_workflows >= self.workflows_per_episode
            no_running = len(self.event_heap) == 0
            no_ready = len(self.ready_task_ids) == 0
            return all_arrived and all_done and no_running and no_ready

    def _energy_reward_to_current_time_with_breakdown(self):
        started = len(self.workflows) > 0
        if not started:
            return (0.0, {int(h): 0.0 for h in self.host_ids})
        t = self.current_time
        clipped_all = []
        clipped_by_host = {int(h): [] for h in self.host_ids}
        for r in self._records:
            if r.end_time <= 0.0:
                continue
            if r.start_time >= t:
                continue
            end = min(r.end_time, t)
            if end <= r.start_time:
                continue
            rr = LoadRecord(r.start_time, end, r.server_id, r.vm_pc)
            clipped_all.append(rr)
            clipped_by_host[int(r.server_id)].append(rr)
        E_total = energy_from_records(clipped_all, self.hosts, start_time=self.first_arrival_time, end_time=t)
        dE_total = max(0.0, E_total - self._energy_cache)
        self._energy_cache = float(E_total)
        dE_by_host = {}
        for h in self.host_ids:
            E_h = energy_from_records(clipped_by_host[int(h)], {int(h): self.hosts[int(h)]}, start_time=self.first_arrival_time, end_time=t)
            prev_h = float(self._energy_cache_by_host.get(int(h), 0.0))
            dE_h = max(0.0, float(E_h) - prev_h)
            self._energy_cache_by_host[int(h)] = float(E_h)
            dE_by_host[int(h)] = float(dE_h)
        self.total_energy += float(dE_total)
        self.lifetime_energy += float(dE_total)
        r_total = -(float(dE_total) * float(self.energy_reward_scale))
        r_by_host = {int(h): -(float(dE_by_host[int(h)]) * float(self.energy_reward_scale)) for h in self.host_ids}
        return (float(r_total), r_by_host)

    def _energy_reward_to_current_time(self) -> float:
        r_total, _ = self._energy_reward_to_current_time_with_breakdown()
        return float(r_total)

    def _advance_until_decision_energy_only(self, return_breakdown: bool=False):
        r_energy = 0.0
        r_energy_by_host = {int(h): 0.0 for h in self.host_ids}
        while True:
            self._add_workflow_if_arrived()
            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break
            next_times = []
            if len(self.event_heap) > 0:
                next_times.append(self.event_heap[0][0])
            if not self._no_more_arrivals():
                next_times.append(self.arrival_times[self.next_arrival_idx])
            if len(next_times) == 0:
                if self._episode_should_end():
                    self.done_flag = True
                    break
                raise RuntimeError('Unfinished MAHDRL episode has no arrival/completion event')
            next_t = min(next_times)
            if next_t <= self.current_time + 1e-12:
                next_t = self.current_time + 1e-09
            self.current_time = next_t
            r0, rb = self._energy_reward_to_current_time_with_breakdown()
            r_energy += float(r0)
            for h in self.host_ids:
                r_energy_by_host[int(h)] += float(rb.get(int(h), 0.0))
            self._process_finish_events_at_current_time()
            if self._has_decision_point():
                self._fuse_no_event_streak = 0
                break
            if self._episode_should_end() or self.current_time >= self.horizon:
                self.done_flag = True
                break
        if return_breakdown:
            return (float(r_energy), {int(h): float(r_energy_by_host[int(h)]) for h in self.host_ids})
        return float(r_energy)

    def _process_finish_events_at_current_time(self):
        while len(self.event_heap) > 0 and self.event_heap[0][0] <= self.current_time + 1e-12:
            _, etype, task_id, vm_idx = heapq.heappop(self.event_heap)
            if etype != 'finish':
                continue
            self.task_state[task_id] = 'Finished'
            self.task_end_time[task_id] = self.current_time
            wf_id, _ = self.task_meta[task_id]
            if wf_id in self.wf_remaining_tasks:
                self.wf_remaining_tasks[wf_id] -= 1
                if self.wf_remaining_tasks[wf_id] == 0:
                    self.completed_workflows += 1
                    wf = self.workflows[wf_id]
                    finish_t = float(self.current_time)
                    self.wf_finish_time[wf_id] = finish_t
                    wf_finish_late = max(0.0, finish_t - float(getattr(wf, 'deadline', finish_t)))
                    self.wf_lateness[wf_id] = float(wf_finish_late)
                    self.wf_timeout[wf_id] = bool(finish_t > float(getattr(wf, 'deadline', finish_t)))
                    self.ep_wf_lateness_sum += float(wf_finish_late)
            wf_idx, _ = self.task_meta[task_id]
            for child_local in self.task_children[task_id]:
                child_gid = None
                for gid, meta in enumerate(self.task_meta):
                    if meta[0] == wf_idx and meta[1] == child_local:
                        child_gid = gid
                        break
                if child_gid is None:
                    continue
                if self.task_state[child_gid] == 'unReady':
                    all_ok = True
                    for p_local in self.task_parents[child_gid]:
                        p_gid = None
                        for gid, meta in enumerate(self.task_meta):
                            if meta[0] == wf_idx and meta[1] == p_local:
                                p_gid = gid
                                break
                        if p_gid is None or self.task_state[p_gid] != 'Finished':
                            all_ok = False
                            break
                    if all_ok:
                        self.task_state[child_gid] = 'Ready'
                        self.task_ready_time[child_gid] = self.current_time
                        self.ready_task_ids.append(child_gid)
        self.ready_task_ids = [tid for tid in self.ready_task_ids if self.task_state[tid] == 'Ready']

    def _compute_task_heuristics_for_ready(self):
        """F15: FCFS/SJF/MCF/HUR/EDF; every returned column is higher-priority-first."""
        sel_ids = [tid for tid in self.ready_task_ids if self.task_state[tid] == 'Ready']
        if not sel_ids:
            return ([], np.zeros((0, self.task_heur_dim), dtype=np.float64))
        now = float(self.current_time)
        pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
        bw_mean_bps = float(np.mean([self.vms[v].bw for v in self.vms])) * 1000000.0
        if not np.isfinite([now, pc_mean, bw_mean_bps]).all() or pc_mean <= 0 or bw_mean_bps <= 0:
            raise ValueError('F15 requires finite time and positive mean VM capacity/bandwidth')
        raw = np.empty((len(sel_ids), self.task_heur_dim), dtype=np.float64)
        for i, tid in enumerate(sel_ids):
            if tid >= len(self.task_baseline_finish):
                raise ValueError(f'F15 Ready task {tid} has no absolute task sub-deadline')
            processing = float(self.task_in_bits[tid]) / bw_mean_bps + float(self.task_mi[tid]) / pc_mean + float(self.task_out_bits[tid]) / bw_mean_bps
            deadline = float(self.task_baseline_finish[tid])
            raw[i] = [self.task_ready_time[tid], processing, self.task_up_rank[tid], deadline - now - processing, deadline]
        if not np.isfinite(raw).all():
            raise ValueError('F15 task attributes must be finite')
        direction = np.array([-1.0, -1.0, 1.0, -1.0, -1.0])
        if not self.normalize:
            return (sel_ids, raw * direction)
        scale = np.maximum(np.max(np.abs(raw), axis=0), 1.0)
        scaled = raw / scale
        low, high = (scaled.min(axis=0), scaled.max(axis=0))
        span = high - low
        numerator = np.where(direction > 0, scaled - low, high - scaled)
        feats = np.divide(numerator, span, out=np.zeros_like(raw), where=span > 0)
        return (sel_ids, np.clip(feats, 0.0, 1.0))

    def _phase_prepare_tasks(self):
        self._add_workflow_if_arrived()
        sel_ids, feats = self._compute_task_heuristics_for_ready()
        if feats.shape[0] == 0:
            self._phase_tasks = []
            return
        scores = feats @ self.combo_weights
        if not np.isfinite(scores).all():
            raise ValueError('F15 weighted priorities must be finite')
        order = sorted(range(len(sel_ids)), key=lambda i: (-float(scores[i]), self.task_ready_time[sel_ids[i]], sel_ids[i]))
        self._phase_tasks = [int(sel_ids[i]) for i in order]

    def _compute_block2(self):
        now = self.current_time
        ready_cnt = float(len(self.ready_task_ids))
        waiting_cnt = float(sum((1 for s in self.task_state if s == 'unReady')))
        running_cnt = float(sum((1 for s in self.task_state if s == 'Running')))
        waits = [max(0.0, now - self.task_ready_time[x]) for x in self.ready_task_ids]
        avg_wait = float(np.mean(waits)) if len(waits) > 0 else 0.0
        max_wait = float(np.max(waits)) if len(waits) > 0 else 0.0
        sum_mi = float(np.sum([self.task_mi[x] for x in self.ready_task_ids])) if len(self.ready_task_ids) > 0 else 0.0
        urgent_cnt = 0
        slacks = []
        for x in self.ready_task_ids:
            exp_t = float(np.mean([self._task_duration(x, j) for j in range(self.num_vms)]))
            wf_idx, _ = self.task_meta[x]
            wf = self.workflows[wf_idx]
            dl = getattr(wf, 'deadline', None)
            if dl is not None:
                s = float(dl) - (now + exp_t)
                slacks.append(s)
                if s < 0:
                    urgent_cnt += 1
        if len(slacks) > 0:
            min_slack = float(np.min(slacks))
            avg_slack = float(np.mean(slacks))
        else:
            min_slack = 0.0
            avg_slack = 0.0
        block2 = np.array([ready_cnt / max(1.0, self.max_ready), waiting_cnt / max(1.0, len(self.task_state) if len(self.task_state) > 0 else 1.0), running_cnt / max(1.0, self.num_vms), avg_wait / self._obs_ref('T_ref', max(self.horizon, 1.0)), max_wait / self._obs_ref('T_ref', max(self.horizon, 1.0)), sum_mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0), float(urgent_cnt) / max(1.0, len(self.ready_task_ids)), min_slack / self._obs_ref('T_ref', max(self.horizon, 1.0)), avg_slack / self._obs_ref('T_ref', max(self.horizon, 1.0)), float(len(set([self.task_meta[x][0] for x in self.ready_task_ids]))) / max(1.0, len(self.workflows) if len(self.workflows) > 0 else 1.0)], dtype=np.float32)
        if getattr(self, 'state_normalizer', None):
            q = queue_features(self)
            block2[[0, 1, 2, 3, 4, 7, 8, 6]] = q
            block2[5] = self.state_normalizer.scale(sum_mi, 'workload_mi_ref')
        return checked_observation(self, block2)

    def _compute_block1(self, tid: int):
        now = self.current_time
        in_bits = float(self.task_in_bits[tid])
        out_bits = float(self.task_out_bits[tid])
        mi = float(self.task_mi[tid])
        io_ratio = _safe_div(in_bits + out_bits, max(mi, 1.0))
        wait_norm = (now - self.task_ready_time[tid]) / self._obs_ref('T_ref', max(self.horizon, 1.0))
        children_cnt_n = float(len(self.task_children[tid])) / max(1.0, np.mean([len(ch) for ch in self.task_children]) + 1e-06)
        up_rank_n = self.task_up_rank[tid] / max(1.0, np.mean(self.task_up_rank) if len(self.task_up_rank) > 0 else 1.0)
        down_rank_n = self.task_down_rank[tid] / max(1.0, np.mean(self.task_down_rank) if len(self.task_down_rank) > 0 else 1.0)
        vm_idle = self._vm_idle_mask()
        idle_indices = np.where(vm_idle)[0].tolist()
        if len(idle_indices) == 0:
            est_best = est_mean = 0.0
        else:
            durations = []
            for j in idle_indices:
                durations.append(self._task_duration(tid, j))
            est_best = float(np.min(durations))
            est_mean = float(np.mean(durations))
        wf_idx, _ = self.task_meta[tid]
        wf = self.workflows[wf_idx]
        if getattr(wf, 'deadline', None) is not None:
            slack_on_best = float(wf.deadline) - (now + est_best)
        else:
            slack_on_best = 0.0
        same_wf_ready_frac = 0.0
        if len(self.ready_task_ids) > 0:
            same = [x for x in self.ready_task_ids if self.task_meta[x][0] == wf_idx]
            same_wf_ready_frac = float(len(same)) / float(max(1, len(self.ready_task_ids)))
        block1 = np.array([mi / max(1.0, np.mean(self.task_mi) if len(self.task_mi) > 0 else 1.0), in_bits / max(1.0, np.mean(self.task_in_bits) if len(self.task_in_bits) > 0 else 1.0), out_bits / max(1.0, np.mean(self.task_out_bits) if len(self.task_out_bits) > 0 else 1.0), io_ratio, wait_norm, children_cnt_n, up_rank_n, down_rank_n, est_best / self._obs_ref('T_ref', max(self.horizon, 1.0)), est_mean / self._obs_ref('T_ref', max(self.horizon, 1.0)), slack_on_best / self._obs_ref('T_ref', max(self.horizon, 1.0)), same_wf_ready_frac], dtype=np.float32)
        if getattr(self, 'state_normalizer', None):
            block1[3] = self.state_normalizer.io_ratio(io_ratio)
            block1[[0, 1, 2, 6, 7]] = [self.state_normalizer.scale(v, k) for v, k in zip((mi, in_bits, out_bits, self.task_up_rank[tid], self.task_down_rank[tid]), ('workload_mi_ref', 'input_bits_ref', 'output_bits_ref', 'up_rank_ref', 'down_rank_ref'))]
        return (checked_observation(self, block1), in_bits, out_bits, mi)

    def _compute_host_block(self):
        now = self.current_time
        host_stats = {'active_pc': {h: 0.0 for h in self.host_ids}, 'idle_vm': {h: 0 for h in self.host_ids}, 'soonest': {h: None for h in self.host_ids}, 'running': {h: 0 for h in self.host_ids}, 'vm_cnt': {h: 0 for h in self.host_ids}}
        for j, vid in enumerate(self.vm_ids):
            h = self.vms[vid].host_id
            host_stats['vm_cnt'][h] += 1
            if self.vm_available_at[j] > now + 1e-12:
                host_stats['active_pc'][h] += self.vms[vid].pc
                host_stats['running'][h] += 1
            else:
                host_stats['idle_vm'][h] += 1
            if host_stats['soonest'][h] is None or self.vm_available_at[j] < host_stats['soonest'][h]:
                host_stats['soonest'][h] = self.vm_available_at[j]
        mean_total_pc = max(1.0, float(np.mean([self.hosts[x].total_pc for x in self.host_ids]))) if self.num_hosts > 0 else 1.0
        host_block = np.zeros((self.num_hosts, self.host_feat_dim), dtype=np.float32)
        for hi, h in enumerate(self.host_ids):
            vm_cnt = max(1, host_stats['vm_cnt'][h])
            total_pc_n = self.hosts[h].total_pc / mean_total_pc
            host_load = _safe_div(host_stats['active_pc'][h], max(self.hosts[h].total_pc, 1e-09))
            host_idle_vm = float(host_stats['idle_vm'][h]) / vm_cnt
            soonest = host_stats['soonest'][h] if host_stats['soonest'][h] is not None else now
            host_soonest_delay = max(0.0, float(soonest - now))
            host_running_ratio = _safe_div(host_stats['running'][h], vm_cnt)
            if getattr(self, 'state_normalizer', None):
                total_pc_n = self.state_normalizer.scale(self.hosts[h].total_pc, 'host_pc_ref')
            host_block[hi] = np.array([total_pc_n, host_load, host_idle_vm, host_soonest_delay / self._obs_ref('T_ref', max(self.horizon, 1e-09)), host_running_ratio], dtype=np.float32)
        return checked_observation(self, host_block)

    def _build_host_obs_for_task(self, tid: int):
        block1, _, _, _ = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block = self._compute_host_block()
        obs = np.concatenate([block1, block2, host_block.reshape(-1)], axis=0)
        mask = np.zeros((self.host_act_dim,), dtype=np.float32)
        now = self.current_time
        for hi, h in enumerate(self.host_ids):
            ok = False
            for j in self.host_to_vm_indices[h]:
                if self.vm_available_at[j] <= now + 1e-09:
                    ok = True
                    break
            if ok:
                mask[hi] = 1.0
        return (checked_observation(self, obs), mask)

    def _build_vm_obs_for_task_host(self, tid: int, host_id: int):
        block1, in_bits, out_bits, mi = self._compute_block1(tid)
        block2 = self._compute_block2()
        host_block_all = self._compute_host_block()
        hi = self.host_ids.index(host_id)
        host_feat = host_block_all[hi].astype(np.float32)
        mean_vm_pc = max(1.0, float(np.mean([self.vms[v].pc for v in self.vms])))
        mean_vm_bw = max(1.0, float(np.mean([self.vms[v].bw for v in self.vms])))
        vm_block = np.zeros((self.max_vms_per_host, self.vm_feat_dim), dtype=np.float32)
        mask = np.zeros((self.vm_act_dim,), dtype=np.float32)
        now = self.current_time
        vm_list = self.host_to_vm_indices[host_id]
        for slot in range(self.max_vms_per_host):
            if slot >= len(vm_list):
                continue
            j = vm_list[slot]
            vid = self.vm_ids[j]
            vm = self.vms[vid]
            idle_flag = 1.0 if self.vm_available_at[j] <= now + 1e-12 else 0.0
            avail_delay = max(0.0, float(self.vm_available_at[j] - now))
            pred_finish = max(now, float(self.vm_available_at[j])) + self._task_duration(tid, j)
            vm_block[slot] = np.array([idle_flag, avail_delay / self._obs_ref('T_ref', max(self.horizon, 1e-09)), vm.pc / self._obs_ref('vm_pc_ref', mean_vm_pc), vm.bw / self._obs_ref('vm_bw_ref', mean_vm_bw), (max(0.0, pred_finish - now) if self.state_normalizer else pred_finish) / self._obs_ref('T_ref', max(self.horizon, 1e-09))], dtype=np.float32)
            if idle_flag > 0.5:
                mask[slot] = 1.0
        obs = np.concatenate([block1, block2, host_feat, vm_block.reshape(-1)], axis=0)
        return (checked_observation(self, obs), mask)

    def _estimate_marginal_energy_proxy(self, host_id: int, start_time: float, duration: float, added_pc: float) -> float:
        if duration <= 0.0:
            return 0.0
        host = self.hosts[host_id]
        total_pc = max(host.total_pc, 1e-09)
        active_pc_before = 0.0
        for j in self.host_to_vm_indices[host_id]:
            if self.vm_available_at[j] > start_time + 1e-12:
                vid = self.vm_ids[j]
                active_pc_before += float(self.vms[vid].pc)
        load_before = _safe_div(active_pc_before, total_pc)
        load_after = _safe_div(active_pc_before + float(added_pc), total_pc)
        load_before = float(np.clip(load_before, 0.0, 1.0))
        load_after = float(np.clip(load_after, 0.0, 1.0))
        P_before = float(host.power(load_before))
        P_after = float(host.power(load_after))
        dP = max(0.0, P_after - P_before)
        return float(dP * duration)

    def _task_duration(self, task_id: int, vm_index: int) -> float:
        wf_id, local_id = self.task_meta[task_id]
        vm_id = self.vm_ids[vm_index]
        vm = self.vms[vm_id]
        return float(exec_time_components(self.workflows[int(wf_id)].tasks[int(local_id)], vm_pc_mi_s=float(vm.pc), vm_bw_Mb_s=float(vm.bw), parents_loc=self.parents_loc_by_wf.get(int(wf_id), {}), child_vm_id=int(vm_id), child_host_id=int(vm.host_id), zero_if_same_vm=True, zero_if_same_host=False)[-1])

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        vm_id = self.vm_ids[vm_index]
        now = self.current_time
        pc_mi_s = self.vms[vm_id].pc
        start_time = max(now, float(self.vm_available_at[vm_index]))
        end_time = start_time + self._task_duration(task_id, vm_index)
        self.task_state[task_id] = 'Running'
        self.task_end_time[task_id] = end_time
        self.vm_available_at[vm_index] = end_time
        if task_id in self.ready_task_ids:
            self.ready_task_ids.remove(task_id)
        heapq.heappush(self.event_heap, (end_time, 'finish', task_id, vm_index))
        host_id = self.vms[vm_id].host_id
        self._records.append(LoadRecord(start_time, end_time, host_id, pc_mi_s))
        wf_id, local_id = self.task_meta[task_id]
        self.parents_loc_by_wf.setdefault(int(wf_id), {})[int(local_id)] = (int(vm_id), int(host_id))

    def _vm_serial_order(self):
        avail = np.round(self.vm_available_at, 12)
        neg_pc = -np.array([self.vms[vid].pc for vid in self.vm_ids], dtype=np.float64)
        idx = np.arange(self.num_vms, dtype=np.int64)
        order = np.lexsort((idx, neg_pc, avail))
        return order

    def _obs_ref(self, key, legacy):
        normalizer = getattr(self, 'state_normalizer', None)
        return normalizer.scales[key] if normalizer else legacy
from mahdrl.workflow import Workflow
from mahdrl.workflow import load_workflow_from_dax
from mahdrl.workflow import sample_deadline_alpha
from mahdrl.workflow import task_subdeadlines
from mahdrl.workflow import validate_cached_instance
from mahdrl.workflow import validate_dax_template_name
from mahdrl.workflow import validate_deadline_cache
import os
import json
import numpy as np
try:
    import gymnasium as gym
except Exception:
    import gym

class _DeadlineSimulator(_WorkflowSimulator):
    """CloudWorkflowEnv VMAgents."""

    def __init__(self, *args, deadline_mode: str='cache_fcfs', deadline_cache_path: str=None, deadline_cache_strict: bool=True, deadline_alpha_small: float=2.0, deadline_alpha_large: float=3.0, deadline_alpha_small_prob: float=0.2, manager_alpha_delay: float=None, manager_alpha_energy: float=None, manager_delay_mode: str='tardiness', **kwargs):
        self.deadline_mode = str(deadline_mode)
        self.deadline_cache_path = deadline_cache_path
        self.deadline_cache_strict = bool(deadline_cache_strict)
        self.deadline_alpha_small = float(deadline_alpha_small)
        self.deadline_alpha_large = float(deadline_alpha_large)
        self.deadline_alpha_small_prob = float(deadline_alpha_small_prob)
        if manager_alpha_delay is not None and manager_alpha_energy is not None:
            raise ValueError('Specify only one of manager_alpha_delay and manager_alpha_energy')
        if manager_alpha_delay is None and manager_alpha_energy is None:
            manager_alpha_delay = 0.5
        elif manager_alpha_delay is None:
            manager_alpha_delay = 1.0 - float(manager_alpha_energy)
        self.manager_alpha_delay = float(manager_alpha_delay)
        self.manager_delay_mode = str(manager_delay_mode).lower()
        self._deadline_cache = None
        super().__init__(*args, **kwargs)
        if self.deadline_mode == 'cache_fcfs':
            if not self.deadline_cache_path:
                raise ValueError("deadline_mode='cache_fcfs' requires deadline_cache_path")
            self._deadline_cache = self._load_deadline_cache(self.deadline_cache_path)
        self._mgr_prev_total_energy = 0.0
        self._mgr_finished_task_ids = set()

    @staticmethod
    def _load_deadline_cache(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'deadline cache not found: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        validate_deadline_cache(obj, path)
        cache = {}
        if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], list):
            for rec in obj['data']:
                sd = int(rec['seed'])
                cache[sd] = rec
        elif isinstance(obj, dict) and 'seed_to_record' in obj:
            for k, rec in obj['seed_to_record'].items():
                cache[int(k)] = rec
        else:
            raise ValueError('Deadline-cache JSON must contain data (list) or seed_to_record (dict)')
        return cache

    def _cache_lookup_makespan(self, episode_seed: int, wf_id: int, dax_name: str=None) -> float:
        if self._deadline_cache is None:
            raise RuntimeError('deadline cache not loaded')
        rec = self._deadline_cache.get(int(episode_seed), None)
        if rec is None:
            msg = f'[deadline-cache] Missing record for seed={episode_seed}'
            if self.deadline_cache_strict:
                raise KeyError(msg)
            print('WARNING:', msg, '-> fallback makespan_ref=0')
            return 0.0
        ms_list = rec.get('wf_makespans', None)
        if ms_list is None:
            raise ValueError(f'[deadline-cache] seed={episode_seed} record is missing wf_makespans')
        if wf_id >= len(ms_list):
            msg = f'[deadline-cache] seed={episode_seed} wf_id={wf_id} is out of range; cached workflows={len(ms_list)}'
            if self.deadline_cache_strict:
                raise IndexError(msg)
            print('WARNING:', msg, '-> fallback makespan_ref=0')
            return 0.0
        if dax_name is not None:
            names = rec.get('wf_dax_names', None)
            if isinstance(names, list) and wf_id < len(names):
                validate_dax_template_name(names[wf_id], dax_name, strict=self.deadline_cache_strict)
        return float(ms_list[wf_id])

    def _sample_deadline_alpha_for_workflow(self, wf_id: int) -> float:
        p = float(np.clip(self.deadline_alpha_small_prob, 0.0, 1.0))
        return sample_deadline_alpha(self.random_seed, wf_id, (self.deadline_alpha_small, self.deadline_alpha_large), (p, 1.0 - p))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.random_seed = int(seed)
        self.rng = np.random.RandomState(int(self.random_seed))
        ret = super().reset(seed=seed, options=options)
        self._mgr_prev_total_energy = float(getattr(self, 'total_energy', 0.0))
        self._mgr_finished_task_ids = set()
        return ret

    def _collect_new_finished_task_ids(self):
        old_finished = set(self._mgr_finished_task_ids)
        cur_finished = set()
        new_finished = []
        n_tasks = len(getattr(self, 'task_state', []))
        for tid in range(n_tasks):
            if self.task_state[tid] == 'Finished':
                cur_finished.add(tid)
                if tid not in old_finished:
                    new_finished.append(tid)
        self._mgr_finished_task_ids = cur_finished
        return new_finished

    def _compute_phase_delay(self, finished_tids):
        phase_delay = 0.0
        for tid in finished_tids:
            if tid >= len(self.task_baseline_finish):
                continue
            finish_t = float(self.task_end_time[tid])
            ddl_t = float(self.task_baseline_finish[tid])
            late = finish_t - ddl_t
            if self.manager_delay_mode == 'tardiness':
                phase_delay += max(0.0, late)
            elif self.manager_delay_mode == 'lateness':
                phase_delay += late
            else:
                raise ValueError(f'Unsupported manager_delay_mode: {self.manager_delay_mode}')
        return float(phase_delay)

    def finish_phase_and_advance(self):
        r_old, pinfo = super().finish_phase_and_advance()
        pinfo = dict(pinfo or {})
        cur_total_energy = float(getattr(self, 'total_energy', 0.0))
        phase_energy = cur_total_energy - float(self._mgr_prev_total_energy)
        self._mgr_prev_total_energy = cur_total_energy
        new_finished_tids = self._collect_new_finished_task_ids()
        phase_delay = self._compute_phase_delay(new_finished_tids)
        energy_scale = float(getattr(self, 'energy_reward_scale', 1.0))
        delay_norm = max(float(getattr(self, 'task_baseline_norm', 1.0)), 1e-12)
        phase_energy_cost = phase_energy * energy_scale
        phase_delay_cost = phase_delay / delay_norm
        alpha_delay = float(np.clip(self.manager_alpha_delay, 0.0, 1.0))
        alpha_energy = 1.0 - alpha_delay
        r_manager_new = -(alpha_delay * phase_delay_cost + alpha_energy * phase_energy_cost)
        pinfo['phase_energy'] = float(phase_energy)
        pinfo['phase_delay'] = float(phase_delay)
        pinfo['phase_energy_cost'] = float(phase_energy_cost)
        pinfo['phase_delay_cost'] = float(phase_delay_cost)
        pinfo['manager_alpha_delay'] = float(alpha_delay)
        pinfo['manager_alpha_energy'] = float(alpha_energy)
        pinfo['manager_reward_old'] = float(r_old)
        pinfo['manager_reward_new'] = float(r_manager_new)
        pinfo['manager_new_finished_task_cnt'] = int(len(new_finished_tids))
        return (float(r_manager_new), pinfo)

    def _add_workflow_if_arrived(self):
        added = False
        while self.next_arrival_idx < len(self.arrival_times) and self.arrival_times[self.next_arrival_idx] <= self.current_time:
            arr_t = self.arrival_times[self.next_arrival_idx]
            if self.dax_probs is None:
                idx = self.rng.randint(len(self.dax_paths))
            else:
                idx = self.rng.choice(len(self.dax_paths), p=self.dax_probs)
            path = self.dax_paths[idx]
            dax_name = os.path.basename(str(path))
            G, tasks = load_workflow_from_dax(path, seed=self.random_seed + self.next_arrival_idx, randomize_payloads=True)
            if self.deadline_mode == 'cache_fcfs':
                validate_cached_instance(self._deadline_cache, self.random_seed, len(self.workflows), G, tasks)
            wf = Workflow(workflow_id=len(self.workflows), graph=G, tasks=tasks, arrival_time=arr_t, source_device_id=0)
            wf.is_arrived = True
            wf.dax_path = str(path)
            pc_mean = float(np.mean([self.vms[v].pc for v in self.vms]))
            bw_mean = float(np.mean([self.vms[v].bw for v in self.vms]))
            wf.compute_upward_ranks(pc_mean, bw_mean)
            wf.compute_downward_ranks(pc_mean, bw_mean)
            self.wf_remaining_tasks[wf.workflow_id] = len(tasks)
            ddl_alpha = self._sample_deadline_alpha_for_workflow(wf.workflow_id)
            wf.deadline_alpha = float(ddl_alpha)
            if self.deadline_mode == 'cache_fcfs':
                makespan_ref = self._cache_lookup_makespan(episode_seed=int(self.random_seed), wf_id=int(wf.workflow_id), dax_name=dax_name)
                wf.ref_makespan = float(makespan_ref)
                wf.deadline = float(arr_t + ddl_alpha * float(makespan_ref))
            elif self.deadline_mode == 'none':
                wf.ref_makespan = None
                wf.deadline = float(arr_t + max(self.horizon, 1.0))
            else:
                raise ValueError(f'Unsupported deadline_mode: {self.deadline_mode}')
            subdeadlines = task_subdeadlines(tasks, self._heft_vm_pc, self._heft_vm_bw_bps, arr_t, wf.deadline)
            self.task_baseline_finish.extend(subdeadlines.tolist())
            base = len(self.task_meta)
            for t in tasks:
                self.task_meta.append((wf.workflow_id, t.task_id))
                self.task_state.append('unReady')
                self.task_parents.append(list(t.parents))
                self.task_children.append(list(t.children))
                self.task_mi.append(float(t.workload_mi))
                in_bits_total = float((t.ext_in_bits or 0.0) + (t.from_parents_bits or 0.0))
                self.task_in_bits.append(in_bits_total)
                self.task_out_bits.append(float(t.out_file_size_sum_bits))
                self.task_up_rank.append(float(t.upward_rank or 0.0))
                self.task_down_rank.append(float(t.downward_rank or 0.0))
                self.task_ready_time.append(0.0)
                self.task_end_time.append(0.0)
            for local_id, t in enumerate(tasks):
                if len(t.parents) == 0:
                    gid = base + local_id
                    self.task_state[gid] = 'Ready'
                    self.task_ready_time[gid] = self.current_time
                    self.ready_task_ids.append(gid)
            self.workflows.append(wf)
            self.next_arrival_idx += 1
            added = True
        return added
from mahdrl.normalization import checked_observation
from bisect import bisect_right
import numpy as np
VARIANT_SCHEME = 'mahdrl_oe_1235_v1'
POWER_REF_W = 100.0
HOST_SUMMARY_FIELDS = ('fastest_duration', 'fastest_energy', 'min_tardiness', 'min_energy', 'min_energy_duration')
VM_EXTRA_FIELDS = ('predicted_tardiness', 'instant_power_delta')

class AssignmentEnergyLedger:
    """Exact piecewise-load integration, with side-effect-free candidate previews.

    Like PD3QN's interval accumulator, segments include zero-load gaps inside
    used host windows. Outside a host's own window there is no baseline cost.
    """

    def __init__(self, hosts):
        self.hosts = hosts
        self.segments = {int(h): [] for h in hosts}
        self._segment_ends = {int(h): [] for h in hosts}
        self.total_energy = 0.0

    def _energy(self, host_id, load, duration):
        host = self.hosts[int(host_id)]
        return float(host.power(load / max(host.total_pc, 1e-12)) * duration)

    def _addition(self, host_id, start, end, added_pc):
        start, end, added_pc = (float(start), float(end), float(added_pc))
        if not np.isfinite([start, end, added_pc]).all() or end <= start or added_pc <= 0:
            raise ValueError('Assignment energy requires a finite positive load interval')
        segs = self.segments[int(host_id)]
        extension = 0.0
        if segs:
            left, right = (segs[0][0], segs[-1][1])
            if start < left:
                segs = [(start, left, 0.0)] + segs
                extension += self._energy(host_id, 0.0, left - start)
            if end > right:
                segs = segs + [(right, end, 0.0)]
                extension += self._energy(host_id, 0.0, end - right)
        updated = []
        index = 0
        while index < len(segs) and segs[index][1] <= start:
            updated.append(segs[index])
            index += 1
        current, delta = (start, extension)
        while current < end:
            if index >= len(segs) or segs[index][0] >= end:
                right = end if index >= len(segs) else min(end, segs[index][0])
                updated.append((current, right, added_pc))
                delta += self._energy(host_id, added_pc, right - current)
                current = right
                continue
            left, right, load = segs[index]
            if left > current:
                stop = min(end, left)
                updated.append((current, stop, added_pc))
                delta += self._energy(host_id, added_pc, stop - current)
                current = stop
                continue
            if left < current:
                updated.append((left, current, load))
            stop = min(end, right)
            delta += self._energy(host_id, load + added_pc, stop - current) - self._energy(host_id, load, stop - current)
            updated.append((current, stop, load + added_pc))
            current = stop
            if stop < right:
                updated.append((stop, right, load))
            index += 1
        updated.extend(segs[index:])
        merged = []
        for left, right, load in updated:
            if right <= left:
                continue
            if merged and merged[-1][1] == left and (merged[-1][2] == load):
                merged[-1] = (merged[-1][0], right, load)
            else:
                merged.append((left, right, load))
        return (float(delta), merged)

    def preview(self, host_id, start, end, added_pc):
        start, end, added_pc = (float(start), float(end), float(added_pc))
        if not np.isfinite([start, end, added_pc]).all() or end <= start or added_pc <= 0:
            raise ValueError('Assignment energy requires a finite positive load interval')
        segs = self.segments[int(host_id)]
        if not segs:
            return self._energy(host_id, added_pc, end - start)
        left, right = (segs[0][0], segs[-1][1])
        delta = self._energy(host_id, 0.0, max(0.0, left - start) + max(0.0, end - right))
        extra_duration = max(0.0, min(end, left) - start) + max(0.0, end - max(start, right))
        delta += self._energy(host_id, added_pc, extra_duration) - self._energy(host_id, 0.0, extra_duration)
        index = bisect_right(self._segment_ends[int(host_id)], start)
        while index < len(segs) and segs[index][0] < end:
            a, b, load = segs[index]
            duration = max(0.0, min(b, end) - max(a, start))
            delta += self._energy(host_id, load + added_pc, duration) - self._energy(host_id, load, duration)
            index += 1
        return float(delta)

    def add(self, host_id, start, end, added_pc):
        delta, updated = self._addition(host_id, start, end, added_pc)
        self.segments[int(host_id)] = updated
        self._segment_ends[int(host_id)] = [item[1] for item in updated]
        self.total_energy += delta
        return delta

class _EnergySimulator(_DeadlineSimulator):
    """Original dynamics with appended observations and a separate energy ledger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_host_obs_dim = self.host_obs_dim
        self.base_vm_obs_dim = self.vm_obs_dim
        self.host_obs_dim += 1 + 5 * self.num_hosts
        self.vm_obs_dim += 1 + self.num_hosts + 2 + 2 * self.max_vms_per_host
        self.observation_space = spaces.Dict({'host_obs': spaces.Box(-np.inf, np.inf, (self.host_obs_dim,), np.float32), 'host_mask': spaces.Box(0.0, 1.0, (self.host_act_dim,), np.float32), 'vm_obs': spaces.Box(-np.inf, np.inf, (self.vm_obs_dim,), np.float32), 'vm_mask': spaces.Box(0.0, 1.0, (self.vm_act_dim,), np.float32)})
        self._reset_assignment_ledger()

    def _reset_assignment_ledger(self):
        self.assignment_ledger = AssignmentEnergyLedger(self.hosts)
        self.assignment_energy_total = 0.0
        self.lower_energy_emitted = 0.0
        self._candidate_cache = {}

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self._reset_assignment_ledger()
        return result

    def variant_metadata(self):
        time_ref = float(self._obs_ref('T_ref', self.task_baseline_norm))
        return dict(scheme=VARIANT_SCHEME, changes=[1, 2, 3, 5], deadline_action_filter=False, time_ref_s=time_ref, power_ref_w=POWER_REF_W, energy_ref_j=POWER_REF_W * time_ref, host_summary_fields=list(HOST_SUMMARY_FIELDS), vm_extra_fields=list(VM_EXTRA_FIELDS))

    def estimate_assignment_energy(self, tid, vm_index):
        vm = self.vms[self.vm_ids[int(vm_index)]]
        start = max(float(self.current_time), float(self.vm_available_at[int(vm_index)]))
        end = start + self._task_duration(int(tid), int(vm_index))
        return self.assignment_ledger.preview(vm.host_id, start, end, vm.pc)

    def _candidate_rows(self, tid):
        key = (len(self._records), float(self.current_time), int(tid), float(self.task_baseline_finish[tid]))
        if key not in self._candidate_cache:
            rows = {}
            for j in np.flatnonzero(self._vm_idle_mask()):
                j = int(j)
                duration = float(self._task_duration(tid, j))
                lateness = max(0.0, self.current_time + duration - self.task_baseline_finish[tid])
                rows[j] = (duration, self.estimate_assignment_energy(tid, j), lateness)
            self._candidate_cache = {key: rows}
        return self._candidate_cache[key]

    def _build_host_obs_for_task(self, tid):
        original, mask = super()._build_host_obs_for_task(tid)
        metadata = self.variant_metadata()
        time_ref, energy_ref = (metadata['time_ref_s'], metadata['energy_ref_j'])
        candidates = self._candidate_rows(tid)
        summary = np.zeros((self.num_hosts, 5), dtype=np.float64)
        for hi, host_id in enumerate(self.host_ids):
            indices = [j for j in self.host_to_vm_indices[host_id] if j in candidates]
            if not indices:
                continue
            fastest = min(indices, key=lambda j: (candidates[j][0], j))
            cheapest = min(indices, key=lambda j: (candidates[j][1], j))
            fast_t, fast_e, _ = candidates[fastest]
            cheap_t, cheap_e, _ = candidates[cheapest]
            summary[hi] = [fast_t / time_ref, fast_e / energy_ref, min((candidates[j][2] for j in indices)) / time_ref, cheap_e / energy_ref, cheap_t / time_ref]
        remaining = (self.task_baseline_finish[tid] - self.current_time) / time_ref
        obs = np.concatenate((original, [remaining], summary.reshape(-1))).astype(np.float32)
        return (checked_observation(self, obs), mask)

    def _build_vm_obs_for_task_host(self, tid, host_id):
        original, mask = super()._build_vm_obs_for_task_host(tid, host_id)
        time_ref = self.variant_metadata()['time_ref_s']
        now, host = (float(self.current_time), self.hosts[host_id])
        active_pc = sum((self.vms[self.vm_ids[j]].pc for j in self.host_to_vm_indices[host_id] if self.vm_available_at[j] > now + 1e-12))
        current_load = active_pc / max(host.total_pc, 1e-12)
        current_power = float(host.power(current_load))
        host_identity = np.zeros(self.num_hosts, dtype=np.float32)
        host_identity[self.host_ids.index(host_id)] = 1.0
        extra = np.zeros((self.max_vms_per_host, 2), dtype=np.float64)
        for slot, j in enumerate(self.host_to_vm_indices[host_id]):
            vm = self.vms[self.vm_ids[j]]
            finish = max(now, float(self.vm_available_at[j])) + self._task_duration(tid, j)
            extra[slot, 0] = max(0.0, finish - self.task_baseline_finish[tid]) / time_ref
            if mask[slot] > 0.5:
                after = host.power((active_pc + vm.pc) / max(host.total_pc, 1e-12))
                extra[slot, 1] = (after - current_power) / POWER_REF_W
        remaining = (self.task_baseline_finish[tid] - now) / time_ref
        obs = np.concatenate((original, [remaining], host_identity, [host.power(0.0) / POWER_REF_W, current_power / POWER_REF_W], extra.reshape(-1))).astype(np.float32)
        return (checked_observation(self, obs), mask)

    def _assign_task_to_specific_vm(self, task_id, vm_index):
        super()._assign_task_to_specific_vm(task_id, vm_index)
        record = self._records[-1]
        self._last_assignment_energy = self.assignment_ledger.add(record.server_id, record.start_time, record.end_time, record.vm_pc)
        self.assignment_energy_total = self.assignment_ledger.total_energy
        self._candidate_cache.clear()

    def vm_assign(self, vm_slot_index):
        _old_host, _old_vm, info = super().vm_assign(vm_slot_index)
        if info.get('invalid'):
            return (_old_host, _old_vm, info)
        delta = float(self._last_assignment_energy)
        self.lower_energy_emitted += delta
        energy = -delta * self.energy_reward_scale
        info.update(physical_interval_r_energy_total=info['r_energy_total'], physical_interval_r_energy_host=info['r_energy_host'], r_energy_total=energy, r_energy_host=energy, assignment_energy=delta, lower_reward_mode=VARIANT_SCHEME)
        host_reward = (1.0 - self.alpha_delay_host) * energy + self.alpha_delay_host * info['r_delay']
        vm_reward = (1.0 - self.alpha_delay_vm) * energy + self.alpha_delay_vm * info['r_delay']
        return (float(host_reward), float(vm_reward), info)

    def finish_phase_and_advance(self):
        manager_reward, info = super().finish_phase_and_advance()
        unreturned = self.assignment_energy_total - self.lower_energy_emitted
        info.update(lower_assignment_energy_unemitted=float(unreturned), lower_energy_already_emitted=True, assignment_energy_total=float(self.assignment_energy_total), assignment_energy_physical_gap=float(self.assignment_energy_total - self.total_energy), lower_reward_mode=VARIANT_SCHEME)
        self.lower_energy_emitted += unreturned
        return (manager_reward, info)
