"""Workflow inputs, resource physics, communication, and energy integration."""
import hashlib
import json
import random
import math
import networkx as nx
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field
DAX_TEMPLATE_ID_SCHEME = 'basename_casefold_v1'
DATA_MODEL = 'dynamic_full_parent_output_v1'
PAYLOAD_GENERATOR_VERSION = 'full_parent_outputs_rng_v1'
FILE_SIZE_MB_RANGE = (512, 1024)
FILE_COUNT_RANGE = (1, 3)
LENGTH_MI_RANGE = (60, 90)
FILE_SIZE_CONVERSION = 'MB * 1024 * 1024 * 8.0'

def data_model_metadata():
    return dict(data_model=DATA_MODEL, file_size_mb_range=list(FILE_SIZE_MB_RANGE), output_file_count_range=list(FILE_COUNT_RANGE), entry_input_file_count_range=list(FILE_COUNT_RANGE), file_size_conversion=FILE_SIZE_CONVERSION, payload_generator_version=PAYLOAD_GENERATOR_VERSION, length_mi_range=list(LENGTH_MI_RANGE))

def workflow_instance_hash(graph, tasks):
    """Hash immutable instance data, scoped to this workflow; omit scheduling state."""

    def files(values):
        return sorted(((f.name, f.size_bits, f.source_task_id) for f in values))
    payload = dict(nodes=sorted(graph.nodes), edges=sorted(graph.edges), tasks=[dict(id=t.task_id, workload=t.workload_mi, inputs=files(t.in_files), outputs=files(t.out_files), parents=sorted(t.parent_in_bits.items()), input_bits=t.in_file_size_sum_bits, output_bits=t.out_file_size_sum_bits, parent_bits=t.from_parents_bits, external_bits=t.ext_in_bits) for t in sorted(tasks, key=lambda t: t.task_id)])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def validate_cached_instance(cache, seed, workflow_id, graph, tasks):
    """Check generated payload against its reference record when hashes are supplied."""
    record = (cache or {}).get(int(seed), {})
    hashes = record.get('wf_instance_hashes')
    if hashes is not None:
        if workflow_id >= len(hashes) or hashes[workflow_id] != workflow_instance_hash(graph, tasks):
            raise ValueError(f'Deadline cache instance hash mismatch: seed={seed} workflow={workflow_id}')

def dax_template_id(path):
    """A portable, case-insensitive identifier for a configured DAX template."""
    return str(path).replace('\\', '/').rsplit('/', 1)[-1].casefold()

def validate_dax_template_name(cached_name, runtime_name, *, strict=True):
    if dax_template_id(cached_name) == dax_template_id(runtime_name):
        return
    message = f'[deadline-cache] DAX template mismatch: runtime={runtime_name} cache={cached_name}'
    if strict:
        raise ValueError(message)
    print('WARNING:', message)

@dataclass(frozen=True)
class DataFile:
    name: str
    size_bits: float
    source_task_id: int | None

@dataclass
class Task:
    task_id: int
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    workload_mi: float = 0.0
    in_files: list = field(default_factory=list)
    out_files: list = field(default_factory=list)
    in_file_size_sum_bits: float = 0.0
    out_file_size_sum_bits: float = 0.0
    parent_in_bits: dict = field(default_factory=dict)
    from_parents_bits: float = 0.0
    ext_in_bits: float = 0.0
    state: str = 'unReady'
    assigned_server_id: int | None = None
    assigned_vm_pc: float | None = None
    ready_time: float | None = None
    arrival_time: float | None = None
    start_processing_time: float | None = None
    end_processing_time: float | None = None
    finish_time: float | None = None
    upward_rank: float | None = None
    downward_rank: float | None = None

def _read_dax_to_graph(xml_path):
    """Read dax to graph."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'a_dax': 'http://pegasus.isi.edu/schema/DAX'}
    id_map = {}
    G = nx.DiGraph()
    for idx, job in enumerate(root.findall('a_dax:job', ns)):
        job_id = job.get('id')
        id_map[job_id] = idx
        G.add_node(idx)
    for child in root.findall('a_dax:child', ns):
        child_ref = child.get('ref')
        if child_ref not in id_map:
            continue
        c = id_map[child_ref]
        for parent in child.findall('a_dax:parent', ns):
            p_ref = parent.get('ref')
            if p_ref in id_map:
                G.add_edge(id_map[p_ref], c)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError('The parsed graph is not a DAG.')
    return (G, id_map)

def _randomize_payloads(G, seed=None, length_mi_range=LENGTH_MI_RANGE, io_file_count_range=FILE_COUNT_RANGE, file_size_mb_range=FILE_SIZE_MB_RANGE):
    """Generate unique outputs once; each direct child receives every parent output."""
    rng = random.Random(seed)
    tasks = [Task(tid, parents=list(G.predecessors(tid)), children=list(G.successors(tid))) for tid in sorted(G.nodes)]

    def make_files(prefix, tid, producer):
        return [DataFile(f'{prefix}_{tid}_{i}', rng.uniform(*file_size_mb_range) * 1024 * 1024 * 8.0, producer) for i in range(rng.randint(*io_file_count_range))]
    for t in tasks:
        t.workload_mi = max(abs(rng.uniform(*length_mi_range)), 1e-06)
        t.out_files = make_files('output', t.task_id, t.task_id)
        t.out_file_size_sum_bits = sum((f.size_bits for f in t.out_files))
        if not t.parents:
            t.in_files = make_files('input', t.task_id, None)
    for t in tasks:
        if t.parents:
            t.in_files = list({f.name: f for p in t.parents for f in tasks[p].out_files}.values())
            t.parent_in_bits = {p: tasks[p].out_file_size_sum_bits for p in t.parents}
            t.from_parents_bits = sum(t.parent_in_bits.values())
            t.ext_in_bits = 0.0
            t.in_file_size_sum_bits = t.from_parents_bits
        else:
            t.in_file_size_sum_bits = sum((f.size_bits for f in t.in_files))
            t.ext_in_bits = t.in_file_size_sum_bits
    return tasks

def load_workflow_from_dax(xml_path, seed=None, randomize_payloads=True, length_mi_range=LENGTH_MI_RANGE, io_file_count_range=FILE_COUNT_RANGE, file_size_mb_range=FILE_SIZE_MB_RANGE):
    """Load workflow from dax."""
    G, id_map = _read_dax_to_graph(xml_path)
    if randomize_payloads:
        tasks = _randomize_payloads(G, seed=seed, length_mi_range=length_mi_range, io_file_count_range=io_file_count_range, file_size_mb_range=file_size_mb_range)
    else:
        tasks = [Task(tid) for tid in G.nodes]
        for u, v in G.edges:
            tasks[u].children.append(v)
            tasks[v].parents.append(u)
    return (G, tasks)

def poisson_arrival_times(lmbda, horizon=None, seed=None, n_max=None, start_time=0.0):
    """Poisson arrival times."""
    if horizon is None and n_max is None:
        raise ValueError('poisson_arrival_times: must common either horizon or n_max')
    import numpy as np
    rng = np.random.default_rng(seed)
    if n_max is not None:
        gaps = rng.exponential(1.0 / max(lmbda, 1e-12), size=int(n_max))
        return (start_time + np.cumsum(gaps)).tolist()
    t = start_time
    arr = []
    expected = lmbda * float(horizon)
    if expected > 2000000:
        raise MemoryError(f'poisson_arrival_times: expected arrivals ~{expected:.0f} is too large. Use n_max instead of horizon.')
    while t < start_time + horizon:
        t += rng.exponential(1.0 / max(lmbda, 1e-12))
        if t < start_time + horizon:
            arr.append(t)
    return arr
from dataclasses import dataclass, field

@dataclass
class VM:
    vm_id: int
    host_id: int
    pc: float
    bw: float

@dataclass
class Host:
    host_id: int
    vm_ids: list = field(default_factory=list)
    power_model: object = None
    total_pc: float = 0.0
    total_bw: float = 0.0

    def power(self, load_ratio: float) -> float:
        if load_ratio < 0.0:
            load_ratio = 0.0
        if load_ratio > 1.0:
            load_ratio = 1.0
        if self.power_model is None:
            return 0.0
        return self.power_model(load_ratio)

def _power_model_factory(host_id):
    mod = host_id % 3

    def nec(load):
        if load <= 0.1:
            return load * 44 + 15.9
        if load <= 0.2:
            return load * 21 + 18.2
        if load <= 0.3:
            return load * 20 + 18.4
        if load <= 0.4:
            return load * 28 + 16.0
        if load <= 0.5:
            return load * 26 + 16.8
        if load <= 0.6:
            return load * 32 + 13.8
        if load <= 0.7:
            return load * 38 + 10.2
        if load <= 0.8:
            return load * 27 + 17.9
        if load <= 0.9:
            return load * 31 + 14.7
        return load * 25 + 20.1

    def fujitsu(load):
        if load <= 0.1:
            return load * 44 + 13.1
        if load <= 0.2:
            return load * 30 + 14.5
        if load <= 0.3:
            return load * 25 + 15.5
        if load <= 0.4:
            return load * 24 + 15.8
        if load <= 0.5:
            return load * 28 + 14.2
        if load <= 0.6:
            return load * 37 + 9.7
        if load <= 0.7:
            return load * 54 - 0.5
        if load <= 0.8:
            return load * 72 - 13.1
        if load <= 0.9:
            return load * 70 - 11.5
        return load * 46 + 10.1

    def lenovo(load):
        if load <= 0.1:
            return load * 32 + 16.8
        if load <= 0.2:
            return load * 28 + 17.2
        if load <= 0.3:
            return load * 32 + 16.4
        if load <= 0.4:
            return load * 39 + 14.3
        if load <= 0.5:
            return load * 51 + 9.5
        if load <= 0.6:
            return load * 54 + 8.0
        if load <= 0.7:
            return load * 59 + 5.0
        if load <= 0.8:
            return load * 93 - 18.8
        if load <= 0.9:
            return load * 132 - 50.0
        return load * 184 - 96.8
    return nec if mod == 0 else fujitsu if mod == 1 else lenovo

def create_cluster(num_hosts=3, vms_per_host=(3, 3, 4), pc_tiers=(2.0, 4.0, 8.0), bw_tiers=(2000.0, 4000.0, 8000.0)):
    hosts, vms = ({}, {})
    vm_counter = 0
    for h in range(num_hosts):
        hosts[h] = Host(host_id=h, power_model=_power_model_factory(h))
        vm_count_this_host = vms_per_host[h % len(vms_per_host)]
        for i in range(vm_count_this_host):
            tier = i % len(pc_tiers)
            vm = VM(vm_counter, h, pc_tiers[tier], bw_tiers[tier])
            vms[vm_counter] = vm
            hosts[h].vm_ids.append(vm_counter)
            hosts[h].total_pc += vm.pc
            hosts[h].total_bw += vm.bw
            vm_counter += 1
    return (hosts, vms)
import networkx as nx
import numpy as np
from dataclasses import dataclass
BITS_PER_MB = 1024.0 * 1024.0 * 8.0

@dataclass
class LoadRecord:
    start_time: float
    end_time: float
    server_id: int
    vm_pc: float

class Workflow:

    def __init__(self, workflow_id, graph, tasks, arrival_time, source_device_id, deadline=None):
        self.workflow_id = workflow_id
        self.graph = graph
        self.tasks = tasks
        self.arrival_time = arrival_time
        self.source_device_id = source_device_id
        self.deadline = deadline
        self.finish_time = None
        self.is_arrived = False

    def _all_parents_finished(self, task):
        for pred in self.graph.predecessors(task.task_id):
            if self.tasks[pred].state != 'Finished':
                return False
        return True

    def get_ready_tasks(self):
        ready = []
        for t in self.tasks:
            if t.state == 'unReady' and self._all_parents_finished(t):
                t.state = 'Ready'
            if t.state == 'Ready':
                ready.append(t)
        return ready

    def compute_upward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """Compute upward ranks."""
        for t in self.tasks:
            t.upward_rank = 0.0
        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-09) * 1000000.0
        for tid in reversed(topo):
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-09)
            self_time = upload + comp + download
            succ_rank = 0.0
            for s in self.graph.successors(tid):
                succ_rank = max(succ_rank, self.tasks[s].upward_rank or 0.0)
            t.upward_rank = self_time + succ_rank

    def compute_downward_ranks(self, avg_proc_speed_mi_s: float, avg_tx_speed_Mb_s: float):
        """Compute downward ranks."""
        for t in self.tasks:
            t.downward_rank = 0.0
        topo = list(nx.topological_sort(self.graph))
        avg_tx_bps = max(avg_tx_speed_Mb_s, 1e-09) * 1000000.0
        for tid in topo:
            t = self.tasks[tid]
            upload = (t.ext_in_bits + t.from_parents_bits) / avg_tx_bps
            download = t.out_file_size_sum_bits / avg_tx_bps
            comp = t.workload_mi / max(avg_proc_speed_mi_s, 1e-09)
            self_time = upload + comp + download
            pred_rank = 0.0
            for p in self.graph.predecessors(tid):
                pred_rank = max(pred_rank, self.tasks[p].downward_rank or 0.0)
            t.downward_rank = self_time + pred_rank

def task_subdeadlines(tasks, vm_pc_mi_s, vm_bw_bps, arrival_time, workflow_deadline):
    """Absolute Latest-Finish deadlines using mean execution time across VMs.

    Placements are unknown at workflow arrival, so include all input/output
    data in each VM duration. Do not clamp deadlines to local earliest finish.
    """
    n = len(tasks)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    pc = np.asarray(vm_pc_mi_s, dtype=np.float64).reshape(1, -1)
    bw = np.asarray(vm_bw_bps, dtype=np.float64).reshape(1, -1)
    mi = np.array([float(t.workload_mi) for t in tasks], dtype=np.float64)
    in_bits = np.array([float((getattr(t, 'ext_in_bits', 0.0) or 0.0) + (getattr(t, 'from_parents_bits', 0.0) or 0.0)) for t in tasks], dtype=np.float64)
    out_bits = np.array([float(getattr(t, 'out_file_size_sum_bits', 0.0) or 0.0) for t in tasks], dtype=np.float64)
    durations = in_bits[:, None] / np.maximum(bw, 1e-12) + mi[:, None] / np.maximum(pc, 1e-12) + out_bits[:, None] / np.maximum(bw, 1e-12)
    avg_duration = durations.mean(axis=1)
    children = [list(getattr(t, 'children', [])) for t in tasks]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(((i, child) for i in range(n) for child in children[i]))
    latest_finish = np.full(n, float(workflow_deadline) - float(arrival_time), dtype=np.float64)
    latest_start = np.zeros(n, dtype=np.float64)
    for i in reversed(list(nx.topological_sort(graph))):
        if children[i]:
            latest_finish[i] = min((latest_start[child] for child in children[i]))
        latest_start[i] = latest_finish[i] - avg_duration[i]
    return float(arrival_time) + latest_finish
DEADLINE_ALPHA_SEED_OFFSET = 20260316

def sample_deadline_alpha(episode_seed, workflow_id, choices, probabilities):
    """A paired alpha determined only by seed, workflow ID and distribution.

    Preserve the MAHDRL seed derivation and inverse-CDF draw. Never consume an
    environment RNG, so lookups are independent of order and scheduling.
    """
    values = np.asarray(choices, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or probs.ndim != 1 or len(values) == 0 or (len(values) != len(probs)) or (not np.all(np.isfinite(values))) or (not np.all(np.isfinite(probs))) or np.any(probs < 0) or (not np.isfinite(probs.sum())) or (probs.sum() <= 0):
        raise ValueError('Alpha choices and probabilities must be finite, matching non-empty vectors with positive probability mass')
    order = np.argsort(values, kind='stable')
    values = values[order]
    cdf = np.cumsum(probs[order] / probs.sum())
    local_seed = int(episode_seed) * 1000003 + int(workflow_id) * 9176 + DEADLINE_ALPHA_SEED_OFFSET & 4294967295
    u = float(np.random.RandomState(local_seed).rand())
    index = min(int(np.searchsorted(cdf, u, side='right')), len(values) - 1)
    return float(values[index])
COMMUNICATION_MODEL = 'same_vm_only_v1'

def validate_deadline_cache(obj, path):
    """Reject caches from incompatible file models, parameters or communication rules."""
    expected = dict(communication_model=COMMUNICATION_MODEL, **data_model_metadata())
    mismatched = [key for key, value in expected.items() if obj.get('meta', {}).get(key) != value]
    if mismatched:
        import re
        match = re.search('task([SML])_res([SML])', str(path))
        scenario = f' --task {match[1]} --resource {match[2]}' if match else ' --task S --resource S'
        raise ValueError(f"Incompatible deadline cache ({', '.join(mismatched)}): {path}. Use the bundled matching reference cache for this scene:{scenario}")

def exec_time_components(task, vm_pc_mi_s: float, vm_bw_Mb_s: float, *, parents_loc: dict | None=None, child_vm_id: int | None=None, child_host_id: int | None=None, zero_if_same_vm: bool=True, zero_if_same_host: bool=False):
    """Exec time components."""
    bps = max(vm_bw_Mb_s, 1e-09) * 1000000.0
    from_parents_bits_eff = 0.0
    if parents_loc is None or (child_vm_id is None and child_host_id is None):
        from_parents_bits_eff = task.from_parents_bits
    else:
        for p, bits in task.parent_in_bits.items():
            vmh = parents_loc.get(p)
            if vmh is None:
                from_parents_bits_eff += bits
                continue
            p_vm, p_host = vmh
            same_vm = child_vm_id is not None and p_vm == child_vm_id
            same_host = child_host_id is not None and p_host == child_host_id
            if same_vm and zero_if_same_vm or (same_host and zero_if_same_host):
                pass
            else:
                from_parents_bits_eff += bits
    tau_in = (task.ext_in_bits + from_parents_bits_eff) / bps
    tau_comp = task.workload_mi / max(vm_pc_mi_s, 1e-09)
    tau_out = task.out_file_size_sum_bits / bps
    tau_exec = tau_in + tau_comp + tau_out
    return (tau_in, tau_comp, tau_out, tau_exec)
ENERGY_ACCOUNTING = 'host_task_window_v1'

def energy_from_records(records, hosts, *, start_time=None, end_time=None, accounting=ENERGY_ACCOUNTING):
    """Integrate each host between its first task start and last task end.

    Idle gaps inside that host window count; unused hosts and zero-load
    bookkeeping records do not extend it. Explicit endpoints only clip the
    host windows. full_cluster_v1 is available for historical figure replay.
    """
    if accounting not in (ENERGY_ACCOUNTING, 'full_cluster_v1'):
        raise ValueError('Unknown energy accounting: ' + str(accounting))
    records = [r for r in records if r.vm_pc > 0 and r.end_time > r.start_time]
    if start_time is None:
        start_time = min((r.start_time for r in records), default=0.0)
    if end_time is None:
        end_time = max((r.end_time for r in records), default=start_time)
    start_time, end_time = (float(start_time), float(end_time))
    if end_time < start_time:
        raise ValueError('Energy window end must not precede its start')
    if end_time == start_time:
        return 0.0
    windows = {}
    for hid in hosts:
        recs = [r for r in records if r.server_id == hid]
        if accounting == 'full_cluster_v1':
            left, right = (start_time, end_time)
        elif recs:
            left = max(start_time, min((float(r.start_time) for r in recs)))
            right = min(end_time, max((float(r.end_time) for r in recs)))
        else:
            continue
        if right > left:
            windows[hid] = (left, right)
    events = {hid: {left: 0.0, right: 0.0} for hid, (left, right) in windows.items()}
    for r in records:
        if r.server_id not in windows:
            continue
        left, right = windows[r.server_id]
        start = max(left, float(r.start_time))
        end = min(right, float(r.end_time))
        if end <= start:
            continue
        host_events = events[r.server_id]
        host_events[start] = host_events.get(start, 0.0) + float(r.vm_pc)
        host_events[end] = host_events.get(end, 0.0) - float(r.vm_pc)
    total_energy = 0.0
    for hid, host_events in events.items():
        host = hosts[hid]
        load = 0.0
        previous = windows[hid][0]
        for t, delta in sorted(host_events.items()):
            ratio = max(0.0, min(1.0, load / max(float(host.total_pc), 1e-09)))
            total_energy += float(host.power(ratio)) * (t - previous)
            load += delta
            previous = t
    return float(total_energy)
import numpy as np
TA_WEIGHT_MODEL = 'f17_normalized_weights_v1'
WEIGHT_TOLERANCE = 1e-08

def normalize_initial_weights(weights):
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('TA initial weights must be five finite nonnegative values')
    scale = float(values.max())
    if scale <= 0:
        raise ValueError('TA initial weights must have positive total mass')
    scaled = values / scale
    return (scaled / scaled.sum()).astype(np.float32)

def weight_candidates(weights, deltas):
    """One rule for masking and execution; only roundoff-sized negatives are repaired."""
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(deltas, dtype=np.float64)
    if w.shape != (5,) or not np.isfinite(w).all() or np.any(w < 0) or (not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-07)):
        raise ValueError('TA persistent weights must lie on the normalized nonnegative simplex')
    if d.ndim != 2 or d.shape[1] != 5:
        raise ValueError('TA increments must have five dimensions')
    finite = np.isfinite(d).all(axis=1)
    action_values = np.isclose(d[..., None], [-0.1, 0.0, 0.1], rtol=0, atol=WEIGHT_TOLERANCE).any(axis=-1).all(axis=1)
    u = w[None, :] + d
    valid = finite & action_values & np.isfinite(u).all(axis=1) & (u >= -WEIGHT_TOLERANCE).all(axis=1)
    cleaned = np.maximum(u, 0.0)
    total = cleaned.sum(axis=1)
    valid &= np.isfinite(total) & (total > 0)
    normalized = np.zeros_like(u)
    np.divide(cleaned, total[:, None], out=normalized, where=valid[:, None])
    normalized = normalized.astype(np.float32)
    valid &= np.isfinite(normalized).all(axis=1) & (normalized >= 0).all(axis=1)
    valid &= np.isclose(normalized.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-07)
    normalized[(d == 0).all(axis=1)] = w
    return (normalized, valid)
