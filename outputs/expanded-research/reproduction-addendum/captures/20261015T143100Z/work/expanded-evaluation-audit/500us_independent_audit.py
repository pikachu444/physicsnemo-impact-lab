"""Bounded independent audit of the frozen central FEM continuation; no ML/GPU."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from collections import Counter
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/expanded-research/benchmarks/drop_test'
FINAL = BASE/'500us-final'
OUT = BASE/'500us-final-review.json'
MD = BASE/'500us-final-review.md'
CHUNK = 8192
RESOURCES = []
READ_BYTES = Counter()


class Memory(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [(x, ctypes.c_ulonglong) for x in ('total_phys', 'avail_phys', 'total_commit', 'avail_commit', 'total_virtual', 'avail_virtual', 'extended')]


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in ('peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]


def now():
    return datetime.now(timezone.utc).isoformat()


def guard(stage):
    notice = 0
    while True:
        host = Memory(); host.length = ctypes.sizeof(host)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(host))
        proc = ProcessMemory(); proc.cb = ctypes.sizeof(proc)
        getproc = ctypes.windll.kernel32.GetCurrentProcess; getproc.restype = ctypes.c_void_p
        getmem = ctypes.windll.psapi.GetProcessMemoryInfo
        getmem.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
        assert getmem(getproc(), ctypes.byref(proc), proc.cb)
        if proc.peak_rss > 200*1024**2:
            raise RuntimeError(f'Own RSS exceeded200MiB at {stage}: {proc.peak_rss}')
        if host.avail_phys >= 1024**3:
            row = {'stage': stage, 'utc': now(), 'host_available_bytes': host.avail_phys, 'rss_bytes': proc.rss, 'peak_rss_bytes': proc.peak_rss}
            RESOURCES.append(row)
            return row
        if time.monotonic() >= notice:
            print(json.dumps({'status': 'waiting_for_1GiB_host_available', 'stage': stage, 'available_bytes': host.avail_phys, 'utc': now()}), flush=True)
            notice = time.monotonic()+30
        time.sleep(2)


def one_cpu():
    getproc = ctypes.windll.kernel32.GetCurrentProcess; getproc.restype = ctypes.c_void_p
    a, b = ctypes.c_size_t(), ctypes.c_size_t()
    fn = ctypes.windll.kernel32.GetProcessAffinityMask
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    assert fn(getproc(), ctypes.byref(a), ctypes.byref(b))
    chosen = a.value & -a.value
    fn = ctypes.windll.kernel32.SetProcessAffinityMask
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    assert fn(getproc(), chosen)
    return chosen


def read_json(path):
    path = Path(path)
    assert path.stat().st_size < 2*1024**2, path
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path, category):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            value.update(block); READ_BYTES[category] += len(block)
    return value.hexdigest()


def layout(path):
    """Offsets independently derived from FASTMAGI10's converter/old verified reader."""
    ranges = {}
    with Path(path).open('rb') as stream:
        def ints(count):
            return struct.unpack('>'+str(count)+'i', stream.read(count*4))
        def skip(count):
            stream.seek(int(count), 1)
        def names(count, width):
            return [stream.read(width).split(b'\0', 1)[0].decode('ascii').replace(' ', '_') for _ in range(count)]
        def field(name, count, size=4):
            ranges[name] = (stream.tell(), count*size); skip(count*size)
        magic, stamp = struct.unpack('>if', stream.read(8)); assert magic == 0x542c
        skip(243); flags = ints(10)
        nn, nf, npart, ns, nes, nv, nt, nk = ints(8)
        skip(nk*12); field('coords', nn*3)
        skip(nf*17+npart*54+nn*6)
        nodal_names = names(ns+nes, 81)
        for name in nodal_names[:ns]: field('nodal_scalar/'+name, nn)
        skip(nf*nes*4)
        for name in names(nv, 81): field('nodal_vector/'+name, nn*3)
        skip(nt*81+nf*nt*12)
        if flags[0]: skip((nf+nn)*4)
        if flags[1]: field('node_ids', nn); skip(nf*4)
        if flags[4]: skip(npart*12)
        assert flags[2]
        ne, parts, scalars, tensors = ints(4)
        field('connectivity', ne*8); field('active', ne, 1)
        ends = ints(parts); part_names = names(parts, 50)
        for name in names(scalars, 81): field('scalar/'+name, ne)
        for name in names(tensors, 81): field('tensor/'+name, ne*6)
        assert nn == 257104 and ne == 130218 and parts == 17 and ends[-1] == ne
        assert stream.tell() <= Path(path).stat().st_size
    return {'path': Path(path), 'time': stamp, 'nnode': nn, 'nelem': ne, 'ends': ends, 'part_names': part_names, 'ranges': ranges}


def array(info, name, count=None, start=0, dtype='>f4'):
    offset, size = info['ranges'][name]
    itemsize = np.dtype(dtype).itemsize
    count = size//itemsize-start if count is None else count
    assert 0 <= start*itemsize <= size and (start+count)*itemsize <= size
    with info['path'].open('rb') as stream:
        stream.seek(offset+start*itemsize)
        value = np.fromfile(stream, dtype=dtype, count=count)
    READ_BYTES['native_selected_arrays'] += count*itemsize
    assert len(value) == count
    return value


def same_ranges(left, right):
    selected = set(left['ranges']).intersection(right['ranges'])
    # Only valid IPs are shared across the100us output-card reduction.
    selected = {x for x in selected if not x.startswith('tensor/') or any('Point'+p in x for p in ('111', '112', '121', '211'))}
    checks = {}
    with left['path'].open('rb') as a, right['path'].open('rb') as b:
        for key in sorted(selected):
            ao, length = left['ranges'][key]; bo, other = right['ranges'][key]
            assert length == other
            a.seek(ao); b.seek(bo); equal = True
            for start in range(0, length, 1024**2):
                size = min(1024**2, length-start)
                equal &= a.read(size) == b.read(size)
                READ_BYTES['native_boundary_comparison'] += 2*size
            checks[key] = equal
    assert left['time'] == right['time'] and all(checks.values())
    return {'left': str(left['path'].relative_to(ROOT)), 'right': str(right['path'].relative_to(ROOT)), 'same_actual_time_s': left['time'], 'all_selected_native_fields_bitwise_equal': True, 'field_checks': checks, 'scope': 'Field bytes, not entire ANIM files/headers; includes geometry, IDs, topology, active flags, native scalars/GPS stress/Velocity and all4validIP stress/strain tensors.'}


def source_contract(path):
    ids, elements, parts, wall = [], Counter(), {}, []
    section = ''; body = []; element_id = None
    with Path(path).open() as stream:
        for line in stream:
            text = line.strip()
            if not text or text.startswith('#'): continue
            if text.startswith('/'):
                section = text; body = []; continue
            values = text.split()
            if section == '/NODE': ids.append(int(values[0]))
            elif section.startswith('/TETRA10/'):
                if len(values) == 1: element_id = int(values[0])
                else:
                    assert len(values) == 10 and element_id is not None
                    pid = int(section.split('/')[2]); elements[(pid, tuple(sorted(map(int, values[:4]))))] += 1
                    element_id = None
            elif section.startswith('/BRICK/'):
                assert len(values) == 9
                pid = int(section.split('/')[2]); elements[(pid, tuple(sorted(set(map(int, values[1:])))))] += 1
            elif section.startswith('/PART/'):
                body.append(text)
                if len(body) == 2: parts[int(section.split('/')[2])] = {'name': body[0], 'property': int(values[0]), 'material': int(values[1])}
            elif section == '/RWALL/PLANE/1': wall.append(text)
    assert len(ids) == 257099 and len(parts) == 17 and sum(elements.values()) == 130218
    point = list(map(float, wall[3].split())); second = list(map(float, wall[4].split()))
    direction = [b-a for a, b in zip(point, second)]; length = math.sqrt(sum(x*x for x in direction))
    return ids, elements, parts, [x/length for x in direction]


def history_check(summary, normal):
    paths = [BASE/'20us/Cell_Phone_DropT01.csv', BASE/'100us/Cell_Phone_DropT02.csv']
    paths += sorted((ROOT/'work/expanded-drop/500us-postprocess-history').rglob('Cell_Phone_DropT*.csv'))
    merged = {}
    for path in paths:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                values = {key: float(value) for key, value in row.items() if key and value}
                merged[values['time']] = values
    final_rows = []
    with (FINAL/'global-history-0-to500us.csv').open(newline='') as stream:
        for row in csv.DictReader(stream): final_rows.append({key: float(value) for key, value in row.items() if key and value})
    # Ignore unrelated converter leftovers outside the actual reported time interval.
    selected = [merged[t] for t in sorted(merged) if t <= final_rows[-1]['time']]
    assert selected == final_rows, 'Frozen combined CSV does not match source-converter CSV merge'
    values = []
    for row in final_rows:
        mass_kg = row['MASS']*1000
        velocity = sum(row[key]*n for key, n in zip(('X-MOMENTUM', 'Y-MOMENTUM', 'Z-MOMENTUM'), normal))/mass_kg
        energy = (sum(row[k] for k in ('KINETIC ENERGY', 'INTERNAL ENERGY', 'ROTATION ENERGY', 'SPRING ENERGY', 'CONTACT ENERGY', 'HOURGLASS ENERGY'))-row['EXTERNAL WORK'])*.001
        values.append((row['time'], velocity, energy))
    assert all(b[0] > a[0] for a, b in zip(values, values[1:]))
    outward = [x for x in values if x[1] > 0]
    got = {'history_samples': len(values), 'last_history_time_s': values[-1][0], 'history_initial_normal_COM_velocity_m_per_s': values[0][1], 'history_last_normal_COM_velocity_m_per_s': values[-1][1], 'any_outward_normal_COM_velocity': bool(outward), 'first_saved_outward_normal_COM_time_s': outward[0][0] if outward else None, 'initial_accounted_energy_J': values[0][2], 'last_accounted_energy_J': values[-1][2], 'accounted_energy_drift_percent': 100*(values[-1][2]/values[0][2]-1), 'last_kinetic_energy_J': final_rows[-1]['KINETIC ENERGY']*.001, 'last_internal_energy_J': final_rows[-1]['INTERNAL ENERGY']*.001, 'last_plastic_work_J': final_rows[-1]['PLASTIC WORK']*.001, 'last_external_work_J': final_rows[-1]['EXTERNAL WORK']*.001, 'last_added_mass_percent': final_rows[-1]['PERCENTAGE ADDED MASS']}
    for key, value in got.items():
        if key not in summary: continue
        expected = summary[key]
        assert value == expected if isinstance(value, (bool, type(None))) else math.isclose(value, expected, rel_tol=2e-12, abs_tol=2e-12), (key, value, expected)
    got.update(source_CSVs=[str(x.relative_to(ROOT)) for x in paths], source_CSV_merge_exact_match=True, normal_from_original_wall_card=normal, energy_definition='(KE+IE+rotation+spring+contact+hourglass-external_work)*0.001J. Plastic work is part of IE and is not added again.', rebound_scope='Saved actual global P/M normal sign, not local-node sign or proof of contact separation/full-event response.')
    return got


def main():
    global np
    assert not OUT.exists() and not MD.exists(), 'Audit outputs are exclusive'
    affinity = one_cpu(); guard('before_numpy_and_completed_evidence')
    freeze = read_json(FINAL/'artifact-freeze.json'); progress = read_json(FINAL/'job-progress.json')
    assert freeze['actual_FEM_completed_to_0_5ms'] and progress['status'] == 'completed'
    frozen_sha = digest(FINAL/'artifact-freeze.json', 'compact_evidence')
    import numpy as np
    checks = []
    for name, expected in freeze['files'].items():
        path = FINAL/name
        assert path.stat().st_size == expected['bytes'] and digest(path, 'compact_evidence') == expected['sha256'], name
        checks.append(name)
    summary = read_json(FINAL/'summary.json'); stats = read_json(FINAL/'frame-field-statistics.json')
    assert summary['completed_to_requested_time'] and summary['last_actual_frame_s'] >= .0005
    native_inventory = {}
    for run in ('20us-run', '100us-run', '500us-run', '500us-resume03-run'):
        for path in sorted((ROOT/'work/expanded-drop'/run).glob('Cell_Phone_DropA[0-9][0-9][0-9]')):
            with path.open('rb') as stream: magic, stamp = struct.unpack('>if', stream.read(8))
            assert magic == 0x542c
            native_inventory[stamp] = str(path.relative_to(ROOT))
    assert len(stats) == summary['saved_frames'] == len(native_inventory)
    assert {r['actual_time_s']: r['file'] for r in stats} == native_inventory
    assert all(b['actual_time_s'] > a['actual_time_s'] for a, b in zip(stats, stats[1:]))
    stdout = (FINAL/'engine.stdout.log').read_text()
    assert 'NORMAL TERMINATION' in stdout and 'USER BREAK' not in stdout
    cycles = int(re.findall(r'TOTAL NUMBER OF CYCLES\s*:\s*(\d+)', stdout)[-1])
    elapsed = float(re.findall(r'ELAPSED TIME\s*=\s*([0-9.]+)\s*s', stdout)[-1])
    progress_rows = [(int(a), float(b)) for a, b in re.findall(r'NC=\s*(\d+)\s+T=\s*([0-9.Ee+\-]+)', stdout)]
    assert progress_rows and cycles >= progress_rows[-1][0] and cycles-progress_rows[-1][0] < 1000
    metrics = read_json(FINAL/'resume03-execution-metrics.json'); assert metrics['exit_code'] == 0 and metrics['stopped_reason'] is None
    normal_contract = read_json(BASE/'contact-time-estimate.json')['plane_unit_normal']
    ids, source_elements, parts, normal = source_contract(ROOT/'work/expanded-drop/source/Cell_Phone_Drop_0000.rad')
    assert np.allclose(normal, normal_contract, rtol=0, atol=1e-15)
    history = history_check(summary, normal); guard('history_and_deck_complete')
    metadata_by_file = {row['file']: row for row in stats}
    final = layout(ROOT/summary['last_frame']['file'])
    node_ids = array(final, 'node_ids', dtype='>i4')
    conn = array(final, 'connectivity', dtype='>i4').reshape(final['nelem'], 8)
    assert np.array_equal(np.sort(node_ids[:257099]), np.sort(np.asarray(ids, dtype=np.int32)))
    assert conn.min() >= 0 and conn.max() < len(node_ids)
    physical_corner_indices = np.unique(conn)
    assert len(physical_corner_indices) == 76065 and np.array_equal(physical_corner_indices, np.arange(76065))
    remaining = source_elements.copy(); tetra = np.zeros(final['nelem'], dtype=bool); part_map = []
    start = 0
    for end, name in zip(final['ends'], final['part_names']):
        pid = int(name.lstrip('_').split(':')[0]); assert pid in parts
        part_map.append((start, end, pid))
        for i in range(start, end):
            corner = tuple(sorted(set(int(node_ids[index]) for index in conn[i])))
            # Original /BRICK/13 includes468 wedge-degenerate8-slot elements with6unique nodes.
            assert len(corner) in (4, 6, 8)
            tetra[i] = len(corner) == 4
            key = (pid, corner); assert remaining[key] > 0, ('source_connectivity_mismatch', i, key)
            remaining[key] -= 1
        start = end
    assert not any(remaining.values())
    topology = {'original_RAD_NODE_ID_set_exact_match': True, 'original_RAD_all130218_corner_connectivities_and_part_exact_match': True, 'source_comparison_definition': 'Multiset of(partID,sorted unique physical cornerNODE_IDs); original TETRA10 corner4 and BRICK8-slot wedges6/hexahedra8. Literal ordered8-slot bytes are compared separately between native ANIM frames.', 'source_BRICK_unique_corner_counts': {'8': 20150, '6': 468}, 'connectivity_index_base': 0, 'solid_part_count': 17, 'physical_source_nodes': 257099, 'corner_nodes': 76065, 'TETRA10_count': int(tetra.sum()), 'BRICK_count': int((~tetra).sum()), 'native_corner_indices_are_exact_prefix_0_to76064': True}
    del ids, source_elements, remaining, conn; gc.collect(); guard('source_ID_connectivity_complete')
    scalar_rows = []; previous = None; largest_decrement = 0
    for row in stats:
        guard('scalar_frame_'+str(row['actual_time_s']))
        info = layout(ROOT/row['file']); assert info['time'] == row['actual_time_s']
        vm = array(info, 'scalar/Von_Mises'); eps = array(info, 'scalar/Plastic_Strain'); active = array(info, 'active', dtype='u1')
        assert np.isfinite(vm).all() and np.isfinite(eps).all() and vm.min() >= 0 and eps.min() >= 0 and np.all(active == 1)
        for start, end, pid in part_map:
            if parts[pid]['material'] != 1: assert not np.any(eps[start:end] != 0), ('EPSP_in_nonplastic_material', pid)
        got = {'file': row['file'], 'time_s': info['time'], 'VM_max_MPa': float(vm.max()), 'EPSP_max': float(eps.max()), 'plastic_cells': int(np.count_nonzero(eps))}
        assert got['VM_max_MPa'] == row['native_cell_vm_max_MPa'] and got['EPSP_max'] == row['native_cell_epsp_max'] and got['plastic_cells'] == row['plastic_cells']
        if previous is not None:
            decrease = previous-eps; largest_decrement = max(largest_decrement, float(decrease.max()))
            assert not np.any(decrease > 0)
        previous = eps; scalar_rows.append(got)
    assert max(x['EPSP_max'] for x in scalar_rows) == summary['observed_maximum_EPSP']
    scalar_peak = max(scalar_rows, key=lambda x: x['VM_max_MPa'])
    assert scalar_peak['file'] == summary['observed_maximum_cell_vm_frame']['file']
    final_part_support = [{'part_id': pid, 'material': parts[pid]['material'], 'cells': end-start,
                           'plastic_cells': int(np.count_nonzero(eps[start:end])),
                           'EPSP_max': float(eps[start:end].max())} for start, end, pid in part_map]
    expected_part_support = {r['part_id']: r for r in summary['last_frame']['plastic_support']}
    for part in final_part_support:
        expected = expected_part_support[part['part_id']]
        assert part['cells'] == expected['cells'] and part['material'] == expected['material']
        assert part['plastic_cells'] == expected['plastic_cells'] and part['EPSP_max'] == expected['epsp_max']
    topology['final_part_plastic_support'] = final_part_support
    topology['all_saved_frames_nonzero_EPSP_only_in_original_material1'] = True
    strain_peak = max(stats, key=lambda x: x['total_strain_all_valid_IP_max_frobenius'])
    selected = list(dict.fromkeys((summary['last_frame']['file'], scalar_peak['file'], strain_peak['file'])))
    selected_stats = []
    for path in selected:
        guard('selected_validIP_frame')
        info = layout(ROOT/path); got = {'file': path, 'time_s': info['time'], 'file_sha256': digest(ROOT/path, 'selected_full_native_hash'), 'strain': {}, 'stress': {}}
        assert got['file_sha256'] == metadata_by_file[path]['sha256']
        for point in ('111', '112', '121', '211'):
            valid = np.ones(info['nelem'], dtype=bool) if point == '111' else tetra
            for prefix, output in (('Stra_', 'strain'), ('Strs_', 'stress')):
                name = next(k for k in info['ranges'] if k.startswith('tensor/'+prefix) and 'Point'+point in k)
                low = np.full(6, np.inf); high = -low; maximum = 0.; nonzero = 0; derived_vm = 0.
                for start in range(0, info['nelem'], CHUNK):
                    count = min(CHUNK, info['nelem']-start)
                    value = array(info, name, count*6, start*6).reshape(count, 6).astype(np.float64)
                    assert np.isfinite(value).all()
                    mask = valid[start:start+count]
                    if point != '111': assert not np.any(value[~mask] != 0), 'Nonphysical BRICK IP output is not zero'
                    value = value[mask]
                    if not len(value): continue
                    low = np.minimum(low, value.min(0)); high = np.maximum(high, value.max(0))
                    norm = np.sqrt((value[:, :3]**2).sum(1)+2*(value[:, 3:]**2).sum(1)); maximum = max(maximum, float(norm.max()))
                    nonzero += int(np.count_nonzero(np.any(value != 0, axis=1)))
                    if output == 'stress':
                        xx, yy, zz, xy, yz, xz = value.T
                        inv = np.sqrt(.5*((xx-yy)**2+(yy-zz)**2+(zz-xx)**2)+3*(xy*xy+yz*yz+xz*xz)); derived_vm = max(derived_vm, float(inv.max()))
                got[output][point] = {'physical_valid_cells': int(valid.sum()), 'min_6': low.tolist(), 'max_6': high.tolist(), 'max_frobenius': maximum, 'nonzero_cells': nonzero}
                if output == 'stress': got[output][point]['derived_IP_VM_max_MPa_distinct_from_native_cell_VM'] = derived_vm
                else:
                    expected = metadata_by_file[path]['valid_IP_strain'][point]
                    assert math.isclose(maximum, expected['max_frobenius'], rel_tol=2e-14, abs_tol=1e-15)
                    assert nonzero == expected['nonzero_cells']
        selected_stats.append(got)
    assert math.isclose(max(v['max_frobenius'] for g in selected_stats for v in g['strain'].values()), summary['observed_maximum_total_strain_all_valid_IP_frobenius'], rel_tol=2e-14)
    velocity = array(final, 'nodal_vector/Velocity').reshape(final['nnode'], 3)[physical_corner_indices].astype(np.float64)@np.asarray(normal)
    local = {'min_mm_s': float(velocity.min()), 'max_mm_s': float(velocity.max()), 'outward_corner_nodes': int(np.count_nonzero(velocity > 0))}
    assert math.isclose(local['min_mm_s'], summary['last_corner_normal_velocity']['min'], rel_tol=2e-12)
    assert math.isclose(local['max_mm_s'], summary['last_corner_normal_velocity']['max'], rel_tol=2e-12)
    assert local['outward_corner_nodes'] == summary['last_corner_normal_velocity']['outward_positive_nodes']
    baseline = layout(ROOT/'work/expanded-drop/20us-run/Cell_Phone_DropA001')
    for name in ('node_ids', 'connectivity'):
        assert np.array_equal(array(baseline, name, dtype='>i4'), array(final, name, dtype='>i4'))
    boundary_data = read_json(FINAL/'final-checkpoint-boundary-bitwise.json')
    if boundary_data.get('comparison_performed') is False:
        assert boundary_data['status'] == 'NOT_VERIFIED_OPTIONAL_RELOAD'
        attempt1 = read_json(FINAL/'final-boundary-engine-metrics.json')
        attempt2 = read_json(FINAL/'final-boundary02-engine-metrics.json')
        failed_stdout = (FINAL/'failed-final-boundary01-engine.stdout.log').read_text()
        assert attempt1['exit_code'] == -1073741819 and 'NORMAL TERMINATION' not in failed_stdout
        assert attempt2['no_engine_started'] and attempt2['status'] == 'deferred_for_memory_preflight'
        final_boundary = {'status': 'NOT_VERIFIED_OPTIONAL_RELOAD', 'independent_field_comparison_performed': False,
                          'native_field_bitwise_identity_claimed': False, 'worker_record': boundary_data,
                          'attempt1_execution_metadata': attempt1, 'attempt2_preflight_metadata': attempt2,
                          'attempt1_stdout_normal_termination_absent': True,
                          'scope': 'Optional final-checkpoint reload is unverified; main0.5ms normal completion is assessed separately.'}
    else:
        final_boundary = same_ranges(layout(ROOT/boundary_data['source_frame']), layout(ROOT/boundary_data['reloaded_frame']))
    initial500 = min((layout(p) for p in (ROOT/'work/expanded-drop/500us-run').glob('Cell_Phone_DropA[0-9][0-9][0-9]')), key=lambda x: x['time'])
    old100 = layout(ROOT/'work/expanded-drop/100us-run/Cell_Phone_DropA021')
    initial_boundary = same_ranges(old100, initial500)
    guard('before_frozen_cache_stream_hash')
    ready_path = BASE/'doe/canonical/dataset_READY_evidence.json'; ready = read_json(ready_path)
    cache_checks = []
    for relative, expected in ready['file_hashes'].items():
        guard('frozen20us_file_hash')
        path = ready_path.parent/relative
        actual = digest(path, 'immutable20us_cache_hash')
        assert actual == expected['sha256'] and path.stat().st_size == expected['bytes']
        cache_checks.append({'file': relative, 'bytes': expected['bytes'], 'sha256': actual})
    direct_source_hash_checks = []
    inherited_checkpoint_checks = []
    for relative, expected in freeze['source_and_checkpoint_hash_checks'].items():
        if relative.endswith('.rst'):
            inherited_checkpoint_checks.append({'path': relative, 'worker_frozen_sha256': expected['sha256'], 'independently_rehashed': False, 'reason': 'Large RST bytes not reread; same-time actual native-field reload independently compared instead.'})
            continue
        path = ROOT/relative; actual = digest(path, 'source_and_binary_hash')
        assert actual == expected['sha256'] == expected['expected_sha256']
        direct_source_hash_checks.append({'path': relative, 'sha256': actual})
    assert digest(FINAL/'artifact-freeze.json', 'compact_evidence') == frozen_sha
    final_resource = guard('audit_complete')
    report = {'status': 'PASS_SCOPED_INDEPENDENT_NUMERIC_AUDIT', 'completed_utc': now(), 'frozen_utc': freeze['freeze_utc'], 'artifact_freeze_sha256': frozen_sha, 'compact_frozen_files_hash_verified': checks, 'CPU_affinity_single_bit': affinity, 'numpy_version': np.__version__, 'ML_or_GPU_imports_or_execution': False, 'normal_termination': {'stdout_normal': True, 'actual_exit_code': 0, 'engine_total_cycles': cycles, 'engine_elapsed_seconds': elapsed, 'first_last_progress_NC_time': [progress_rows[0], progress_rows[-1]], 'last_actual_ANIM_time_s': final['time'], 'requested_stop_s': .0005}, 'history_independent_recalculation': history, 'source_ID_topology': topology, 'all_saved_native_scalar_rechecks': scalar_rows, 'all_saved_active_solids': 130218, 'EPSP_monotonic_all_saved_native_scalars': {'passed': True, 'maximum_positive_decrement': max(0, largest_decrement)}, 'native_VM_snapshot_peak_independently_selected': scalar_peak, 'selected_native_stress_strain_rechecks': selected_stats, 'local_corner_normal_velocity': local, 'restart_boundaries': {'100us_to_first500us': initial_boundary, '117us_guard_to_resume': {'same_time_native_ANIM_available': False, 'field_bitwise_equivalence_claimed': False, 'evidence': '500us-final/source-control-and-restart-audit.json', 'limitation': 'Unequal saved times; source/checkpoint provenance and forward progress do not prove field-byte equality at this interruption.'}, 'final_checkpoint_same_time_reload': final_boundary}, 'frozen20us_AI_cache_independent_hash_checks': cache_checks, 'direct_source_binary_hash_checks': direct_source_hash_checks, 'large_RST_hash_checks_inherited_not_repeated': inherited_checkpoint_checks, 'valid_tensor_definition': 'Global tensor components xx,yy,zz,xy,yz,xz. TETRA10 has4validIPs111/112/121/211; BRICK uses111. ANIM shears already tensor epsilon; Frob diagonal squares+2offdiagonal squares. Native cell VM, derived IP stress VM, GPS nodal VM and projected strain are distinct supports.', 'AI_scope': 'Single-center FEM continuation to0.5ms only. Frozen nine-condition AI labels remain nominal2..20us. No AI0.5ms prediction/evaluation.', 'limitations': ['All stored scalar frames were independently reread; valid-IP tensor statistics were independently recomputed only for final/observedVMpeak/metadata-selectedstrainpeak frames, not every complete tensor frame.', 'The reported observed strain peak is verified at its candidate frame and against frozen row maxima, not independently proven globally over every native tensor frame or continuous time.', 'No new constitutive/mesh/time-step convergence study, contact-separation criterion, or product failure assessment.', 'Restart byte comparisons concern represented native arrays at identical time, not entire ANIM binary headers or new117us boundary.', 'Large RST hashes remain worker provenance; compact freeze, original deck/engine and all33canonical20us files were independently hashed.'], 'read_bytes_by_category': dict(READ_BYTES), 'resources': {'samples': RESOURCES, 'final': final_resource, 'host_minimum_available_bytes': min(x['host_available_bytes'] for x in RESOURCES), 'process_peak_rss_bytes': max(x['peak_rss_bytes'] for x in RESOURCES)}, 'helper_sha256': digest(Path(__file__), 'audit_helper')}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rebound = '있음' if history['any_outward_normal_COM_velocity'] else '없음'
    body = f'''# 중앙 조건 0.5ms FEM 독립 감사

판정: **범위를 한정한 독립 수치검산 PASS**. 완료시각 {report['completed_utc']}, 원결과 동결 {report['frozen_utc']}. GPU/ML을 실행하지 않았으며 CPU affinity1개, 최대 RSS {report['resources']['process_peak_rss_bytes']/1024**2:.2f}MiB에서 검사했다.

실제 stdout의 NORMAL TERMINATION, exit0, {cycles} cycles, engine {elapsed:.2f}s와 마지막 native ANIM {final['time']*1e6:.9f}µs를 확인했다. 저장된 원 TH converter CSV를 다시 합쳐 최종 history와 모든 행이 같음을 확인하고, 실제 P/M을 원 RAD wall normal에 투영했다. 마지막 COM normal 속도는 {history['history_last_normal_COM_velocity_m_per_s']:.9f}m/s이며 저장 history의 양의 부호 전환은 **{rebound}**다. 이는 contact separation이나 전체 낙하 요구의 완결 판정과 다르다.

Energy는 KE+IE+rotation+spring+contact+hourglass−external work에0.001J 변환을 적용해 다시 계산했다. Plastic work를 IE에 중복 가산하지 않았다. 원 t0 대비 마지막 drift는 {history['accounted_energy_drift_percent']:.9f}%, 마지막 accounted energy는 {history['last_accounted_energy_J']:.12f}J다.

모든 저장 ANIM에서 native cell VM/EPSP scalar와130,218 active flags만 제한적으로 다시 읽었다. 관측 native VM 최대는 {scalar_peak['VM_max_MPa']:.9f}MPa, 실제 {scalar_peak['time_s']*1e6:.9f}µs이며 EPSP의 저장 시점 간 감소는 없었다. 최종·VM피크·strain피크 후보에서 실제 stress/strain4validIP tensor를 chunk로 별도 재계산했다. 다른 모든 시점의 전체 tensor를 재스캔하지 않았으므로 strain 연속시간/전수 native peak 독립 증명으로 확대하지 않는다.

원 RAD257,099 node ID와130,218 corner connectivity·17part를 실제 최종 ANIM과 대조했다. 물리 corner76,065개가 native 앞부분 support라는 것도 연결에서 직접 확인했다. 원20µs ANIM과 최종 ID/연결은 동일하다. 모든 원 canonical20µs 동결파일 {len(cache_checks)}개의 실제 SHA를 다시 계산해 보존을 확인했다.

재시작 근거는 분리했다. 100µs→첫500µs의 같은 시각 native field bytes를 다시 비교해 일치했고, **117µs guard stop 경계는 같은 시각 ANIM이 없어 bitwise 일치를 주장하지 않는다**. 최종0.5ms checkpoint reload는 별도의 같은 시각 native field bytes 비교가 통과했다. 큰 RST 파일 hash는 worker 동결기록을 인용하고 중복 재독립 hash하지 않았다.

이것은 **단일 중앙 조건의0.5ms FEM**이며0.5ms AI 검증이 아니다. AI9조건 target은 여전히 명목2..20µs다. Native cell scalar VM, IP응력에서 계산한 VM, GPS절점 VM, 투영된 strain의 지지점과 정의를 섞지 않는다. 자세한 직접 계산값·동결 SHA·자원·검산하지 않은 범위는 [감사 JSON](500us-final-review.json)에 있다.
'''
    if final_boundary.get('status') == 'NOT_VERIFIED_OPTIONAL_RELOAD':
        body = body.replace('최종0.5ms checkpoint reload는 별도의 같은 시각 native field bytes 비교가 통과했다.',
                            '최종0.5ms checkpoint의 추가 reload는 **NOT_VERIFIED**다. 첫 시도는 access violation이었고 재시도는 RAM preflight에서 시작하지 못했다. 같은 시각 field 비교를 수행하거나 성공했다고 주장하지 않는다. 이 항목은 메인0.5ms 정상완주와 분리한다.')
    assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'physicsnemo'))
    MD.write_text(body, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'json': str(OUT), 'md': str(MD), 'VM_peak': scalar_peak, 'history_last_normal_COM_m_s': history['history_last_normal_COM_velocity_m_per_s'], 'resource': final_resource, 'read_bytes': dict(READ_BYTES)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
