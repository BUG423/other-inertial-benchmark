#!/usr/bin/env python3
"""Audit and execute the dynamic benchmark on immutable AutoDL source snapshots.

Coverage keeps every active profile x every accepted dataset x the same three
once-randomized seeds. Qualification failures remain visible. Framework imports
and execution are server-only; learned training requires actual CUDA and an
explicit authorization flag. The existing benchmark engine owns training.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import signal
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

SERVER_ROOT = Path('/root/autodl-tmp/other-inertial-benchmark-v1')
SNAPSHOTS = SERVER_ROOT / '.local/source-snapshots'
LOCK_ROOT = Path('/root/autodl-tmp/inertial-benchmark/.local/campaigns')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def source_identity(root):
    root = Path(root).resolve()
    if not root.is_relative_to(SNAPSHOTS) or not (root / 'source_manifest.json').is_file():
        raise RuntimeError('Full queue requires an immutable AutoDL source snapshot')
    manifest = load(root / 'source_manifest.json')
    files = {**manifest.get('files', {}), **manifest.get('operational_acceptance_helpers', {})}
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise RuntimeError(f'Immutable managed source mismatch: {name}')
    return {'source_root': str(root), 'source_manifest_sha256': sha(root / 'source_manifest.json'),
            'deployment_id': manifest['deployment_id'], 'files': files}


def seed_record(path):
    record = load(path)
    seeds = record.get('seeds', [])
    if (record.get('schema') != 'ib.benchmark.random-seeds.v1'
            or record.get('entropy_source') != 'operating_system_random_via_python_secrets'
            or record.get('sample_once') is not True or len(seeds) != 3
            or len(set(seeds)) != 3
            or any(type(n) is not int or not 0 <= n < 2 ** 31 for n in seeds)):
        raise RuntimeError('Formal queue requires three unique once-randomized OS-entropy seeds')
    return seeds, sha(path)


def accepted_datasets(path):
    record = load(path)
    rows = record.get('datasets', [])
    result, independent = [], set()
    for row in rows:
        if row.get('accepted') is not True:
            continue
        native = Path(row['root'])
        manifest = load(native / 'dataset.json')
        if not manifest.get('sequences') or manifest.get('engineering_only'):
            raise RuntimeError(f'Nonformal or empty native in accepted registry: {row["name"]}')
        if manifest.get('fingerprint') != row['fingerprint']:
            raise RuntimeError(f'Accepted native fingerprint changed: {row["name"]}')
        source = row.get('independent_source_id', row['name'])
        if source in independent:
            raise RuntimeError(f'Dataset aliases cannot increase independent source count: {source}')
        independent.add(source)
        result.append({**row, 'root': str(native), 'manifest_sha256': sha(native / 'dataset.json'),
                       'split_sha256': {p.name: sha(p) for p in sorted((native / 'splits').glob('*.txt'))},
                       'independent_source_id': source})
    if not result:
        raise RuntimeError('No standardized accepted formal datasets in registry')
    return result


@contextmanager
def resource_lock(kind, inherited_fd=None):
    path = LOCK_ROOT / ('gpu-0.lock' if kind == 'gpu' else 'cpu-resource.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    if inherited_fd is not None:
        if kind != 'gpu' or Path(os.readlink(f'/proc/self/fd/{inherited_fd}')).resolve() != path.resolve():
            raise RuntimeError('Invalid inherited GPU resource lease')
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield inherited_fd
        return
    with path.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle.fileno()


def framework(root):
    source_identity(root)
    import inertial_benchmark
    package = Path(inertial_benchmark.__file__).resolve()
    if not package.is_relative_to(Path(root).resolve() / 'src'):
        raise RuntimeError(f'Framework import escaped immutable snapshot: {package}')
    import torch
    torch.set_num_threads(2)
    return torch


def cuda_gate(args):
    if not args.gpu_authorized:
        raise RuntimeError('GPU execution requires explicit authorization; no CPU fallback')
    torch = framework(args.source_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one actual CUDA device; no CPU fallback')
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    return torch, {'device': 'cuda:0', 'name': torch.cuda.get_device_name(0),
                   'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                   'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}


def recipe(path, source_sha=None, hardware=None):
    policy = load(path)
    if policy.get('status') != 'frozen' or not policy.get('capacity_receipt_sha256'):
        raise RuntimeError('Formal training policy must be frozen against actual capacity evidence')
    overrides = policy['overrides']
    required = ('epochs', 'batch', 'optimizer', 'lr', 'scheduler', 'weight_decay', 'fitness',
                'fitness_stat', 'val_interval', 'patience', 'min_epochs', 'train_windows_budget', 'val_batch',
                'amp', 'cache', 'workers', 'gap_windows', 'save_predictions', 'plots', 'max_plots',
                'stride', 'eval_stride')
    if any(key not in overrides for key in required):
        raise RuntimeError('Frozen policy is missing common fairness rules')
    if overrides.get('recipe', 'unified') != 'unified' or overrides.get('loss') is not None:
        raise RuntimeError('Main recipe must preserve scientific model loss under unified training')
    if int(overrides['train_windows_budget']) != 0:
        raise RuntimeError('Primary protocol requires common fixed epochs; equal-window budget is a separate protocol')
    if int(overrides['min_epochs']) < 11 or int(overrides['epochs']) < int(overrides['min_epochs']):
        raise RuntimeError('Common primary recipe must reach the latest scientific phase (zero-based epoch10)')
    if overrides['save_predictions'] is not False:
        raise RuntimeError('Full formal matrix requires common save_predictions=false and separate bounded mechanism replay artifacts')
    capacity_file = policy.get('capacity_receipt')
    if not capacity_file or sha(capacity_file) != policy['capacity_receipt_sha256']:
        raise RuntimeError('Frozen recipe capacity receipt missing or changed')
    from inertial_benchmark.engine.benchmark import verify_capacity_policy
    verify_capacity_policy(policy, source_sha, hardware)
    return policy, sha(path)


def inventory(args):
    framework(args.source_root)
    from inertial_benchmark.cfg import INPUT_KEYS, get_cfg, load_model_cfg
    from inertial_benchmark.engine.budget import is_sequence_model
    seeds, seeds_sha = seed_record(args.seeds)
    source = source_identity(args.source_root)
    datasets = accepted_datasets(args.registry)
    profiles = sorted(p.stem for p in (Path(args.source_root) / 'src/inertial_benchmark/cfg/models').glob('*.yaml'))
    rows = []
    for profile in profiles:
        sampling = {key: value for key, value in {'stride': getattr(args, 'sampling_stride', None),
                    'eval_stride': getattr(args, 'eval_stride', None)}.items() if value is not None}
        cfg = get_cfg({'model': profile, 'recipe': 'unified', **sampling})
        description = load_model_cfg(profile)
        classic = is_sequence_model(profile)
        privileged = cfg.orientation == 'reference' or 'init_velocity' in cfg.extra_inputs
        for data in datasets:
            has_train = bool((Path(data['root']) / 'splits/train.txt').is_file()
                             and (Path(data['root']) / 'splits/train.txt').read_text().strip())
            has_val = bool((Path(data['root']) / 'splits/val.txt').is_file()
                           and (Path(data['root']) / 'splits/val.txt').read_text().strip())
            domain = data.get('domain', 'human' if data['name'] not in ('euroc', 'blackbird') else 'uav')
            for seed in seeds:
                row = {'profile': profile, 'architecture': description['arch'], 'dataset': data['name'],
                       'data_root': data['root'], 'domain': domain, 'seed': seed, 'classic': classic,
                       'dims': cfg.dims, 'main_xyz': cfg.dims == 3, 'native_xyz': cfg.dims == 3,
                       'learned_cuda_training': not classic,
                       'input_track': 'reference_oracle' if privileged else 'deployment',
                       'scientific_input_spec': {k: getattr(cfg, k) for k in INPUT_KEYS},
                       'sampling_spec': {'stride': int(cfg.stride), 'eval_stride': int(cfg.eval_stride)},
                       'scientific_loss': description.get('loss'),
                       'profile_config_sha256': sha(Path(args.source_root) / f'src/inertial_benchmark/cfg/models/{profile}.yaml'),
                       'dataset_fingerprint': data['fingerprint'], 'dataset_manifest_sha256': data['manifest_sha256'],
                       'split_sha256': data['split_sha256'], 'seeds_record_sha256': seeds_sha,
                       'state': 'pending_qualification', 'trainable': has_train and has_val,
                       'checkpoint_training_source': (None if has_train else
                            ('euroc' if data['name'] == 'blackbird' else data.get('checkpoint_training_source'))),
                       'reasons': []}
                if not has_train:
                    row.update(state='test_only_checkpoint_required')
                    row['reasons'].append('test-only; explicitly declared same-profile/input-track/seed training checkpoint required')
                elif not has_val:
                    row.update(state='unsupported_missing_validation')
                    row['reasons'].append('No nonempty independent validation split')
                row['id'] = object_sha({k: row[k] for k in ('profile', 'dataset', 'seed', 'input_track')})[:20]
                rows.append(row)
    report = {'schema': 'ib.dynamic-matrix.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source': source, 'seeds': seeds, 'seeds_record_sha256': seeds_sha,
              'profiles': profiles, 'datasets': datasets, 'coverage': rows,
              'profile_count': len(profiles), 'independent_dataset_count': len(datasets),
              'base_coverage_rows': len(profiles) * len(datasets) * 3,
              'minimum_architectures_for_delivery': 31, 'minimum_datasets_for_delivery': 10,
              'architecture_count': len({r['architecture'] for r in rows}),
              'main_xyz_training_coverage_rows': sum(r['main_xyz'] and r['trainable'] and not r['classic'] for r in rows),
              'main_xyz_calibration_coverage_rows': sum(r['main_xyz'] and r['trainable'] and r['classic'] for r in rows),
              'scope_is_dynamic': True, 'coverage_completed': False}
    save(Path(args.output) / 'coverage.json', report)
    print(json.dumps({k: report[k] for k in ('profile_count', 'architecture_count', 'independent_dataset_count',
                                           'base_coverage_rows', 'main_xyz_training_coverage_rows')}), flush=True)
    return report


def qualify(args):
    torch = framework(args.source_root)
    import numpy as np
    from inertial_benchmark.cfg import get_cfg
    from inertial_benchmark.data.build import build_dataset
    from inertial_benchmark.data.manifest import resolve_dataset
    from inertial_benchmark.data.views import read_window
    report = load(Path(args.output) / 'coverage.json')
    cache, proof_cache = {}, {}
    taskview_files = source_identity(args.source_root)['files']
    implementation = object_sha({name: value for name, value in taskview_files.items()
        if name.startswith('src/inertial_benchmark/data/') and '/converters/' not in name
        or name == 'src/inertial_benchmark/utils/geometry.py'})
    with resource_lock('cpu'):
        for row in report['coverage']:
            if args.profiles and row['profile'] not in args.profiles.split(','): continue
            if args.datasets and row['dataset'] not in args.datasets.split(','): continue
            cfg = get_cfg({'model': row['profile'], 'data': row['data_root'], 'recipe': 'unified',
                           'device': 'cpu', 'cache': False, 'workers': 0, 'seed': 0,
                           **row.get('sampling_spec', {})})
            key = object_sha({'input_spec': row['scientific_input_spec'],
                              'augment': cfg.augment, 'stride': cfg.stride, 'eval_stride': cfg.eval_stride,
                              'native_fingerprint': row['dataset_fingerprint'], 'implementation': implementation})
            cache_path = Path(args.output) / 'taskview-cache' / f'{key}.json'
            if key not in cache and cache_path.exists(): cache[key] = load(cache_path)
            if key not in cache:
                result = {'status': 'passed', 'splits': {}, 'training_windows': 0,
                          'taskview_identity': key, 'origin_profile': row['profile']}
                try:
                    spec = resolve_dataset(row['data_root'])
                    splits = list(dict.fromkeys(['train', 'val'] +
                                 [s for s in spec.test_splits if not s.startswith('official_')]))
                    for split in splits:
                        if not spec.has_split(split) or not spec.sequence_ids(split):
                            result['splits'][split] = {'status': 'not_applicable_missing_split'}
                            continue
                        ds = build_dataset(cfg, split, training=(split == 'train'), spec=spec)
                        if not len(ds):
                            raise RuntimeError(f'No genuine valid {split} scientific windows')
                        k, start = (int(v) for v in ds.index[0])
                        sample = ds.load_window(k, start)
                        if not np.isfinite(sample['imu']).all() or not np.isfinite(sample['target']).all():
                            raise RuntimeError(f'Nonfinite actual native window in {split}')
                        lz = ds.lazy[k]
                        xyz = read_window(lz.handle(), start, replace(ds.view, dims=3), lz.yaw_offset, lz.has_velocity)
                        np.testing.assert_array_equal(sample['target'], xyz['target'][..., :int(cfg.dims)])
                        real = ds[0]
                        if not torch.isfinite(real['imu']).all() or not torch.isfinite(real['target']).all():
                            raise RuntimeError('Declared training augmentation yielded nonfinite native target')
                        result['splits'][split] = {'status': 'passed', 'valid_windows': len(ds),
                            'sequence_id': lz.sequence_id, 'start': start, 'imu_shape': list(real['imu'].shape),
                            'target_shape': list(real['target'].shape), 'native_xyz_and_xy_slice_exact': True,
                            'extra_shapes': {n: list(v.shape) for n, v in real.get('extra', {}).items()},
                            'raw_target_sha256': hashlib.sha256(sample['target'].tobytes()).hexdigest()}
                        if split == 'train': result['training_windows'] = len(ds)
                        del ds
                        gc.collect()
                except Exception as exc:
                    result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                cache[key] = result
                save(cache_path, result)
                save(Path(args.output) / 'taskviews' / f'{row["profile"]}__{row["dataset"]}.json', result)
            row['taskview_receipt'] = {**cache[key], 'reused_from': (
                cache[key]['origin_profile'] if cache[key]['origin_profile'] != row['profile'] else None)}
            if cache[key]['status'] != 'passed':
                row['state'] = 'unsupported_taskview'
                row['reasons'].append(cache[key]['error'])
            elif row['trainable']:
                row['state'] = 'awaiting_model_cpu_acceptance'
            if args.cpu_receipts:
                try:
                    if row['profile'] not in proof_cache:
                        proof_cache[row['profile']] = cpu_proofs(args, row['profile'])
                    proof = proof_cache[row['profile']]
                    row['model_cpu_accepted'] = True
                    row['model_cpu_receipt_sha256'] = proof['sha256']
                except Exception as exc:
                    row['model_cpu_accepted'] = False
                    row['model_cpu_pending_reason'] = f'{type(exc).__name__}: {exc}'
            save(Path(args.output) / 'coverage.json', report)
    return report


def cpu_proofs(args, profile):
    for directory in args.cpu_receipts:
        path = Path(directory) / profile / 'receipt.json'
        if path.is_file():
            result = load(path)
            if result.get('status') == 'passed' and result.get('standard_NIO_Trainer') is True:
                current = source_identity(args.source_root)
                previous_sha = result.get('source_manifest_sha256')
                manifests = [Path(args.source_root) / 'source_manifest.json']
                manifests.extend((SERVER_ROOT / '.local/deployments').glob('*/manifest.json'))
                previous = next((load(p) for p in manifests if sha(p) == previous_sha), None)
                if previous is None:
                    raise RuntimeError('CPU proof historical managed source manifest unavailable')
                prefixes = tuple(f'src/inertial_benchmark/{name}/' for name in
                                 ('models', 'data', 'nn', 'cfg', 'engine', 'utils', 'metrics'))
                def science(files):
                    return {name: value for name, value in files.items()
                            if name.startswith(prefixes) or name == 'src/inertial_benchmark/__init__.py'}
                previous_files = {**previous.get('files', {}),
                                  **previous.get('operational_acceptance_helpers', {})}
                before, after = science(previous_files), science(current['files'])
                if before != after:
                    changed = {name: {'before': before.get(name), 'after': after.get(name)}
                               for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
                    verified_bridge = None
                    for bridge_path in args.science_bridges:
                        bridge = load(bridge_path)
                        if (bridge.get('schema') == 'ib.scientific-source-bridge.v1'
                                and bridge.get('approved_for_model_cpu_reuse') is True
                                and bridge.get('status') == 'passed'
                                and bridge.get('source_before') == previous_sha
                                and bridge.get('source_after') == current['source_manifest_sha256']
                                and bridge.get('changed_scientific_files') == changed
                                and bridge.get('reason') and bridge.get('targeted_test_receipts')):
                            if all(sha(item['path']) == item['sha256'] and load(item['path']).get('exit_code') == 0
                                   for item in bridge['targeted_test_receipts']):
                                verified_bridge = {'path': str(bridge_path), 'sha256': sha(bridge_path)}
                                break
                    if not verified_bridge:
                        raise RuntimeError('CPU science changed without an exact approved source bridge and actual targeted tests')
                else:
                    verified_bridge = None
                expected_env = result.get('environment_identity')
                environment = {'python_version': __import__('sys').version.split()[0],
                               'package_versions': {n: importlib.metadata.version(n)
                                 for n in ('torch', 'numpy', 'h5py', 'scipy', 'pandas')}}
                if expected_env != environment:
                    raise RuntimeError('CPU proof actual Python/core dependency environment changed')
                return {'path': str(path), 'sha256': sha(path), 'record': result,
                        'science_and_environment_bridge_verified': True, 'explicit_science_bridge': verified_bridge,
                        'historical_source_manifest_sha256': previous_sha,
                        'current_source_manifest_sha256': current['source_manifest_sha256']}
    raise RuntimeError(f'No genuine standard NIO CPU acceptance receipt for {profile}')


def _cuda_batch(native):
    return {key: ({name: value.to('cuda:0', non_blocking=True) for name, value in item.items()}
                  if isinstance(item, dict) else item.to('cuda:0', non_blocking=True))
            for key, item in native.items()}


def _gpu_measure(torch, row, batch):
    """Time different real DataLoader batches, including native/cache reads and transfer."""
    from inertial_benchmark.cfg import get_cfg
    from inertial_benchmark.data.build import build_dataset, build_dataloader
    from inertial_benchmark.engine.predictor import load_model
    cfg = get_cfg({'model': row['profile'], 'data': row['data_root'], 'recipe': 'unified',
                   'device': '0', 'batch': batch, 'workers': 2, 'cache': True,
                   'optimizer': 'adamw', 'lr': .001, 'weight_decay': .0001, 'seed': 0,
                   **row.get('sampling_spec', {})})
    model = optimizer = ds = loader = iterator = native = out = loss = val_set = None
    begun = time.monotonic()
    observed_memory = 0
    try:
        model = load_model(cfg, torch.device('cuda:0')).train()
        ds = build_dataset(cfg, 'train', training=True)
        loader = build_dataloader(ds, batch, False, workers=2, pin_memory=True, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        iterator = iter(loader)
        cold_setup = time.monotonic() - begun
        torch.cuda.reset_peak_memory_stats()
        pipeline, compute, fetch = [], [], []
        for step in range(6):
            torch.cuda.synchronize(); start = time.monotonic()
            host = next(iterator)
            if host['imu'].shape[0] != batch:
                raise RuntimeError('insufficient real independent batches at requested common batch')
            native = _cuda_batch(host)
            torch.cuda.synchronize(); fetched = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            out = model(native['imu'], native['extra']) if native.get('extra') else model(native['imu'])
            loss, _ = model.loss(out, native, max(int(cfg.loss_switch_epoch), 10))
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite native-target GPU loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            observed_memory = max(observed_memory, int(Path('/sys/fs/cgroup/memory.current').read_text()))
            elapsed = time.monotonic() - start
            if step >= 2:
                pipeline.append(elapsed); compute.append(time.monotonic() - fetched); fetch.append(fetched - start)
            del host, native, out, loss
            native = out = loss = None
        train_peak = torch.cuda.max_memory_allocated()
        reserve_peak = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        del iterator, loader, ds, optimizer
        iterator = loader = ds = optimizer = None
        gc.collect(); torch.cuda.empty_cache()
        model.eval()
        val_set = build_dataset(cfg, 'val', training=False)
        loader = build_dataloader(val_set, batch, False, workers=2, pin_memory=True)
        iterator = iter(loader)
        validation = []
        with torch.no_grad():
            for _ in range(min(4, len(loader))):
                torch.cuda.synchronize(); start = time.monotonic()
                native = _cuda_batch(next(iterator))
                out = model(native['imu'], native['extra']) if native.get('extra') else model(native['imu'])
                loss, _ = model.loss(out, native, max(int(cfg.loss_switch_epoch), 10))
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite real GPU validation loss')
                torch.cuda.synchronize(); validation.append(time.monotonic() - start)
                del native, out, loss
                native = out = loss = None
        median = lambda values: sorted(values)[len(values) // 2]
        utilization = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return {'batch': batch, 'status': 'passed', 'optimizer_updates': 6,
                'independent_native_batches': True, 'timing_scope': 'DataLoader+augmentation+transfer+forward+loss+backward+AdamW',
                'cache': True, 'workers': 2, 'cold_dataset_setup_s': cold_setup,
                'median_pipeline_step_s': median(pipeline), 'median_compute_step_s': median(compute),
                'median_fetch_transfer_s': median(fetch), 'pipeline_windows_per_s': batch / median(pipeline),
                'median_validation_batch_s': median(validation), 'peak_allocated_bytes': train_peak,
                'peak_reserved_bytes': reserve_peak, 'driver_free_bytes': free,
                'peak_cgroup_memory_bytes_observed': observed_memory,
                'nvidia_utilization_sample': utilization.stdout.strip(),
                'validation_batches': len(validation)}
    finally:
        del model, optimizer, ds, val_set, loader, iterator, native, out, loss
        gc.collect(); torch.cuda.empty_cache()


def _gpu_lifecycle(args, torch, row, batch, gap_windows):
    """Use standard NIO for true optimizer/validation/predict/checkpoint lifecycle."""
    import importlib.util
    import numpy as np
    from inertial_benchmark import NIO
    from inertial_benchmark.cfg import get_cfg
    from inertial_benchmark.data.build import build_dataset
    from inertial_benchmark.utils.torch_utils import _CKPT_CACHE
    from inertial_benchmark.engine.predictor import load_model
    helper_path = Path(args.source_root) / 'tools/ops_v1/all_model_cpu.py'
    helper_spec = importlib.util.spec_from_file_location('ib_cpu_fixture_helper', helper_path)
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    cfg = get_cfg({'model': row['profile'], 'recipe': 'unified'})
    output = Path(args.output) / 'gpu-lifecycle' / row['profile'] / f'batch-{batch}-gap-{gap_windows}'
    fixture, provenance = helper.make_fixture(SERVER_ROOT, output, row['profile'], cfg, 'ridi',
        train_windows=2 * batch + 1, val_windows=max(batch, gap_windows), test_windows=max(batch, 2))
    overrides = {'data': str(fixture), 'device': '0', 'epochs': 1, 'min_epochs': 0,
                 'batch': batch, 'val_batch': batch, 'workers': 2, 'cache': True, 'amp': False,
                 'recipe': 'unified', 'metric_dims': int(cfg.dims), 'seed': 0,
                 'optimizer': 'adamw', 'lr': .001, 'weight_decay': .0001, 'gap_windows': gap_windows,
                 'efficiency': False, 'plots': False, 'save_predictions': True,
                 'project': str(output / 'runs'), 'name': 'actual-cuda-engineering', 'exist_ok': True}
    initial, updates = {}, [0]
    nio = trainer = reload = before = None
    legacy = {'status': 'not_requested', 'requested_windows': 1024}
    try:
        nio = NIO(row['profile'], metric_dims=int(cfg.dims), device='0')
        def hashes(model):
            return {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                    for name, p in model.named_parameters() if p.requires_grad}
        nio.add_callback('on_pretrain_routine_end', lambda trainer: initial.update(hashes(trainer.model)))
        nio.add_callback('on_train_batch_end', lambda trainer: updates.__setitem__(0, updates[0] + 1))
        nio.train(**overrides)
        trainer = nio.trainer
        changed = sum(initial[name] != value for name, value in hashes(nio.model).items())
        if updates[0] < 2 or not changed: raise RuntimeError('GPU lifecycle has no real two-update weight change')
        metrics = load(trainer.save_dir / 'metrics.json')
        gap = metrics.get('train_eval_gap')
        if not gap or len(gap.get('modes', {})) != 4 or gap.get('config', {}).get('mc_samples') != 8:
            raise RuntimeError('common diagnostic four-mode MC8 lifecycle missing')
        if args.probe_legacy_gap and row['profile'] in ('ctin_3d', 'rnin', 'gnio', 'tartanimu'):
            original_set, original_sources = trainer.val_set, trainer.val_sources
            full_cfg = get_cfg({**overrides, 'model': row['profile'], 'data': row['data_root']})
            full_val = build_dataset(full_cfg, 'val', training=False)
            try:
                trainer.val_set = full_val
                trainer.val_sources = [full_val.sequence_view(k) for k in range(len(full_val.sequence_ids))]
                diagnostic = trainer.train_eval_gap(windows=1024)
                legacy = {'status': 'passed' if diagnostic.get('num_windows', 0) >= 1024 else 'insufficient_actual_windows',
                          'requested_windows': 1024, 'actual_windows': diagnostic.get('num_windows'), 'receipt': diagnostic}
            except Exception as exc:
                legacy = {'status': 'failed', 'requested_windows': 1024, 'error': f'{type(exc).__name__}: {exc}'}
            finally:
                trainer.val_set, trainer.val_sources = original_set, original_sources
                del full_val, original_set, original_sources
                gc.collect(); torch.cuda.empty_cache()
        checkpoint = str(nio.ckpt_path)
        last_checkpoint_bytes = trainer.last.stat().st_size if trainer.last.exists() else 0
        nio.callbacks.clear(); trainer.optimizer.state.clear(); trainer.batch = None
        nio.trainer = None; del trainer; trainer = None
        _CKPT_CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
        nio.val(**{**overrides, 'split': 'val', 'name': 'before-reload-val'})
        prediction_source = fixture / 'sequences' / f"{provenance[2]['sequence_id']}.h5"
        nio.predict(prediction_source, **{**overrides, 'name': 'before-reload-predict'}, save=True)
        before = nio.results[0]
        expected = {name: getattr(before, name).copy() for name in ('pos_pred', 'vel_pred', 'vel_target')}
        if any(not np.isfinite(value).all() for value in expected.values()):
            raise FloatingPointError('GPU lifecycle prediction/target is nonfinite')
        del before, nio; before = nio = None
        _CKPT_CACHE.clear(); gc.collect(); torch.cuda.empty_cache()
        reload = NIO(checkpoint, metric_dims=int(cfg.dims), device='0')
        reload.val(**{**overrides, 'split': 'val', 'name': 'after-reload-val'})
        reload.predict(prediction_source, **{**overrides, 'name': 'after-reload-predict'}, save=True)
        for name, value in expected.items(): np.testing.assert_array_equal(value, getattr(reload.results[0], name))
        return {'status': 'passed', 'standard_NIO_Trainer': True, 'device': 'cuda:0', 'engineering_seed': 0,
                'formal_training_performed': False, 'optimizer_updates': updates[0],
                'changed_trainable_parameter_tensors': changed, 'checkpoint': checkpoint,
                'checkpoint_sha256': sha(checkpoint), 'checkpoint_bytes': Path(checkpoint).stat().st_size,
                'last_checkpoint_bytes': last_checkpoint_bytes,
                'model_tensor_bytes': sum(value.numel() * value.element_size() for value in reload.model.state_dict().values()),
                'pos_vel_prediction_and_target_reload_exact': True,
                'validation_before_after_reload': True, 'fixture_provenance': provenance,
                'scientific_window_history_output_loss_unchanged': True,
                'common_gap': gap, 'legacy_1024_gap': legacy, 'gap_windows_requested': gap_windows}
    finally:
        del nio, trainer, reload, before
        _CKPT_CACHE.clear(); gc.collect(); torch.cuda.empty_cache()


def capacity(args):
    torch, hardware = cuda_gate(args)
    report = load(Path(args.output) / 'coverage.json')
    choices, results = {}, []
    for row in report['coverage']:
        if args.profiles and row['profile'] not in args.profiles.split(','): continue
        if row['classic'] or not row['trainable'] or row.get('taskview_receipt', {}).get('status') != 'passed': continue
        if row['profile'] not in choices or row['dataset'] == 'ridi': choices[row['profile']] = row
    with resource_lock('gpu'):
        for profile, row in choices.items():
            record = {'profile': profile, 'dataset': row['dataset'], 'measurements': [], 'status': 'failed'}
            try:
                proof = cpu_proofs(args, profile)
                for entry in report['coverage']:
                    if entry['profile'] == profile:
                        entry['model_cpu_accepted'] = True; entry['model_cpu_receipt_sha256'] = proof['sha256']
                for batch in (128, 64, 32, 16, 8, 4, 2):
                    try:
                        measurement = _gpu_measure(torch, row, batch)
                        record['measurements'].append(measurement)
                        if max(measurement['peak_allocated_bytes'], measurement['peak_reserved_bytes']) <= hardware['total_memory_bytes'] * .8:
                            record.update(status='passed', measured_batch=batch); break
                    except Exception as exc:
                        record['measurements'].append({'batch': batch, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                        gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            results.append(record)
            save(Path(args.output) / 'capacity.json', {'hardware': hardware, 'results': results, 'formal_ready': False, 'complete': False})
            print(json.dumps(record), flush=True)
        common = min((r['measured_batch'] for r in results if r['status'] == 'passed'), default=None)
        common_gap = min(128, common or 0)
        if common and all(r['status'] == 'passed' for r in results):
            for record in results:
                row = choices[record['profile']]
                try:
                    record['common_batch_measurement'] = _gpu_measure(torch, row, common)
                    record['gpu_lifecycle'] = _gpu_lifecycle(args, torch, row, common, common_gap)
                    record['formal_admission_ready'] = True
                except Exception as exc:
                    record.update(formal_admission_ready=False, lifecycle_error=f'{type(exc).__name__}: {exc}')
                save(Path(args.output) / 'capacity.json', {'hardware': hardware, 'results': results, 'formal_ready': False, 'complete': False})
                print(json.dumps({'profile': record['profile'], 'common_batch': common,
                                  'gpu_lifecycle': record.get('gpu_lifecycle', {}).get('status'),
                                  'error': record.get('lifecycle_error')}), flush=True)
    expected = {r['profile'] for r in report['coverage'] if not r['classic']}
    ready = bool(results) and expected == set(choices) and all(r.get('formal_admission_ready') for r in results)
    measured = {r['profile']: r.get('common_batch_measurement') for r in results if r.get('common_batch_measurement')}
    training_s, validation_s, missing = 0., 0., []
    main_training_s, main_validation_s = 0., 0.
    for row in report['coverage']:
        if row['classic'] or not row['trainable']: continue
        windows = row.get('taskview_receipt', {}).get('training_windows')
        timing = measured.get(row['profile'])
        if windows is None or not timing:
            missing.append(row['id']); continue
        row_training_s = max(windows // common, 1) * timing['median_pipeline_step_s']
        training_s += row_training_s
        val_windows = row['taskview_receipt']['splits'].get('val', {}).get('valid_windows', 0)
        row_validation_s = math.ceil(val_windows / common) * timing['median_validation_batch_s']
        validation_s += row_validation_s
        if row['main_xyz']:
            main_training_s += row_training_s; main_validation_s += row_validation_s
    cost = {'candidate_common_epochs': 100, 'train_windows_budget': 0,
            'same_three_seeds_already_in_rows': True, 'actual_taskview_windows_and_stride': True,
            'gpu_timing_is_measured_on_native_ridi': True, 'other_dataset_timing_is_projection': True,
            'complete_all_learned_estimate': not missing, 'missing_coverage_rows': missing,
            'sum_all_learned_one_epoch_all_rows_gpu_hours': (training_s + validation_s) / 3600,
            'candidate_100_epochs_all_learned_gpu_hours': 100 * (training_s + validation_s) / 3600,
            'sum_main_xyz_one_epoch_all_rows_gpu_hours': (main_training_s + main_validation_s) / 3600,
            'candidate_100_epochs_main_xyz_gpu_hours': 100 * (main_training_s + main_validation_s) / 3600,
            'excludes_setup_test_extra_checkpoint_gap_and_early_stop_savings': True}
    checkpoints = {r['profile']: r.get('gpu_lifecycle', {}) for r in results}
    storage = {'formal_save_predictions': False, 'minimum_free_reserve_bytes': 30 * 2 ** 30,
               'learned_training_rows': sum(not r['classic'] and r['trainable'] for r in report['coverage']),
               'estimated_best_checkpoint_bytes': sum(checkpoints.get(r['profile'], {}).get('checkpoint_bytes', 0)
                   for r in report['coverage'] if not r['classic'] and r['trainable']),
               'estimated_stripped_model_tensor_bytes': sum(checkpoints.get(r['profile'], {}).get('model_tensor_bytes', 0)
                   for r in report['coverage'] if not r['classic'] and r['trainable']),
               'dense_prediction_matrix_saving_enabled': False,
               'largest_incomplete_last_checkpoint_bytes': max((r.get('last_checkpoint_bytes', 0) for r in checkpoints.values()), default=0),
               'completed_retention': 'best checkpoint plus identity/state/metrics; delete only verified completed cell last.pt',
               'complete_estimate': expected == set(checkpoints)}
    result = {'schema': 'ib.full-matrix.cuda-capacity.v2', 'hardware': hardware, 'results': results,
              'source_manifest_sha256': report['source']['source_manifest_sha256'],
              'common_batch_candidate': common, 'common_gap_windows_candidate': common_gap,
              'common_cache': True, 'common_workers': 2, 'common_val_batch': common,
              'common_amp': False, 'complete': True,
              'all_profiles_capacity_passed': expected == set(choices) and all(r['status'] == 'passed' for r in results),
              'formal_training_performed': False, 'formal_ready': ready, 'budget_estimate': cost, 'storage_estimate': storage,
              'remaining': ([] if ready else ['all active learned profiles common-batch GPU lifecycle acceptance'])}
    save(Path(args.output) / 'coverage.json', report)
    save(Path(args.output) / 'capacity.json', result)
    return result


def run(args):
    """A dispatched cell exclusively owns its stable ID until its terminal state."""
    if not getattr(args, 'row_id', None):
        raise RuntimeError('Use dispatch for the complete queue; run requires an explicit --row-id')
    path = Path(args.output) / 'run-locks' / f'{args.row_id}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(args)


def verify_current_completed(row, policy, policy_sha, source_sha, seeds_sha,
                             coverage_rows=None, transfer_origin=None,
                             transfer_origin_context=None):
    """File integrity alone cannot admit an old result into a new protocol."""
    from inertial_benchmark.engine.benchmark import BenchmarkItem, scientific_run_identity, verify_completed_state
    context = verify_completed_state(
        row, coverage_rows, transfer_origin, transfer_origin_context)
    if row.get('state') != 'completed': return
    overrides = {**policy['overrides'], 'recipe': 'unified', 'device': 'cpu' if row['classic'] else '0',
                 'seed': row['seed'], 'metric_dims': row['dims'], 'workers': policy['overrides'].get('workers', 2)}
    item = BenchmarkItem(row['profile'], row['profile'], row['data_root'], row['seed'], overrides,
        metadata={**row, 'source_manifest_sha256': source_sha, 'policy_sha256': policy_sha,
                  'seeds_record_sha256': seeds_sha,
                  'transfer_checkpoint_sha256': row.get('checkpoint_source', {}).get('sha256')})
    if context['run_identity'] != scientific_run_identity(item):
        raise RuntimeError('completed run requested scientific identity mismatch; refusing old-result reuse')


def verified_transfer_origin(row, coverage_rows, admission):
    """Resolve the one authoritative completed training row and snapshot it once."""
    from inertial_benchmark.engine.benchmark import matrix_state, verify_completed_state
    source_dataset = row.get('checkpoint_training_source')
    if not isinstance(source_dataset, str) or not source_dataset:
        raise RuntimeError('Transfer row lacks an explicit checkpoint training source')
    origin = next((candidate for candidate in coverage_rows
                   if candidate.get('profile') == row.get('profile')
                   and candidate.get('dataset') == source_dataset
                   and candidate.get('seed') == row.get('seed')
                   and candidate.get('input_track') == row.get('input_track')), None)
    if origin is None:
        raise RuntimeError('Matching declared-source origin is absent from qualified coverage')
    origin = matrix_state(admission, origin)
    if origin.get('state') != 'completed':
        raise RuntimeError('Matching declared-source origin is not completed')
    context = verify_completed_state(origin, coverage_rows)
    if context is None:
        raise RuntimeError('Matching declared-source origin lacks verified completion context')
    return origin, context


def checkpoint_source_from_verified_origin(origin, context):
    """Build transfer provenance from the already verified fd snapshot, without rereading it."""
    checkpoint = context.get('checkpoint') if isinstance(context, dict) else None
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('path'), Path)
            or checkpoint.get('sha256') != origin.get('checkpoint_sha256')):
        raise RuntimeError('Verified transfer origin checkpoint context is inconsistent')
    try:
        origin_path = Path(origin['checkpoint']).resolve(strict=True)
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError('Verified transfer origin checkpoint path is missing') from exc
    if checkpoint['path'] != origin_path:
        raise RuntimeError('Verified transfer origin checkpoint path is inconsistent')
    path = str(checkpoint['path'])
    return path, {'run_id': origin['id'], 'path': path, 'sha256': checkpoint['sha256']}


def _run(args):
    if args.lane == 'classic':
        torch = framework(args.source_root)
        hardware = {'device': 'cpu', 'nongradient_calibration_lane': True,
                    'visible_cpu_count': os.cpu_count()}
    else:
        torch, hardware = cuda_gate(args)
    from inertial_benchmark.engine.benchmark import (BenchmarkItem, _snapshot_regular_file,
        bind_completed_results, run_item, dependency_order,
        matrix_state, save_matrix_state, scientific_run_identity, verify_dataset_identity,
        verify_completed_state, retain_completed_checkpoint)
    from inertial_benchmark.data.manifest import resolve_dataset
    source = source_identity(args.source_root)
    policy, policy_sha = recipe(args.policy, source['source_manifest_sha256'],
                                None if args.lane == 'classic' else hardware)
    report = load(Path(args.output) / 'coverage.json')
    seeds, seeds_sha = seed_record(args.seeds)
    if seeds != report['seeds'] or seeds_sha != report['seeds_record_sha256']:
        raise RuntimeError('Formal seeds changed since coverage qualification')
    if source['source_manifest_sha256'] != report['source']['source_manifest_sha256']:
        raise RuntimeError('Immutable source differs from qualified queue')
    rows = sorted(report['coverage'], key=lambda r: (seeds.index(r['seed']), not r['main_xyz'],
                  r['dataset'] != 'euroc', r['dataset'] == 'blackbird', r['profile'], r['dataset']))
    ordered = dependency_order([BenchmarkItem(r['profile'], r['profile'], r['data_root'], r['seed'], metadata={
        'dataset_name': r['dataset'], 'input_track': r['input_track'], 'trainable': r['trainable'],
        'checkpoint_training_source': r.get('checkpoint_training_source'), 'id': r['id']}) for r in rows])
    by_id = {r['id']: r for r in rows}
    rows = [by_id[item.metadata['id']] for item in ordered]
    for row in rows:
        if row['id'] != args.row_id: continue
        if args.lane == 'learned' and row['classic']: continue
        if args.lane == 'classic' and not row['classic']: continue
        if args.profiles and row['profile'] not in args.profiles.split(','): continue
        if args.datasets and row['dataset'] not in args.datasets.split(','): continue
        if row.get('taskview_receipt', {}).get('status') != 'passed': continue
        state_file = Path(args.output) / 'run-states' / f'{row["id"]}.json'
        row.update(matrix_state(Path(args.output) / 'coverage.json', row))
        if row.get('state') == 'unsupported_missing_validation': continue
        try:
            proof = cpu_proofs(args, row['profile'])
            row['model_cpu_accepted'] = True; row['model_cpu_receipt_sha256'] = proof['sha256']
            verify_dataset_identity(row['data_root'], row)
            origin = None
            origin_context = None
            if row.get('state') == 'completed' and not row['trainable']:
                origin, origin_context = verified_transfer_origin(
                    row, rows, Path(args.output) / 'coverage.json')
            if row.get('state') == 'completed':
                verify_current_completed(
                    row, policy, policy_sha, source['source_manifest_sha256'], seeds_sha, rows,
                    origin, origin_context)
                continue
            verify_completed_state(row)
            if row.get('sampling_spec') != {key: policy['overrides'][key] for key in ('stride', 'eval_stride')}:
                raise RuntimeError('Qualified sampling differs from the frozen common recipe')
            over = {**policy['overrides'], 'recipe': 'unified', 'device': 'cpu' if row['classic'] else '0', 'seed': row['seed'],
                    'metric_dims': row['dims'], 'workers': policy['overrides'].get('workers', 2)}
            model = row['profile']
            if not row['trainable']:
                if row['dataset'] == 'blackbird':
                    transfer_source = 'euroc'
                else:
                    entry = next(d for d in report['datasets'] if d['name'] == row['dataset'])
                    transfer_source = entry.get('checkpoint_training_source')
                if not transfer_source:
                    row.update(state='dependency_pending', dependency_reason='explicit training source unspecified')
                    save_matrix_state(Path(args.output) / 'coverage.json', row)
                    continue
                candidate = next((r for r in rows if r['profile'] == row['profile'] and r['dataset'] == transfer_source
                                  and r['seed'] == row['seed'] and r['input_track'] == row['input_track']), None)
                if not candidate or matrix_state(
                        Path(args.output) / 'coverage.json', candidate).get('state') != 'completed':
                    row.update(state='dependency_pending', dependency_reason='matching declared-source origin not completed')
                    save_matrix_state(Path(args.output) / 'coverage.json', row)
                    continue
                origin, origin_context = verified_transfer_origin(
                    row, rows, Path(args.output) / 'coverage.json')
                verify_dataset_identity(origin['data_root'], origin)
                model, checkpoint_source = checkpoint_source_from_verified_origin(
                    origin, origin_context)
                origin_identity = origin_context['run_identity']
                if (origin_identity['source_manifest_sha256'] != source['source_manifest_sha256']
                        or origin_identity['policy_sha256'] != policy_sha
                        or origin_identity['seed_record_sha256'] != seeds_sha
                        or origin_identity['seed'] != row['seed']
                        or origin_identity['profile'] != row['profile']
                        or origin_identity['input_track'] != row['input_track']
                        or origin_identity['profile_config_sha256'] != row['profile_config_sha256']
                        or origin_identity['scientific_input_spec'] != row['scientific_input_spec']
                        or origin_identity['dataset_fingerprint'] != origin['dataset_fingerprint']):
                    raise RuntimeError('Transfer source scientific run identity mismatch')
                row['checkpoint_source'] = checkpoint_source
                row['cross_domain_transfer'] = origin['domain'] != row['domain']
            test_splits = [s for s in resolve_dataset(row['data_root']).test_splits if not s.startswith('official_')]
            if not test_splits:
                raise RuntimeError('No declared primary or extra scientific evaluation splits')
            item = BenchmarkItem(row['profile'], model, row['data_root'], row['seed'], over,
                                 test_splits, Path(args.output) / 'runs' / row['id'],
                                 split_receipts=row['taskview_receipt']['splits'],
                                 transfer_origin=origin,
                                 transfer_origin_verification=origin_context,
                                 coverage_rows=rows)
            item.metadata = {**row, 'source_manifest_sha256': source['source_manifest_sha256'],
                             'policy_sha256': policy_sha, 'seeds_record_sha256': seeds_sha,
                             'transfer_checkpoint_sha256': row.get('checkpoint_source', {}).get('sha256')}
            identity = scientific_run_identity(item)
            item.metadata = {}  # Admission already verified above; the shared state/identity format is identical.
            identity_file = item.run_dir / 'run-identity.json'
            if identity_file.exists() and load(identity_file) != identity:
                raise RuntimeError('Existing run identity mismatch; refusing resume or score merge')
            if not identity_file.exists() and item.run_dir.exists() and any(item.run_dir.iterdir()):
                raise RuntimeError('Existing outputs lack scientific run identity; refusing implicit resume')
            save(identity_file, identity)
            save(item.run_dir / 'execution-audit.json', {'hardware': hardware, 'cpu_receipt_sha256': proof['sha256'],
                 'cpu_proof_bridge': {k: v for k, v in proof.items() if k != 'record'},
                 'execution_lane': 'cpu_nongradient_calibration' if row['classic'] else 'cuda_learned'})
            free = os.statvfs(args.output)
            if free.f_bavail * free.f_frsize < 30 * 2 ** 30:
                raise RuntimeError('Insufficient disk headroom for checkpoints and continuing acquisitions')
            with resource_lock('cpu' if row['classic'] else 'gpu', getattr(args, 'gpu_lease_fd', None)):
                row['state'] = 'running'; save(Path(args.output) / f'execution-cell-{args.row_id}.json', report)
                save_matrix_state(Path(args.output) / 'coverage.json', row)
                status = run_item(item)
                if not row['classic']: torch.cuda.synchronize()
            if row['trainable']:
                checkpoint_context = _snapshot_regular_file(
                    item.run_dir / 'train/weights/best.pt', 'checkpoint', keep_bytes=False)
            else:
                checkpoint_context = origin_context['checkpoint']
            artifacts = bind_completed_results(row, item.run_dir, test_splits,
                                                row['taskview_receipt']['splits'])
            row.update(state='completed', result=status, run_identity_sha256=sha(identity_file),
                       run_identity_path=str(identity_file),
                       checkpoint=str(checkpoint_context['path']),
                       checkpoint_sha256=checkpoint_context['sha256'], result_artifacts=artifacts)
            verify_dataset_identity(row['data_root'], row)
            retain_completed_checkpoint(
                Path(args.output) / 'coverage.json', row, item.run_dir, policy,
                coverage_rows=rows, transfer_origin=origin,
                transfer_origin_context=origin_context)
        except BlockingIOError as exc:
            row.update(state='pending_resource', resource_reason=f'{type(exc).__name__}: {exc}')
        except Exception as exc:
            row.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        save_matrix_state(Path(args.output) / 'coverage.json', row)
        save(Path(args.output) / f'execution-cell-{args.row_id}.json', report)
        print(json.dumps({'profile': row['profile'], 'dataset': row['dataset'], 'seed': row['seed'],
                          'state': row['state'], 'error': row.get('error')}), flush=True)
    for row in report['coverage']:
        state_file = Path(args.output) / 'run-states' / f'{row["id"]}.json'
        if state_file.is_file(): row.update(load(state_file))
    report['coverage_completed'] = all(r['state'] == 'completed' for r in report['coverage'])
    report['delivery_scope_satisfied'] = report['architecture_count'] >= 31 and report['independent_dataset_count'] >= 10
    save(Path(args.output) / f'execution-cell-{args.row_id}.json', report)
    return report


def host_memory():
    values = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
    group = Path('/sys/fs/cgroup')
    maximum = (group / 'memory.max').read_text().strip()
    total = min(values['MemTotal'], int(maximum)) if maximum != 'max' else values['MemTotal']
    used = int((group / 'memory.current').read_text())
    return {'total_bytes': total, 'used_bytes': used, 'available_bytes': values['MemAvailable']}


def dispatch(args):
    """Keep the full dynamic coverage while running bounded independent cells."""
    if not 1 <= args.max_active <= 3:
        raise RuntimeError('At most three independently measured GPU cells may run concurrently')
    source = source_identity(args.source_root)
    if args.lane == 'learned':
        torch, hardware = cuda_gate(args)
    else:
        framework(args.source_root); torch = None; hardware = None
        if args.max_active != 1:
            raise RuntimeError('Nongradient calibration has one CPU resource lane')
    from inertial_benchmark.engine.benchmark import matrix_state, verify_completed_state
    policy, _ = recipe(args.policy, source['source_manifest_sha256'], hardware)
    capacity_receipt = load(policy['capacity_receipt'])
    measured = {r['profile']: r for r in capacity_receipt['results']}
    report_path = Path(args.output) / 'coverage.json'
    _, seeds_sha = seed_record(args.seeds)
    active, attempted, source_bytes = {}, set(), {}
    stop = [False]
    old_handlers = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
    results, started = [], time.monotonic()

    def publish(state):
        save(Path(args.output) / f'dispatch-{args.lane}.json', {
            'schema': 'ib.dynamic-matrix.dispatch.v1', 'state': state,
            'source_manifest_sha256': source['source_manifest_sha256'],
            'policy_sha256': sha(args.policy), 'maximum_active_cells': args.max_active,
            'memory_limit_fraction': .75, 'active': [{k: v for k, v in a.items() if k != 'process'} for a in active.values()],
            'results': results, 'elapsed_s': time.monotonic() - started,
            'full_scope_not_reduced_by_execution_order': True})

    lease = resource_lock('gpu') if args.lane == 'learned' else nullcontext(None)
    try:
        with lease as lease_fd:
            while not stop[0]:
                for cell, entry in list(active.items()):
                    code = entry['process'].poll()
                    if code is not None:
                        results.append({'id': cell, 'pid': entry['pid'], 'exit_code': code,
                                        'elapsed_s': time.monotonic() - entry['started_monotonic']})
                        del active[cell]
                report = load(report_path)
                rows = report['coverage']
                ready = []
                for original in rows:
                    row = {**original, **matrix_state(report_path, original)}
                    if row['classic'] != (args.lane == 'classic') or row['id'] in active or row['id'] in attempted:
                        continue
                    if args.profiles and row['profile'] not in args.profiles.split(','): continue
                    if args.datasets and row['dataset'] not in args.datasets.split(','): continue
                    if row.get('state') == 'completed':
                        origin = None
                        origin_context = None
                        if not row['trainable']:
                            origin, origin_context = verified_transfer_origin(
                                row, rows, report_path)
                        verify_current_completed(
                            row, policy, sha(args.policy), source['source_manifest_sha256'],
                            seeds_sha, rows, origin, origin_context)
                        continue
                    if row.get('taskview_receipt', {}).get('status') != 'passed': continue
                    if not row['trainable']:
                        origin = next((r for r in rows if r['profile'] == row['profile']
                            and r['dataset'] == row.get('checkpoint_training_source') and r['seed'] == row['seed']
                            and r['input_track'] == row['input_track']), None)
                        if not origin: continue
                        origin = {**origin, **matrix_state(report_path, origin)}
                        if origin.get('state') != 'completed': continue
                        verified_transfer_origin(row, rows, report_path)
                    ready.append(row)
                ready.sort(key=lambda r: (r['trainable'], not r['main_xyz'],
                    r['dataset'] not in ('euroc', 'ridi'), report['seeds'].index(r['seed']), r['profile'], r['dataset']))
                launched = False
                for row in ready:
                    if len(active) >= args.max_active or (args.max_jobs and len(attempted) >= args.max_jobs): break
                    cpu_proofs(args, row['profile'])
                    if row['data_root'] not in source_bytes:
                        source_bytes[row['data_root']] = sum(p.stat().st_size for p in (Path(row['data_root']) / 'sequences').glob('*.h5'))
                    record = measured.get(row['profile'], {})
                    parameter_bytes = record.get('gpu_lifecycle', {}).get('model_tensor_bytes', 0)
                    host_estimate = source_bytes[row['data_root']] * 4 + parameter_bytes * 10 + 2 ** 30
                    memory = host_memory()
                    reserved_host = sum(a['estimated_host_bytes'] for a in active.values())
                    if memory['used_bytes'] + reserved_host + host_estimate > memory['total_bytes'] * .75:
                        continue
                    peak = record.get('common_batch_measurement', {}).get('peak_reserved_bytes', 0)
                    gpu_estimate = max(int(peak * 1.25), 2 * 2 ** 30)
                    if torch is not None:
                        free, _ = torch.cuda.mem_get_info()
                        if free < gpu_estimate + sum(a['estimated_gpu_bytes'] for a in active.values()) + 2 * 2 ** 30:
                            continue
                    command = [sys.executable, str(Path(__file__).resolve()), 'run', '--source-root', str(args.source_root),
                        '--output', str(args.output), '--policy', str(args.policy), '--seeds', str(args.seeds),
                        '--lane', args.lane, '--row-id', row['id']]
                    if args.gpu_authorized: command.append('--gpu-authorized')
                    if lease_fd is not None: command.extend(['--gpu-lease-fd', str(lease_fd)])
                    for flag, paths in (('--cpu-receipts', args.cpu_receipts), ('--science-bridges', args.science_bridges)):
                        if paths: command.extend([flag, *map(str, paths)])
                    log = Path(args.output) / 'worker-logs' / f'{row["id"]}.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = {**os.environ, 'PYTHONPATH': str(Path(args.source_root) / 'src'),
                           **{name: '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}}
                    with log.open('a') as stream:
                        process = subprocess.Popen(command, cwd=args.source_root, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(() if lease_fd is None else (lease_fd,)))
                    active[row['id']] = {'process': process, 'pid': process.pid, 'profile': row['profile'],
                        'dataset': row['dataset'], 'seed': row['seed'], 'started_monotonic': time.monotonic(),
                        'estimated_host_bytes': host_estimate, 'estimated_gpu_bytes': gpu_estimate, 'log': str(log)}
                    attempted.add(row['id']); launched = True
                    publish('running')
                if not active and (not ready or not launched):
                    break  # Pending qualification/dependencies remain in coverage for the next invocation.
                publish('running')
                time.sleep(2)
    finally:
        for entry in active.values():
            os.killpg(entry['pid'], signal.SIGTERM)
        for entry in active.values():
            try: entry['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(entry['pid'], signal.SIGKILL); entry['process'].wait(timeout=5)
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)
        publish('operator_stopped' if stop[0] else 'finite_ready_batch_complete')
    return {'coverage': load(report_path)['coverage'], 'worker_results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory', 'qualify', 'capacity', 'run', 'dispatch'))
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--seeds', type=Path, default=SERVER_ROOT / '.local/plans/random-seeds.json')
    parser.add_argument('--cpu-receipts', type=Path, nargs='*', default=[])
    parser.add_argument('--science-bridges', type=Path, nargs='*', default=[],
                        help='Exact approved source-change bridges with actual targeted test receipt hashes')
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--profiles', help='Finite qualification/capacity batch; preserves full coverage rows')
    parser.add_argument('--datasets', help='Finite taskview batch; preserves every other coverage row as pending')
    parser.add_argument('--lane', choices=('learned', 'classic'), default='learned')
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--probe-legacy-gap', action='store_true', help='Keep actual 1024-window representative diagnostic receipts')
    parser.add_argument('--sampling-stride', type=int, help='Explicit common training sampling; scientific spans/loss stay unchanged')
    parser.add_argument('--eval-stride', type=int, help='Explicit common evaluation sampling')
    parser.add_argument('--row-id', help='One stable coverage ID; exclusively locked for the complete lifecycle')
    parser.add_argument('--gpu-lease-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--max-active', type=int, default=3, help='Measured independent CUDA cell concurrency (maximum three)')
    parser.add_argument('--max-jobs', type=int, help='Finite execution batch; all other coverage rows remain pending')
    args = parser.parse_args()
    result = globals()[args.action](args)
    if args.action == 'run' and any(r['state'] == 'failed' for r in result['coverage']): return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
