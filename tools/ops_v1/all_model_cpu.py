#!/usr/bin/env python3
"""Run every existing model profile through genuine, isolated AutoDL CPU lifecycles.

This is an engineering acceptance campaign, never a research training budget.
The standard NIO/Trainer, scientific input spans, augmentations and losses remain
intact. Only the number of genuine training windows and CPU batch size are small.
Classic sequence baselines calibrate on train; they never claim backpropagation.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# The execution checkout and shared locks live on a private Agent host.  The
# public source carries only safe placeholders; operators set these variables
# in that host's service environment.
REMOTE_ROOT = Path(os.environ.get('IPB_V1_REMOTE_ROOT', '<agent-project-root>'))
LOCK_ROOT = Path(os.environ.get('IPB_V1_LOCK_ROOT', '<agent-runtime-root>/campaigns'))
THREAD_NAMES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    def finite_json(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {'nonfinite_diagnostic': str(item)}
        if isinstance(item, dict):
            return {key: finite_json(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite_json(child) for child in item]
        return item

    temporary.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')
    temporary.replace(path)


def server_only(root):
    resolved = Path(root).resolve()
    if resolved != REMOTE_ROOT and not resolved.is_relative_to(REMOTE_ROOT / '.local/source-snapshots'):
        raise RuntimeError('CPU framework execution is restricted to the AutoDL v1 checkout')


@contextmanager
def shared_locks(live_source=True):
    handles = []
    try:
        names = ('cpu-resource.lock', 'gpu-0.lock') if live_source else ('cpu-resource.lock',)
        for name in names:
            path = LOCK_ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a+')
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def resource_snapshot():
    cgroup = Path('/sys/fs/cgroup')
    result = {'self_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for name in ('memory.current', 'memory.max', 'cpu.max', 'memory.events'):
        path = cgroup / name
        if path.exists():
            result[name] = path.read_text().strip()
    return result


def environment_identity():
    versions = {}
    for name in ('torch', 'numpy', 'h5py', 'scipy', 'pandas'):
        versions[name] = importlib.metadata.version(name)
    return {'python_version': sys.version.split()[0], 'package_versions': versions}


def source_guard(root):
    path = Path(root) / 'source_manifest.json'
    manifest = json.loads(path.read_text())
    files = {**manifest.get('files', {}),
             **manifest.get('operational_acceptance_helpers', {})}
    if not files:
        raise RuntimeError('deployment has no managed source inventory')
    for name, expected in files.items():
        if digest(Path(root) / name) != expected:
            raise RuntimeError(f'managed source changed: {name}')
    return {'source_manifest_sha256': digest(path), 'managed_files_verified': len(files)}


def native_metadata_guard(root):
    paths = list((Path(root) / 'datasets/ipb-v1/full-core').glob('*/dataset.json'))
    paths += [Path(root) / f'datasets/ipb-v1/{name}/dataset.json'
              for name in ('euroc', 'blackbird')]
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        result[str(path.relative_to(root))] = {
            'manifest_sha256': digest(path), 'fingerprint': manifest['fingerprint'],
            'split_sha256': {p.name: digest(p) for p in sorted((path.parent / 'splits').glob('*.txt'))},
        }
    return result


def stop_process(process):
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def monitored_process(argv, cwd, env, log, seconds, memory_fraction=0.93):
    before = resource_snapshot()
    maximum = before.get('memory.max', 'max')
    cap = int(int(maximum) * memory_fraction) if maximum.isdigit() else None
    started = time.monotonic()
    peak = 0
    reason = None
    with Path(log).open('w') as stream:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            current = Path('/sys/fs/cgroup/memory.current')
            memory = int(current.read_text()) if current.exists() else 0
            peak = max(peak, memory)
            if time.monotonic() - started > seconds:
                reason = 'time_limit'
                stop_process(process)
                break
            if cap is not None and memory > cap:
                reason = 'cgroup_memory_guard'
                stop_process(process)
                break
            time.sleep(0.25)
    return {'exit_code': process.returncode, 'elapsed_s': round(time.monotonic() - started, 3),
            'termination_reason': reason, 'peak_cgroup_memory_bytes': peak,
            'resources_before': before, 'resources_after': resource_snapshot()}


def make_fixture(root, output, profile, cfg, dataset, train_windows=4, val_windows=2, test_windows=2):
    import numpy as np

    from inertial_benchmark.data.format import load_sequence, save_sequence
    from inertial_benchmark.data.manifest import fingerprint
    from inertial_benchmark.data.splits import read_split, write_split
    from inertial_benchmark.data.views import ViewConfig
    from inertial_benchmark.engine.budget import is_sequence_model

    native = Path(root) / f'datasets/ipb-v1/full-core/{dataset}'
    parent_manifest = json.loads((native / 'dataset.json').read_text())
    spec = ViewConfig.from_cfg(cfg)
    fixture = Path(output) / 'engineering-only-fixture'
    records, sequences, splits = [], {}, {}
    classic = is_sequence_model(profile)
    for split in ('train', 'val', 'test'):
        stride = int(cfg.stride) if split == 'train' else int(cfg.eval_stride)
        count = {'train': train_windows, 'val': val_windows, 'test': test_windows}[split]
        samples = spec.input_span + (count - 1) * stride
        if classic and split == 'train':
            samples = max(samples, 2400)
        found = False
        for sid in read_split(native / f'splits/{split}.txt'):
            source = native / parent_manifest['sequences'][sid].get('file', f'sequences/{sid}.h5')
            seq = load_sequence(source)
            if seq.sample_rate != spec.rate:
                raise RuntimeError('CPU smoke must preserve the configured scientific sample rate')
            valid = seq.valid_device if spec.orientation == 'device' else seq.valid
            if spec.orientation == 'device' and seq.device_orientation is None:
                continue
            if len(valid) < samples:
                continue
            prefix = np.r_[0, np.cumsum(valid.astype(np.int64))]
            candidates = np.flatnonzero(prefix[samples:] - prefix[:-samples] == samples)
            if not len(candidates):
                continue
            start = int(candidates[0])
            excerpt = seq.slice(start, start + samples)
            original_spacing = np.diff(excerpt.timestamp).copy()
            excerpt.timestamp = excerpt.timestamp - excerpt.timestamp[0]
            np.testing.assert_allclose(np.diff(excerpt.timestamp), original_spacing,
                                       rtol=0, atol=1e-10)
            for key in ('gyroscope', 'accelerometer', 'position', 'orientation',
                        'valid_imu', 'valid_pose', 'device_orientation', 'velocity',
                        'valid_device_orientation'):
                original = getattr(seq, key)
                if original is not None:
                    np.testing.assert_array_equal(getattr(excerpt, key), original[start:start + samples])
            parent_sha = digest(source)
            excerpt.attrs.update(engineering_only=True, source_sequence_sha256=parent_sha,
                                 source_slice_start=start, source_slice_stop=start + samples,
                                 timestamp_origin_reset_only=True)
            target = fixture / f'sequences/{sid}.h5'
            save_sequence(target, excerpt)
            sequences[sid] = {'file': f'sequences/{sid}.h5', 'sha256': digest(target),
                              'num_samples': samples, 'group_id': excerpt.group_id}
            splits[split] = [sid]
            write_split(fixture / f'splits/{split}.txt', [sid])
            records.append({'split': split, 'sequence_id': sid, 'sample_range': [start, start + samples],
                            'parent_sha256': parent_sha, 'full_xyz_sensor_pose_arrays_exact': True,
                            'sample_rate_unchanged': seq.sample_rate, 'samples': samples})
            found = True
            del excerpt, seq
            gc.collect()
            break
        if not found:
            raise RuntimeError(f'no genuine {dataset}/{split} crop for {profile}: {samples} samples')
    if len(sequences) != 3:
        raise RuntimeError('CPU fixture must use three distinct real recordings')
    derived = {'schema_version': '1.0', 'dataset': f'{dataset}_{profile}_engineering_only',
               'engineering_only': True, 'parent_fingerprint': parent_manifest['fingerprint'],
               'sequences': sequences, 'splits': splits,
               'fingerprint': fingerprint({sid: row['sha256'] for sid, row in sequences.items()}, splits)}
    write_json(fixture / 'dataset.json', derived)
    return fixture, records


def cpu_worker(args):
    server_only(args.root)
    os.environ.update({name: str(args.threads) for name in THREAD_NAMES})
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    import torch

    from inertial_benchmark import NIO
    from inertial_benchmark.cfg import INPUT_KEYS, get_cfg, load_model_cfg
    from inertial_benchmark.nn.base import SequenceModel
    from inertial_benchmark.utils.torch_utils import _CKPT_CACHE

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    output = Path(args.output)
    receipt_path = output / 'receipt.json'
    source_before = source_guard(args.root)
    native_before = native_metadata_guard(args.data_root)
    cfg = get_cfg({'model': args.profile, 'recipe': 'unified'})
    model_description = load_model_cfg(args.profile)
    receipt = {'schema': 'ib.dev-v1.all-profile-cpu.v1', 'profile': args.profile,
               'architecture': model_description['arch'], 'status': 'running',
               **source_before, 'scientific_input_spec': {k: getattr(cfg, k) for k in INPUT_KEYS},
               'science_window_history_output_steps_loss_overridden': False,
               'gpu_used': False, 'stage': 'fixture', 'updates': 0, 'events': [],
               'engineering_seed': 0, 'formal_seed_run': False, 'threads': args.threads}
    receipt['environment_identity'] = environment_identity()

    def stage(name):
        receipt['stage'] = name
        receipt['events'].append({'stage': name, 'elapsed_s': round(time.monotonic() - started, 3)})
        write_json(receipt_path, receipt)

    def parameter_hashes(model):
        return {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                for name, value in model.named_parameters() if value.requires_grad}

    try:
        fixture, records = make_fixture(args.data_root, output, args.profile, cfg, args.dataset)
        receipt['fixtures'] = records
        metric_dims = int(cfg.dims)
        overrides = {'data': str(fixture), 'device': 'cpu', 'epochs': 1, 'batch': args.batch,
                     'val_batch': args.batch, 'workers': 0, 'cache': False, 'amp': False,
                     'recipe': 'unified', 'metric_dims': metric_dims, 'seed': 0,
                     'optimizer': 'adamw', 'lr': 0.001, 'weight_decay': 0.0001,
                     'efficiency': False, 'plots': False, 'save_predictions': True,
                     'project': str(output / 'runs'), 'name': 'real-cpu-lifecycle'}
        receipt['cpu_engineering_optimizer_override'] = None
        stage('construct')
        nio = NIO(args.profile, metric_dims=metric_dims, device='cpu')
        counters = {'forward': 0, 'backward': 0, 'updates': 0}
        initial = {}

        def setup(trainer):
            initial.update(parameter_hashes(trainer.model))
            receipt['loss_name'] = trainer.model.loss_name
            receipt['loss_kwargs'] = trainer.model.loss_kwargs
            receipt['input_shape'] = list(trainer.model.input_spec.input_shape)
            receipt['output_shape'] = list(trainer.model.input_spec.output_shape)
            receipt['train_windows'] = len(trainer.train_set)
            receipt['val_windows'] = len(trainer.val_set)
            receipt['sequence_baseline'] = trainer.sequence_model
            receipt['scientific_input_spec_retained'] = all(
                getattr(trainer.args, key) == getattr(cfg, key) for key in INPUT_KEYS)
            if not receipt['scientific_input_spec_retained']:
                raise AssertionError('configured scientific input spec changed')
            trainer.model.register_forward_pre_hook(lambda *unused: stage('forward'))

            def forward_done(*unused):
                counters['forward'] += 1
                stage('forward_complete')

            trainer.model.register_forward_hook(forward_done)
            if not trainer.sequence_model:
                for parameter in trainer.model.parameters():
                    if parameter.requires_grad:
                        def gradient_hook(gradient):
                            if not torch.isfinite(gradient).all():
                                raise FloatingPointError('nonfinite gradient in real backward')
                            counters['backward'] += 1
                            return gradient
                        parameter.register_hook(gradient_hook)
            stage('train')

        def batch_done(trainer):
            counters['updates'] += 1
            receipt['updates'] = counters['updates']
            states = [state for state in trainer.optimizer.state.values() if 'step' in state]
            receipt['optimizer_state_parameter_count'] = len(states)
            receipt['optimizer_state_steps'] = sorted({int(state['step']) for state in states})
            stage('optimizer_step_complete')

        nio.add_callback('on_pretrain_routine_end', setup)
        nio.add_callback('on_train_batch_end', batch_done)
        nio.add_callback('on_val_start', lambda *unused: stage('validation'))
        nio.add_callback('on_model_save', lambda *unused: stage('checkpoint_saved'))
        nio.train(**overrides)
        trainer = nio.trainer
        classic = isinstance(nio.model, SequenceModel)
        final_hashes = parameter_hashes(nio.model)
        changed = sum(initial[name] != value for name, value in final_hashes.items())
        import csv
        with trainer.csv.open() as stream:
            training_rows = list(csv.DictReader(stream))
        actual_train_loss = None if classic else float(training_rows[-1]['train/loss'])
        if not classic and not math.isfinite(actual_train_loss):
            raise AssertionError('recorded native-target training loss is not finite')
        receipt.update(counters=counters, changed_trainable_parameter_tensors=changed,
                       train_loss=None if classic else actual_train_loss,
                       train_loss_finite=None if classic else True,
                       checkpoint=str(nio.ckpt_path), batch=args.batch,
                       standard_NIO_Trainer=True)
        if classic:
            if counters['updates'] != 0 or counters['backward'] != 0:
                raise AssertionError('classic baseline must not pretend to backpropagate')
            receipt['nontrainable_reason'] = 'SequenceModel: train-only calibration, no gradient optimizer'
            receipt['calibration'] = nio.model.model_cfg.get('calibration', {})
            stats = receipt['calibration']
            if not stats or any(not math.isfinite(v) for v in stats.values()
                                if isinstance(v, (int, float))):
                raise AssertionError('missing or nonfinite train-only calibration statistics')
            if args.profile == 'pdr' and int(stats.get('segments', 0)) < 1:
                raise AssertionError('PDR calibration has no genuine train segments')
            if args.profile == 'mean_speed_heading' and int(stats.get('windows', 0)) < 1:
                raise AssertionError('mean-speed-heading calibration has no genuine train windows')
            receipt['train_loss_not_applicable'] = 'nontrainable train-only calibration'
        else:
            expected_updates = len(trainer.train_loader)
            if counters['updates'] != expected_updates or expected_updates < 1:
                raise AssertionError('missing genuine optimizer update')
            if not changed or not counters['backward']:
                raise AssertionError('no genuine gradient/parameter update')
            # A one-epoch engineering lifecycle does not exercise the scientific
            # covariance phase. Probe both sides without changing the training recipe.
            from inertial_benchmark.nn.losses import build_loss
            loss_fn = build_loss(nio.model.loss_name, **nio.model.loss_kwargs)
            phase = getattr(loss_fn, 'phase', getattr(loss_fn, 'stage', None))
            if phase is not None:
                import copy
                probe_model = copy.deepcopy(nio.model)
                switch = int(nio.model.loss_kwargs.get('switch_epoch', cfg.loss_switch_epoch))
                phase_records = []
                for epoch in sorted({max(switch - 1, 0), switch}):
                    probe_model.train()
                    probe_model.zero_grad(set_to_none=True)
                    native = next(iter(trainer.train_loader))
                    out = (probe_model(native['imu'], native['extra']) if native.get('extra')
                           else probe_model(native['imu']))
                    cov = out.get('all_logstd', out.get('logstd'))
                    means = out.get('all_vel', out.get('vel'))
                    if cov is not None and cov.requires_grad: cov.retain_grad()
                    if means.requires_grad: means.retain_grad()
                    scientific_loss, _ = probe_model.loss(out, native, epoch)
                    if not torch.isfinite(scientific_loss):
                        raise AssertionError('scientific phase probe loss is nonfinite')
                    scientific_loss.backward()
                    torch.nn.utils.clip_grad_norm_(probe_model.parameters(), math.inf, error_if_nonfinite=True)
                    active = sum(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                 for p in probe_model.parameters())
                    if not active:
                        raise AssertionError('scientific phase probe has no active gradients')
                    phase_records.append({'epoch': epoch, 'phase': phase(epoch),
                                          'native_target_loss': float(scientific_loss.detach()),
                                          'active_gradient_parameter_tensors': active,
                                          'covariance_output_gradient_nonzero': bool(cov is not None and cov.grad is not None
                                               and torch.count_nonzero(cov.grad)),
                                          'z_output_gradient_nonzero': bool(cfg.dims == 3 and means.grad is not None
                                               and torch.count_nonzero(means.grad[..., 2])),
                                          'optimizer_step_performed': False})
                    if args.profile == 'meio' and epoch == switch:
                        if not (phase_records[-1]['covariance_output_gradient_nonzero']
                                and phase_records[-1]['z_output_gradient_nonzero']):
                            raise AssertionError('MEIO switched scientific covariance/XYZ gradients missing')
                    del native, out, scientific_loss
                receipt['scientific_loss_phase_backprobes'] = phase_records
                del probe_model, cov, means
        metrics = json.loads((trainer.save_dir / 'metrics.json').read_text())
        receipt['default_train_eval_gap'] = metrics.get('train_eval_gap')
        receipt['gap_was_not_patched'] = True
        # Release training-only optimizer/checkpoint memory before independent reload.
        nio.callbacks.clear()
        trainer.optimizer.state.clear()
        trainer.batch = None
        nio.trainer = None
        del trainer
        _CKPT_CACHE.clear()
        gc.collect()
        stage('val_api')
        before_val = nio.val(**{**overrides, 'split': 'val', 'name': 'before-reload-val'})
        prediction_source = fixture / 'sequences' / f"{records[2]['sequence_id']}.h5"
        stage('predict_api')
        nio.predict(prediction_source, **{**overrides, 'name': 'before-reload-predict'}, save=True)
        before = nio.results[0]
        if before.skipped or not len(before.vel_pred):
            raise AssertionError(f'prediction missing: {before.skipped}')
        reference_position = before.pos_pred.copy()
        reference_velocity = before.vel_pred.copy()
        reference_target = before.vel_target.copy()
        if reference_position.shape[-1] != metric_dims or reference_velocity.shape[-1] != metric_dims:
            raise AssertionError('predict output does not match declared dimensions')
        if metric_dims == 3:
            before.compute_metrics(3)
            if not any(key.startswith('xy_') for key in before.metrics):
                raise AssertionError('missing XY metrics derived from the same XYZ prediction')
            receipt['xyz_prediction_and_xy_slicing'] = True
        checkpoint = str(nio.ckpt_path)
        del before, nio
        _CKPT_CACHE.clear()
        gc.collect()
        stage('checkpoint_reload')
        reloaded = NIO(checkpoint, device='cpu')
        after_val = reloaded.val(**{**overrides, 'split': 'val', 'name': 'reload-val'})
        reloaded.predict(prediction_source, **{**overrides, 'name': 'reload-predict'}, save=True)
        after = reloaded.results[0]
        np.testing.assert_array_equal(reference_position, after.pos_pred)
        np.testing.assert_array_equal(reference_velocity, after.vel_pred)
        np.testing.assert_array_equal(reference_target, after.vel_target)
        for left, right in zip(before_val.sequences, after_val.sequences):
            np.testing.assert_array_equal(left.pos_pred, right.pos_pred)
            np.testing.assert_array_equal(left.vel_pred, right.vel_pred)
        if len(before_val.sequences) != len(after_val.sequences) or not len(before_val.sequences):
            raise AssertionError('validation sequence coverage changed after checkpoint reload')
        receipt.update(status='passed', validation_api=True, prediction_api=True,
                       checkpoint_reload_position_velocity_target_array_equal=True,
                       actual_prediction_shape=list(after.pos_pred.shape),
                       cuda_initialized=torch.cuda.is_initialized(),
                       native_metadata_unchanged=native_metadata_guard(args.data_root) == native_before,
                       source_guard_after=source_guard(args.root),
                       elapsed_s=round(time.monotonic() - started, 3),
                       resources=resource_snapshot())
        if receipt['cuda_initialized'] or not receipt['native_metadata_unchanged']:
            raise AssertionError('CPU isolation/data preservation guard failed')
        stage('complete')
        return 0
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                       traceback=traceback.format_exc(), elapsed_s=round(time.monotonic() - started, 3),
                       resources=resource_snapshot())
        write_json(receipt_path, receipt)
        traceback.print_exc()
        return 1


def cpu_campaign(args):
    server_only(args.root)
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    profiles = sorted(p.stem for p in (root / 'src/inertial_benchmark/cfg/models').glob('*.yaml'))
    if args.profiles:
        selected = args.profiles.split(',')
        if set(selected) - set(profiles):
            raise ValueError('unknown profiles in campaign selection')
        profiles = selected
    env = {**os.environ, **{name: str(args.threads) for name in THREAD_NAMES},
           'CUDA_VISIBLE_DEVICES': '-1',
           'PYTHONPATH': str(root / 'src'), 'PYTHONUNBUFFERED': '1', 'IPB_VERBOSE': '0'}
    live_source = root.resolve() == REMOTE_ROOT
    with shared_locks(live_source=live_source):
        source_before = source_guard(root)
        native_before = native_metadata_guard(args.data_root)
        campaign = {'schema': 'ib.dev-v1.all-profile-cpu-campaign.v1', 'status': 'running',
                    'created_utc': datetime.now(timezone.utc).isoformat(), **source_before,
                    'expected_profiles': profiles, 'results': [], 'gpu_used': False,
                    'gpu_queue_armed': False, 'scientific_spec_overrides': False,
                    'cpu_lock_exclusive': True, 'gpu_lock_exclusive': live_source, 'skipped': 0,
                    'immutable_snapshot_mode': not live_source,
                    'parallel_processes': args.jobs, 'threads_per_process': args.threads,
                    'batch_per_process': args.batch}
        write_json(output / 'campaign.json', campaign)

        def run_profile(profile):
            directory = output / profile
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(root),
                       '--output', str(directory), '--profile', profile, '--batch', str(args.batch),
                       '--dataset', args.dataset, '--threads', str(args.threads),
                       '--data-root', str(args.data_root)]
            process = monitored_process(command, root, env, directory / 'worker.log', args.seconds)
            receipt_file = directory / 'receipt.json'
            receipt = json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
            passed = process['exit_code'] == 0 and receipt.get('status') == 'passed'
            row = {'profile': profile, 'status': 'passed' if passed else 'failed',
                   'architecture': receipt.get('architecture'), **process,
                   'last_stage': receipt.get('stage'), 'error': receipt.get('error'),
                   'receipt': str(receipt_file), 'receipt_sha256': digest(receipt_file)
                   if receipt_file.exists() else None}
            return row

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = {pool.submit(run_profile, profile): profile for profile in profiles}
            campaign['active_profiles'] = list(profiles)
            write_json(output / 'campaign.json', campaign)
            for future in as_completed(pending):
                row = future.result()
                campaign['results'].append(row)
                campaign['active_profiles'].remove(row['profile'])
                campaign['passed'] = sum(r['status'] == 'passed' for r in campaign['results'])
                campaign['failed'] = sum(r['status'] != 'passed' for r in campaign['results'])
                write_json(output / 'campaign.json', campaign)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if source_guard(root) != source_before or native_metadata_guard(args.data_root) != native_before:
                    raise RuntimeError('source/native metadata changed during frozen campaign')
        campaign.update(status='complete', active_profile=None,
                        passed=sum(row['status'] == 'passed' for row in campaign['results']),
                        failed=sum(row['status'] != 'passed' for row in campaign['results']),
                        native_metadata_unchanged=native_metadata_guard(args.data_root) == native_before,
                        source_guard_after=source_guard(root))
        campaign['all_profiles_passed'] = campaign['failed'] == 0
        write_json(output / 'campaign.json', campaign)
    return 0 if campaign['all_profiles_passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for name in ('cpu', 'worker'):
        action = actions.add_parser(name)
        action.add_argument('--root', type=Path, default=REMOTE_ROOT)
        action.add_argument('--data-root', type=Path, default=REMOTE_ROOT)
        action.add_argument('--output', type=Path, required=True)
        action.add_argument('--batch', type=int, choices=(1, 2), default=2)
        action.add_argument('--threads', type=int, choices=range(1, 9), default=2)
        action.add_argument('--dataset', default='ridi')
        if name == 'cpu':
            action.add_argument('--profiles')
            action.add_argument('--seconds', type=float, default=600)
            action.add_argument('--jobs', type=int, choices=range(1, 9), default=4)
        else:
            action.add_argument('--profile', required=True)
    args = parser.parse_args()
    return cpu_campaign(args) if args.action == 'cpu' else cpu_worker(args)


if __name__ == '__main__':
    raise SystemExit(main())
