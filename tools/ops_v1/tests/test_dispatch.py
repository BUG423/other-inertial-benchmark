"""Resource leases are tested on the server without starting a formal training cell."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def ops():
    path = Path(__file__).resolve().parents[1] / 'full_matrix.py'
    spec = importlib.util.spec_from_file_location('dispatch_ops_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_lease_is_shared_only_with_its_owned_child(tmp_path):
    module = ops()
    module.LOCK_ROOT = tmp_path
    with module.resource_lock('gpu') as lease:
        code = """import fcntl, os, sys
fd = int(sys.argv[1])
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
with open(sys.argv[2], 'a+') as unrelated:
    try:
        fcntl.flock(unrelated.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError('unrelated GPU worker acquired the existing lease')
"""
        subprocess.run([sys.executable, '-c', code, str(lease), str(tmp_path / 'gpu-0.lock')],
                       pass_fds=(lease,), check=True, timeout=10)
        with module.resource_lock('gpu', lease):
            pass
        with pytest.raises(BlockingIOError):
            with module.resource_lock('gpu'):
                pass
    with module.resource_lock('gpu'):
        pass


def test_cpu_lock_cannot_be_forged_as_a_gpu_lease(tmp_path):
    module = ops()
    module.LOCK_ROOT = tmp_path
    with module.resource_lock('cpu') as cpu:
        with pytest.raises(RuntimeError, match='Invalid inherited'):
            with module.resource_lock('gpu', cpu):
                pass


def test_run_requires_an_exclusive_stable_cell_id():
    module = ops()
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match='explicit --row-id'):
        module.run(SimpleNamespace())


def test_ops_transfer_provenance_reuses_verified_checkpoint_context(
        tmp_path, monkeypatch):
    module = ops()
    checkpoint = tmp_path / 'origin-best.pt'
    checkpoint.write_bytes(b'authoritative-origin-a')
    digest = module.sha(checkpoint)
    origin = {'id': 'origin-run', 'checkpoint': str(checkpoint),
              'checkpoint_sha256': digest}
    context = {'checkpoint': {
        'path': checkpoint.resolve(), 'sha256': digest, 'size': checkpoint.stat().st_size}}
    monkeypatch.setattr(
        module, 'sha', lambda *_args, **_kwargs:
        pytest.fail('ops transfer must not reread/re-hash the verified checkpoint'))

    model, source = module.checkpoint_source_from_verified_origin(origin, context)

    assert model == str(checkpoint.resolve())
    assert source == {'run_id': 'origin-run', 'path': model, 'sha256': digest}


def test_ops_cleared_metadata_all_na_fails_before_trainer(tmp_path, monkeypatch):
    """Ops clears metadata after admission but keeps its formal split receipt."""
    from inertial_benchmark.engine.benchmark import BenchmarkItem, _run_item
    from inertial_benchmark.engine.trainer import Trainer

    native = tmp_path / 'native'
    (native / 'splits').mkdir(parents=True)
    (native / 'sequences').mkdir()
    (native / 'dataset.json').write_text('{"dataset":"native"}')
    (native / 'splits' / 'test.txt').write_text('')
    item = BenchmarkItem(
        'formal-profile', 'formal-profile', str(native), 1735231569,
        splits=['test'], root=tmp_path / 'runs', metadata={},
        split_receipts={'test': {'status': 'not_applicable_missing_split'}},
    )
    monkeypatch.setattr(
        Trainer, 'train', lambda *_args, **_kwargs: pytest.fail('Trainer must not run'))

    with pytest.raises(RuntimeError, match='no applicable nonempty evaluation split'):
        _run_item(item)
    assert not item.run_dir.exists()


def test_completed_cell_reuse_rejects_changed_policy_source_and_seed_record(
        tmp_path, monkeypatch):
    module = ops()
    from inertial_benchmark.engine.benchmark import (BenchmarkItem, _validate_result_split,
        scientific_run_identity)
    import json
    native = tmp_path / 'native'; (native / 'splits').mkdir(parents=True)
    (native / 'dataset.json').write_text(json.dumps({'dataset': 'native'}))
    (native / 'splits' / 'test.txt').write_text('s0\n')
    row = {'profile': 'ronin_resnet18_3d', 'dataset': 'native', 'data_root': str(native),
           'seed': 1735231569,
           'input_track': 'deployment', 'classic': False, 'dims': 3, 'state': 'completed',
           'dataset_fingerprint': 'data-fp', 'dataset_manifest_sha256': 'manifest',
           'split_sha256': {'test.txt': module.sha(native / 'splits' / 'test.txt')},
           'profile_config_sha256': 'model-yaml',
           'scientific_input_spec': {'window': 200}, 'sampling_spec': {'stride': 200, 'eval_stride': 10},
           'taskview_receipt': {'status': 'passed', 'splits': {
               'test': {'status': 'passed', 'valid_windows': 1}}}}
    policy = {'overrides': {'batch': 128, 'epochs': 30, 'workers': 2, 'stride': 200, 'eval_stride': 10}}
    expected = scientific_run_identity(BenchmarkItem(row['profile'], row['profile'], row['data_root'], row['seed'],
        {**policy['overrides'], 'recipe': 'unified', 'device': '0', 'seed': row['seed'], 'metric_dims': 3},
        metadata={**row, 'policy_sha256': 'original-policy', 'source_manifest_sha256': 'original-source',
                  'seeds_record_sha256': 'original-seeds'}))
    checkpoint = tmp_path / 'best.pt'; checkpoint.write_bytes(b'checkpoint')
    identity = tmp_path / 'run-identity.json'; identity.write_text(json.dumps(expected))
    result_dir = tmp_path / 'test'; result_dir.mkdir()
    (result_dir / 'metrics.json').write_text(json.dumps({
        'mode': 'val', 'model': row['profile'], 'dataset': row['dataset'], 'split': 'test',
        'seed': row['seed'], 'protocol': {'metric_dims': 3},
        'metric_dims_requested': 3, 'metric_dims_effective': 3,
        'predicted_axes': ['x', 'y', 'z'], 'num_sequences': 1, 'num_skipped': 0,
        'metrics': {'ate': 1.0}, 'median': {'ate': 1.0},
        'std': {'ate': None}, 'count': {'ate': 1}}))
    (result_dir / 'sequences.csv').write_text(
        'sequence_id,skipped,metric_dims_requested,metric_dims_effective,predicted_axes,ate\n'
        's0,,3,3,"x,y,z",1.0\n')
    binding = _validate_result_split(row, 'test', run_dir=tmp_path)
    row.update(checkpoint=str(checkpoint), checkpoint_sha256=module.sha(checkpoint),
               run_identity_path=str(identity), run_identity_sha256=module.sha(identity),
               result={'splits': {'test': 'evaluated'}},
               result_artifacts={'schema': 'ib.completed-results.v1',
                   'applicable_splits': ['test'], 'not_applicable_splits': {},
                   'splits': {'test': binding}})
    module.verify_current_completed(row, policy, 'original-policy', 'original-source', 'original-seeds')

    from inertial_benchmark.engine import benchmark as benchmark_module
    original_identity = identity.read_bytes()
    replacement = identity.with_name('forged-run-identity.json')
    forged = json.loads(original_identity)
    forged['profile'] = 'forged-profile'
    replacement.write_text(json.dumps(forged))
    real_fdopen = benchmark_module.os.fdopen
    opens = 0

    def replace_identity_after_open(fd, *args, **kwargs):
        nonlocal opens
        opens += 1
        if opens == 2:  # checkpoint is first; run identity is second
            replacement.replace(identity)
        return real_fdopen(fd, *args, **kwargs)

    with monkeypatch.context() as race:
        race.setattr(benchmark_module.os, 'fdopen', replace_identity_after_open)
        with pytest.raises(RuntimeError, match='run_identity_path pathname changed'):
            module.verify_current_completed(
                row, policy, 'original-policy', 'original-source', 'original-seeds')
    identity.write_bytes(original_identity)

    for args in [('new-policy', 'original-source', 'original-seeds'),
                 ('original-policy', 'new-source', 'original-seeds'),
                 ('original-policy', 'original-source', 'new-seeds')]:
        with pytest.raises(RuntimeError, match='requested scientific identity mismatch'):
            module.verify_current_completed(row, policy, *args)
    with pytest.raises(RuntimeError, match='requested scientific identity mismatch'):
        module.verify_current_completed(row, {'overrides': {**policy['overrides'], 'epochs': 31}},
                                        'original-policy', 'original-source', 'original-seeds')
