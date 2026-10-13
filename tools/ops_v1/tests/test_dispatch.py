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


def test_completed_cell_reuse_rejects_changed_policy_source_and_seed_record(tmp_path):
    module = ops()
    from inertial_benchmark.engine.benchmark import BenchmarkItem, scientific_run_identity
    import json
    row = {'profile': 'ronin_resnet18_3d', 'data_root': '/a/native', 'seed': 1735231569,
           'input_track': 'deployment', 'classic': False, 'dims': 3, 'state': 'completed',
           'dataset_fingerprint': 'data-fp', 'dataset_manifest_sha256': 'manifest',
           'split_sha256': {'train.txt': 'split'}, 'profile_config_sha256': 'model-yaml',
           'scientific_input_spec': {'window': 200}, 'sampling_spec': {'stride': 200, 'eval_stride': 10}}
    policy = {'overrides': {'batch': 128, 'epochs': 30, 'workers': 2, 'stride': 200, 'eval_stride': 10}}
    expected = scientific_run_identity(BenchmarkItem(row['profile'], row['profile'], row['data_root'], row['seed'],
        {**policy['overrides'], 'recipe': 'unified', 'device': '0', 'seed': row['seed'], 'metric_dims': 3},
        metadata={**row, 'policy_sha256': 'original-policy', 'source_manifest_sha256': 'original-source',
                  'seeds_record_sha256': 'original-seeds'}))
    checkpoint = tmp_path / 'best.pt'; checkpoint.write_bytes(b'checkpoint')
    identity = tmp_path / 'run-identity.json'; identity.write_text(json.dumps(expected))
    row.update(checkpoint=str(checkpoint), checkpoint_sha256=module.sha(checkpoint),
               run_identity_path=str(identity), run_identity_sha256=module.sha(identity))
    module.verify_current_completed(row, policy, 'original-policy', 'original-source', 'original-seeds')
    for args in [('new-policy', 'original-source', 'original-seeds'),
                 ('original-policy', 'new-source', 'original-seeds'),
                 ('original-policy', 'original-source', 'new-seeds')]:
        with pytest.raises(RuntimeError, match='requested scientific identity mismatch'):
            module.verify_current_completed(row, policy, *args)
    with pytest.raises(RuntimeError, match='requested scientific identity mismatch'):
        module.verify_current_completed(row, {'overrides': {**policy['overrides'], 'epochs': 31}},
                                        'original-policy', 'original-source', 'original-seeds')
