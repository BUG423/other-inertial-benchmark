"""Bounded Agent CPU lifecycle on new excerpts from existing RIDI recordings.

Four train windows, two validation windows and one held-out engineering
recording. No full-dataset training or paper accuracy claim. Run from a NEW
research directory in a namespace with all formal inputs mounted read-only.
"""
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import torch

from inertial_benchmark import NIO
from inertial_benchmark.data.format import load_sequence, save_sequence
from inertial_benchmark.data.manifest import fingerprint, imu_content_hash
from inertial_benchmark.nn.registry import MODELS, import_models
from inertial_benchmark.utils.torch_utils import load_checkpoint
from adapter import CandidateAdapter


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root = Path.cwd()
    source = Path('/root/autodl-tmp/other-inertial-benchmark-v1/datasets/ipb-v1/full-core/ridi')
    fixture = root / 'engineering-fixture'
    fixture.mkdir(exist_ok=False)
    (fixture / 'splits').mkdir()
    entries, splits, source_hashes = {}, {}, {}
    for split, samples in (('train', 801), ('val', 401), ('test', 401)):
        sid = (source / 'splits' / (split + '.txt')).read_text().splitlines()[0].strip()
        path = source / 'sequences' / (sid + '.h5')
        assert path.stat().st_size < 64 * 1024 ** 2
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        seq = load_sequence(path)
        starts = [i for i in range(0, len(seq) - samples, 200) if seq.valid[i:i + samples].all()]
        assert starts, 'No bounded fully valid engineering excerpt'
        excerpt = seq.slice(starts[0], starts[0] + samples)
        excerpt.timestamp = excerpt.timestamp - excerpt.timestamp[0]
        dest = save_sequence(fixture / 'sequences' / (sid + '.h5'), excerpt)
        entries[sid] = dict(file='sequences/' + sid + '.h5', sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                            group_id=excerpt.group_id, imu_sha256=imu_content_hash(excerpt.gyroscope, excerpt.accelerometer))
        splits[split] = [sid]
        (fixture / 'splits' / (split + '.txt')).write_text(sid + '\n')
    assert len(entries) == 3
    (fixture / 'dataset.json').write_text(json.dumps(dict(dataset='ridi-engineering', engineering_only=True,
        sequences=entries, splits=splits, fingerprint=fingerprint({k: v['sha256'] for k, v in entries.items()}, splits))))
    import_models()
    registry_before = dict(MODELS)
    arch = 'coherent_spectral_io_private'
    assert arch not in MODELS
    MODELS[arch] = CandidateAdapter
    options = dict(data=str(fixture), device='cpu', metric_dims=3, epochs=1, min_epochs=1,
                   batch=2, val_batch=2, workers=0, cache=False, amp=False, seed=1735231569,
                   augment=[], stride=200, eval_stride=200, efficiency=False, plots=False,
                   save_predictions=True, project=str(root / 'native-runs'), name='candidate',
                   gap_windows=2, optimizer='adamw', lr=.001, weight_decay=.0001)
    profile = dict(name=arch, arch=arch, registered=False, loss='mse',
                   args=dict(width=48, depth=3, spectral=True, cross_sensor=True, moments=True),
                   input=dict(window=200, rate=200, dims=3, frame='gravity_world', orientation='reference',
                              remove_gravity=False, target='avg_velocity', history=0, extra_inputs=[]))
    start = time.time()
    try:
        nio = NIO(profile, device='cpu', metric_dims=3)
        updates = []
        nio.add_callback('on_train_batch_end', lambda trainer: updates.append(1))
        nio.train(**options)
        assert len(updates) == 2
        assert load_checkpoint(nio.trainer.last)['training_evidence']['total_optimizer_updates'] == 2
        nio.callbacks.clear()
        validation = nio.val(**{**options, 'split': 'val', 'name': 'before_reload'})
        assert np.isfinite(validation.metrics['ate'])
        held_out = fixture / 'sequences' / (splits['test'][0] + '.h5')
        nio.predict(held_out, **options)
        expected = nio.results[0].vel_pred.copy()
        loaded = NIO(nio.ckpt_path, device='cpu', metric_dims=3)
        loaded.predict(held_out, **options)
        np.testing.assert_array_equal(loaded.results[0].vel_pred, expected)
        package = loaded.export(format='native', path=root / 'native-export')
        restored = NIO(package, device='cpu', metric_dims=3)
        restored.predict(held_out, **options)
        np.testing.assert_array_equal(restored.results[0].vel_pred, expected)
        assert expected.shape[-1] == 3 and np.isfinite(expected).all()
        for filename, sha in source_hashes.items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == sha
        receipt = dict(status='passed', engineering_only=True, elapsed_s=time.time() - start,
                       optimizer_updates=len(updates), train_windows=len(nio.trainer.train_set),
                       fixture_ids=splits, source_hashes=source_hashes, params=nio.model.num_params,
                       checkpoint_reload_exact=True, native_export_reload_exact=True,
                       cuda_initialized=torch.cuda.is_initialized(), torch=torch.__version__,
                       full_dataset_accuracy_established=False)
        (root / 'native-receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))
    finally:
        MODELS.clear()
        MODELS.update(registry_before)


if __name__ == '__main__':
    main()
