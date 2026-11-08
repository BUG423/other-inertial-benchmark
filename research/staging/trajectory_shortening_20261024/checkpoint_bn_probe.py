"""Read-only CPU BN control on completed RIDI/RoNIN XYZ checkpoints.

Requires the capacity probe beside this file and private checkpoint bindings in
a fresh, isolated Agent directory. Uses train windows only; never saves weights.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView, ViewConfig
from inertial_benchmark.models.ronin.model import RoNINResNet
from inertial_benchmark.nn.base import InputSpec
from inertial_benchmark.utils.torch_utils import load_checkpoint

from capacity_probe import bn_variance_control, evaluate


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    root = Path(os.environ['IPB_V1_RIDI_ROOT'])
    split = root / 'splits/train.txt'
    sid = next(s.strip() for s in split.read_text().splitlines() if s.strip())
    path = root / 'sequences' / (sid + '.h5')
    assert path.stat().st_size < 64 * 1024 ** 2
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_bytes())
    assert len(inputs) == 3 and len({p['seed'] for p in inputs}) == 3
    files = [path, split] + [Path(p['checkpoint']) for p in inputs]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    seq = load_sequence(path)
    view = SequenceView(seq, ViewConfig(dims=3))
    starts = view.starts(200, require_valid=True)
    starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
    assert len(starts) >= 16
    starts = starts[np.linspace(0, len(starts) - 1, 16, dtype=int)]
    x, y = torch.from_numpy(view.imu_windows(starts)), torch.from_numpy(view.targets(starts))
    Path('input-bindings-private.json').write_text(json.dumps(dict(hashes=before, starts=starts.tolist())))
    report = dict(schema='ib.completed-checkpoint-bn-control.v1',
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  torch_version=torch.__version__, training_only=True,
                  profile='ronin_resnet18_3d', dataset='ridi', recordings=1,
                  unique_windows=16, optimizer_updates=0, runs=[])
    for item in inputs:
        checkpoint = Path(item['checkpoint'])
        assert checkpoint.stat().st_size < 64 * 1024 ** 2
        ckpt = load_checkpoint(checkpoint, map_location='cpu')
        spec = InputSpec.from_dict(ckpt['input_spec'])
        assert all(getattr(spec, k) == getattr(view.cfg, k) for k in ViewConfig.__dataclass_fields__)
        assert ckpt['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(ckpt['model_cfg'].get('args') or {}))
        model.load_state_dict(ckpt['model'], strict=True)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for repeat in (1, 8):
            assert time.monotonic() - started < 90, 'Diagnostic time budget exceeded'
            bx, by = x.repeat(repeat, 1, 1), y.repeat(repeat, 1)
            original = evaluate(model, bx, by)
            control = bn_variance_control(model, bx, by)
            assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
            report['runs'].append(dict(seed=item['seed'], inference_batch_size=len(bx),
                                       repeat=repeat, **original, **control))
    report.update(complete=len(report['runs']) == 6,
                  checkpoint_and_data_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                                         for p, h in before.items()),
                  original_parameters_and_buffers_restored=True,
                  cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic() - started)
    assert report['checkpoint_and_data_files_unchanged'] and not report['cuda_initialized']
    Path('checkpoint-bn-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
