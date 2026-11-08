"""Read-only radial loss diagnostic; train-fit gains, one validation readout.

Run on the Agent in a fresh CPU namespace. Target-derived readouts are diagnostic
controls, never candidate predictions. No test access, model updates or saves.
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


def gain_controls(prediction, target):
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    assert p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all()
    assert np.square(p).sum() > 0
    return dict(vector_mse=max(0., float((p * y).sum() / np.square(p).sum())),
                norm_sum=float(np.linalg.norm(y, axis=1).sum() / np.linalg.norm(p, axis=1).sum()))


def radial_statistics(prediction, target):
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    assert p.shape == y.shape and p.ndim == 2 and p.shape[1] == 3
    assert np.isfinite(p).all() and np.isfinite(y).all()
    r, s = np.linalg.norm(p, axis=1), np.linalg.norm(y, axis=1)
    keep = (r > 1e-8) & (s > .2)
    if not keep.any():
        return dict(windows=len(p), usable_windows=0)
    p, y, r, s = p[keep], y[keep], r[keep], s[keep]
    u = p / r[:, None]
    projection = (u * y).sum(axis=1)
    cosine = np.clip(projection / s, -1., 1.)
    radial = (r - s) ** 2
    directional = 2 * r * s * (1 - cosine)
    error = ((p - y) ** 2).sum(axis=1)
    assert np.allclose(error, radial + directional, atol=1e-10, rtol=1e-9)
    norm_matched = s[:, None] * u
    projected = np.maximum(projection, 0)[:, None] * u
    oracle_error = ((norm_matched - y) ** 2).sum(axis=1)
    optimum_error = ((projected - y) ** 2).sum(axis=1)
    assert np.all(optimum_error <= error + 1e-10)
    return dict(windows=len(keep), usable_windows=int(keep.sum()),
                speed_ratio=float(r.sum() / s.sum()), relative_l2=float(np.sqrt(error.sum() / (s*s).sum())),
                mean_speed_bias_mps=float((r - s).mean()), mean_direction_deg=float(np.degrees(np.arccos(cosine)).mean()),
                radial_mse=float(radial.mean()), directional_mse=float(directional.mean()),
                vector_mse=float(error.mean()),
                mean_prediction_direction_residual_mps=float((r - projection).mean()),
                fraction_below_directional_optimum=float((r < np.maximum(projection, 0)).mean()),
                oracle_norm_relative_l2=float(np.sqrt(oracle_error.sum() / (s*s).sum())),
                oracle_projection_relative_l2=float(np.sqrt(optimum_error.sum() / (s*s).sum())),
                oracle_norm_worse_window_fraction=float((oracle_error > error + 1e-10).mean()))


def strata(p, y):
    speed = np.linalg.norm(y, axis=1)
    direction = np.degrees(np.arccos(np.clip((p * y).sum(1) /
                           np.maximum(np.linalg.norm(p, axis=1) * speed, 1e-12), -1, 1)))
    masks = dict(speed_02_to_08=(speed >= .2) & (speed < .8),
                 speed_08_to_12=(speed >= .8) & (speed < 1.2), speed_ge12=speed >= 1.2,
                 direction_lt20=direction < 20, direction_20_to_45=(direction >= 20) & (direction < 45),
                 direction_ge45=direction >= 45)
    return {name: radial_statistics(p[mask], y[mask]) for name, mask in masks.items()}


@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    root = Path(os.environ['IPB_V1_RIDI_ROOT'])
    splits = {s: [v.strip() for v in (root/'splits'/f'{s}.txt').read_text().splitlines() if v.strip()]
              for s in ('train', 'val')}
    assert len(splits['train']) == 45 and 1 <= len(splits['val']) <= 16
    assert not set(splits['train']) & set(splits['val'])
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_bytes())
    assert len(inputs) == 3 and len({r['seed'] for r in inputs}) == 3
    files = [root/'splits'/f'{s}.txt' for s in splits]
    files += [root/'sequences'/f'{sid}.h5' for ids in splits.values() for sid in ids]
    files += [Path(r['checkpoint']) for r in inputs]
    assert all(p.stat().st_size < 64*1024**2 for p in files)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in files}
    cfg, arrays, private = ViewConfig(dims=3), {}, {}
    for split, ids in splits.items():
        xs, ys, groups = [], [], []
        private[split] = []
        for sid in ids:
            seq = load_sequence(root/'sequences'/f'{sid}.h5')
            view = SequenceView(seq, cfg)
            starts = view.starts(200, require_valid=True)
            starts = starts[(starts > 0) & (starts < len(seq) - cfg.window)]
            starts = starts[seq.valid_pose[starts-1] & seq.valid_pose[starts+1]]
            starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
            assert len(starts) >= 16
            starts = starts[np.linspace(0, len(starts)-1, 16, dtype=int)]
            xs.append(view.imu_windows(starts)); ys.append(view.targets(starts)); groups.append(seq.group_id)
            private[split].append(dict(recording=sid, group_id=seq.group_id, starts=starts.tolist()))
        arrays[split] = (np.concatenate(xs), np.concatenate(ys), groups)
    report = dict(schema='ib.radial-loss-diagnostic.v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                  dataset='ridi', checkpoint_profile='ronin_resnet18_3d', torch_version=torch.__version__,
                  splits={s: dict(recordings=len(ids), windows=16*len(ids), groups=len(set(arrays[s][2]))) for s, ids in splits.items()},
                  group_overlap=len(set(arrays['train'][2]) & set(arrays['val'][2])),
                  gain_fit_split='train', gain_objective='Unweighted sum of per-window vector squared errors or norm sums.',
                  optimizer_updates=0, test_accessed=False, seeds=[])
    for item in inputs:
        assert time.monotonic() - started < 180
        ckpt = load_checkpoint(item['checkpoint'], map_location='cpu')
        spec = InputSpec.from_dict(ckpt['input_spec'])
        assert all(getattr(spec, k) == getattr(cfg, k) for k in ViewConfig.__dataclass_fields__)
        assert spec.channels == InputSpec(dims=3).channels and ckpt['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(ckpt['model_cfg'].get('args') or {})).eval()
        model.load_state_dict(ckpt['model'], strict=True)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        predictions = {s: np.concatenate([model(torch.from_numpy(x[i:i+16]))['vel'].numpy()
                                          for i in range(0, len(x), 16)]) for s, (x, _, _) in arrays.items()}
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        gains = gain_controls(predictions['train'], arrays['train'][1])
        row = dict(seed=item['seed'], train_fit_gains=gains, splits={})
        for split, (_, y, _) in arrays.items():
            p = predictions[split]
            row['splits'][split] = dict(identity=radial_statistics(p, y), strata=strata(p, y),
                gain_readouts={name: radial_statistics(p * gain, y) for name, gain in gains.items()})
        report['seeds'].append(row)
    assert all(digest(Path(p)) == h for p, h in before.items())
    Path('input-bindings-private.json').write_text(json.dumps(dict(hashes=before, selections=private)))
    report.update(complete=True, weights_and_buffers_unchanged=True, input_files_unchanged=True,
                  cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic()-started,
                  limitations=['Sampled moving windows only; vector diagnostics are not trajectory ATE/RTE or PLR.',
                               'Target-derived norm and projection controls use labels and cannot be deployed.',
                               'Oracle projections identify a geometric error component, not a training-loss causal effect.',
                               'Train-fit scalar gains are diagnostic controls; validation was already used for original checkpoint selection.',
                               'Speed/direction bins condition on observed targets/errors and cannot identify causal mechanisms.',
                               'Shared recordings, subjects and checkpoints prevent interpreting pooled windows as independent observations.'])
    assert not report['cuda_initialized']
    Path('radial-loss-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
