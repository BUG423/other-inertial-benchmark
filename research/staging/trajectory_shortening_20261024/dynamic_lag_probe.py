"""Read-only dynamics stratification for trajectory shortening.

The probe compares completed checkpoints on fixed train/validation windows.
Reference endpoint velocity is used only to define a descriptive dynamics
stratum; no target-derived correction is applied to predictions.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def dynamics(seq, starts, cfg, sequence_view_cls, end_velocity_fn):
    view = sequence_view_cls(seq, cfg)
    target = view.targets(starts).astype(np.float64)
    first = end_velocity_fn(seq.position, seq.velocity, starts, cfg.rate)
    last = end_velocity_fn(seq.position, seq.velocity, starts + cfg.window - 1, cfg.rate)
    delta = np.linalg.norm(last - first, axis=1)
    acceleration = view.imu_windows(starts)[:, 3:, :].astype(np.float64)
    roughness = np.sqrt(np.mean(np.diff(acceleration, axis=2) ** 2, axis=(1, 2)))
    return target, delta, roughness


def bins(values, edges):
    values, edges = np.asarray(values, float), np.asarray(edges, float)
    return np.clip(np.digitize(values, edges, right=False), 0, len(edges))


def summarize(pred, target, delta, roughness, labels):
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    p_speed, t_speed = np.linalg.norm(pred, axis=1), np.linalg.norm(target, axis=1)
    cosine = np.clip((pred * target).sum(1) /
                     np.maximum(p_speed * t_speed, 1e-12), -1, 1)
    speed_error = p_speed - t_speed
    along_error = ((pred - target) * target).sum(1) / np.maximum(t_speed, 1e-12)
    out = dict(windows=len(pred), speed_ratio=float(p_speed.sum() / t_speed.sum()),
               speed_bias_mps=float(speed_error.mean()), along_bias_mps=float(along_error.mean()),
               direction_deg=float(np.degrees(np.arccos(cosine)).mean()),
               delta_speed_correlation=correlation(delta, speed_error),
               roughness_speed_correlation=correlation(roughness, speed_error), bins=[])
    for label in range(len(np.unique(labels))):
        keep = labels == label
        if not keep.any():
            continue
        out['bins'].append(dict(label=int(label), windows=int(keep.sum()),
                                delta_median=float(np.median(delta[keep])),
                                roughness_median=float(np.median(roughness[keep])),
                                speed_ratio=float(p_speed[keep].sum() / t_speed[keep].sum()),
                                speed_bias_mps=float(speed_error[keep].mean()),
                                along_bias_mps=float(along_error[keep].mean()),
                                direction_deg=float(np.degrees(np.arccos(cosine[keep])).mean())))
    return out


def main():
    import torch

    from inertial_benchmark.data.format import load_sequence
    from inertial_benchmark.data.views import SequenceView, ViewConfig, end_velocity
    from inertial_benchmark.models.ronin.model import RoNINResNet
    from inertial_benchmark.nn.base import InputSpec
    from inertial_benchmark.utils.torch_utils import load_checkpoint

    # The probe is intentionally CPU-only and read-only.  Keep the heavy
    # framework imports inside main so the analytical helpers remain usable by
    # the standard-library test environment.
    with torch.no_grad():
        _run(torch, load_sequence, SequenceView, ViewConfig, end_velocity,
             RoNINResNet, InputSpec, load_checkpoint)


def _run(torch, load_sequence, SequenceView, ViewConfig, end_velocity,
         RoNINResNet, InputSpec, load_checkpoint):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    root = Path(os.environ['IPB_V1_RIDI_ROOT'])
    splits = {name: [s.strip() for s in (root / 'splits' / f'{name}.txt').read_text().splitlines() if s.strip()]
              for name in ('train', 'val')}
    assert len(splits['train']) == 45 and 1 <= len(splits['val']) <= 16
    assert not set(splits['train']) & set(splits['val'])
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_bytes())
    assert len(inputs) == 3 and len({item['seed'] for item in inputs}) == 3
    files = [root / 'splits' / f'{name}.txt' for name in splits]
    files += [root / 'sequences' / f'{sid}.h5' for ids in splits.values() for sid in ids]
    files += [Path(item['checkpoint']) for item in inputs]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in files}
    cfg = ViewConfig(window=200, rate=200., dims=3, frame='gravity_world',
                     orientation='reference', target='avg_velocity')
    arrays, selections = {}, {}
    for split, ids in splits.items():
        xs, ys, deltas, roughnesses, groups = [], [], [], [], []
        selections[split] = []
        for sid in ids:
            seq = load_sequence(root / 'sequences' / f'{sid}.h5')
            view = SequenceView(seq, cfg)
            starts = view.starts(200, require_valid=True)
            starts = starts[(starts > 0) & (starts < len(seq) - cfg.window)]
            starts = starts[seq.valid_pose[starts - 1] & seq.valid_pose[starts + 1]]
            starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
            assert len(starts) >= 16
            starts = starts[np.linspace(0, len(starts) - 1, 16, dtype=int)]
            target, delta, roughness = dynamics(seq, starts, cfg, SequenceView, end_velocity)
            xs.append(view.imu_windows(starts)); ys.append(target)
            deltas.append(delta); roughnesses.append(roughness); groups.append(seq.group_id)
            selections[split].append(dict(recording=sid, group_id=seq.group_id, starts=starts.tolist()))
        arrays[split] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(deltas),
                         np.concatenate(roughnesses), groups)
    train_delta = arrays['train'][2]
    edges = np.quantile(train_delta, [0.25, 0.5, 0.75]).tolist()
    report = dict(schema='ib.dynamic-lag-diagnostic.v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                  dataset='ridi', checkpoint_profile='ronin_resnet18_3d', torch_version=torch.__version__,
                  split_shapes={split: dict(recordings=len(ids), windows=16 * len(ids), groups=len(set(arrays[split][4])))
                                for split, ids in splits.items()},
                  group_overlap=len(set(arrays['train'][4]) & set(arrays['val'][4])),
                  dynamics_definition='Endpoint reference velocity change norm across each 200-sample window; IMU roughness is RMS first difference of world-frame accelerometer channels.',
                  train_quantile_edges_mps=edges, optimizer_updates=0, test_accessed=False, seeds=[])
    for item in inputs:
        assert time.monotonic() - started < 180
        checkpoint = load_checkpoint(item['checkpoint'], map_location='cpu')
        spec = InputSpec.from_dict(checkpoint['input_spec'])
        assert all(getattr(spec, key) == getattr(cfg, key) for key in ViewConfig.__dataclass_fields__)
        assert spec.channels == InputSpec(dims=3).channels and checkpoint['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(checkpoint['model_cfg'].get('args') or {})).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        predictions = {}
        for split, (x, _, _, _, _) in arrays.items():
            predictions[split] = np.concatenate([model(torch.from_numpy(x[i:i + 16]))['vel'].numpy()
                                                 for i in range(0, len(x), 16)])
        assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
        row = dict(seed=item['seed'], splits={})
        for split, (_, target, delta, roughness, _) in arrays.items():
            labels = bins(delta, edges)
            row['splits'][split] = summarize(predictions[split], target, delta, roughness, labels)
        report['seeds'].append(row)
    assert all(digest(Path(path)) == value for path, value in before.items())
    Path('input-bindings-private.json').write_text(json.dumps(dict(hashes=before, selections=selections)))
    report.update(complete=True, input_files_unchanged=True, weights_and_buffers_unchanged=True,
                  cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic() - started,
                  limitations=['Descriptive train/validation windows only; no test, trajectory ATE/RTE or causal intervention.',
                               'Quantile thresholds are defined on train labels and reused on validation; they are not a deployable feature.',
                               'Endpoint reference velocity is a diagnostic label and can mix boundary-state and dynamic effects.',
                               'RMS accelerometer difference depends on sensor noise and frame conversion; correlation is not a lag proof.',
                               'Recordings and seeds are not independent observations; validation contains one group.'])
    assert not report['cuda_initialized']
    Path('dynamic-lag-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
