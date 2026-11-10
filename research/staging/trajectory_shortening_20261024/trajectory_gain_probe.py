"""Read-only full-validation trajectories with previously frozen train-fit gains.

No fitting, training, test split access or checkpoint writes. Run on the Agent
with CPU-only, read-only inputs in a separate research directory.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def scale_trajectory(position, anchor, gain):
    """Scaling velocities scales displacements, preserving the original anchor."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('gain must be finite and positive')
    return anchor + gain * (position - anchor)


def readout(pred, target, valid, rate, oracle):
    from inertial_benchmark.metrics.trajectory import ate, ate_aligned, length_ratios, relative_error

    detail = relative_error(pred, target, valid, rate, delta=10., dims=3)
    lag = int(round(10 * rate))
    errors = np.linalg.norm(pred[valid] - target[valid], axis=1)
    pairs = valid[:-lag] & valid[lag:]
    drift = (pred[lag:] - pred[:-lag]) - (target[lag:] - target[:-lag])
    return dict(ate_m=ate(pred, target, valid, dims=3),
                ate_aligned_m=ate_aligned(pred, target, valid, dims=3),
                rte_10s_m=detail['value'], rte_10s_scaled=detail['scaled'],
                rte_10s_pairs=detail['pairs'],
                position_error_p95_m=float(np.quantile(errors, .95)),
                rte_10s_p95_m=float(np.quantile(np.linalg.norm(drift[pairs], axis=1), .95))
                if pairs.any() else None,
                **length_ratios(pred, target, valid, rate, oracle=oracle, dims=3))


def main():
    import torch
    from inertial_benchmark.data.format import load_sequence
    from inertial_benchmark.data.views import ViewConfig
    from inertial_benchmark.engine.predictor import Predictor
    from inertial_benchmark.models.ronin.model import RoNINResNet
    from inertial_benchmark.nn.base import InputSpec
    from inertial_benchmark.utils.torch_utils import load_checkpoint

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    root = Path(os.environ['IPB_V1_RIDI_ROOT'])
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_text())
    prior = json.loads(Path('frozen-train-gains.json').read_text())
    assert prior['complete'] and prior['gain_fit_split'] == 'train' and not prior['test_accessed']
    assert prior['group_overlap'] == 0 and prior['splits']['val']['recordings'] == 4
    gains = {row['seed']: row['train_fit_gains'] for row in prior['seeds']}
    assert len(inputs) == len(gains) == 3 and {row['seed'] for row in inputs} == set(gains)
    split = root / 'splits/val.txt'
    ids = [s.strip() for s in split.read_text().splitlines() if s.strip()]
    assert len(ids) == len(set(ids)) == 4
    files = [split, Path('checkpoint-inputs-private.json'), Path('frozen-train-gains.json')]
    files += [root / 'sequences' / f'{sid}.h5' for sid in ids]
    files += [Path(row['checkpoint']) for row in inputs]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in files}
    sequences = [load_sequence(root / 'sequences' / f'{sid}.h5') for sid in ids]
    assert len({seq.group_id for seq in sequences}) == 1
    report = dict(schema='ib.trajectory-gain-diagnostic.v1',
                  observed_utc=datetime.now(timezone.utc).isoformat(), dataset='ridi',
                  profile='ronin_resnet18_3d', split='val', recordings=4, groups=1,
                  gain_fit_split='train', gains_refitted=False, eval_stride=200,
                  optimizer_updates=0, test_accessed=False, seeds=[])
    metrics = ('ate_m', 'ate_aligned_m', 'rte_10s_m', 'position_error_p95_m', 'rte_10s_p95_m', 'plr')
    for item in inputs:
        assert time.monotonic() - started < 180
        checkpoint = load_checkpoint(item['checkpoint'], map_location='cpu')
        spec = InputSpec.from_dict(checkpoint['input_spec'])
        cfg = ViewConfig(dims=3)
        assert all(getattr(spec, k) == getattr(cfg, k) for k in ViewConfig.__dataclass_fields__)
        assert spec.channels == InputSpec(dims=3).channels
        assert checkpoint['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(checkpoint['model_cfg'].get('args') or {})).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        predictor = Predictor(model=model, device=torch.device('cpu'),
                              overrides=dict(device='cpu', eval_stride=200, val_batch=32, amp=False))
        rows = []
        for index, seq in enumerate(sequences):
            result = predictor.predict_sequence(seq)
            assert not result.skipped and np.isfinite(result.pos_pred).all()
            anchor = seq.position[np.flatnonzero(seq.valid_pose)[0]]
            arms = {}
            for name, gain in dict(identity=1., **gains[item['seed']]).items():
                position = scale_trajectory(result.pos_pred, anchor, gain)
                arms[name] = readout(position, seq.position, seq.valid_pose, spec.rate, result.pos_oracle)
                assert arms[name]['rte_10s_scaled'] == 0 and arms[name]['rte_10s_pairs'] > 0
            rows.append(dict(recording_index=index, windows=len(result.starts), arms=arms))
        assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        summaries = {}
        for arm in rows[0]['arms']:
            summaries[arm] = dict(
                recording_medians={k: float(np.median([r['arms'][arm][k] for r in rows])) for k in metrics},
                recordings_worse_than_identity={k: sum(r['arms'][arm][k] > r['arms']['identity'][k] + 1e-10
                                                       for r in rows) for k in metrics if k != 'plr'})
        report['seeds'].append(dict(seed=item['seed'], frozen_gains=gains[item['seed']],
                                    recordings=rows, summary=summaries))
    assert all(digest(Path(p)) == h for p, h in before.items())
    report.update(complete=True, input_files_unchanged=True, weights_and_buffers_unchanged=True,
                  cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic()-started,
                  limitations=['Validation has four recordings from one group, used for checkpoint selection.',
                               'Gains were frozen in an earlier train-window diagnostic; no validation refit.',
                               'Full trajectories include stationary windows; no speed-based selection.',
                               'ATE is unaligned; aligned ATE is separate and never fits scale.',
                               'P95 and ten-second errors are within-recording summaries, not independent subjects.',
                               'No test accuracy, new method, loss intervention or causal explanation established.'])
    assert not report['cuda_initialized']
    Path('input-bindings-private.json').write_text(json.dumps(before))
    Path('trajectory-gain-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
