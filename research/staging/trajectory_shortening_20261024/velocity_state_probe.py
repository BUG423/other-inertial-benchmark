"""Train-only boundary-state diagnostic; reference velocity is privileged.

Run on the Agent in a new namespace with read-only data/framework/checkpoints,
hidden CUDA, two CPU threads and an external timeout. No fitting or weight save.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView, ViewConfig, end_velocity
from inertial_benchmark.models.ronin.model import RoNINResNet
from inertial_benchmark.nn.base import InputSpec
from inertial_benchmark.utils.geometry import GRAVITY
from inertial_benchmark.utils.torch_utils import load_checkpoint

from capacity_probe import metrics


def average_velocity_increment(acceleration, dt):
    """Trapezoid-integrate acceleration, then average the velocity increment."""
    a = np.asarray(acceleration, dtype=np.float64)
    assert a.ndim == 3 and a.shape[1] >= 2 and a.shape[2] == 3 and dt > 0
    dv = np.concatenate([np.zeros((len(a), 1, 3)),
                         np.cumsum(.5 * (a[:, 1:] + a[:, :-1]) * dt, axis=1)], axis=1)
    return (.5 * (dv[:, 1:] + dv[:, :-1])).mean(axis=1)


def boosted_view(seq, cfg, boost):
    """Add constant world velocity in memory; acceleration/orientation stay fixed."""
    position = seq.position + (seq.timestamp - seq.timestamp[0])[:, None] * boost
    velocity = None if seq.velocity is None else seq.velocity + boost
    return SequenceView(replace(seq, position=position, velocity=velocity), cfg)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=int, default=45)
    args = parser.parse_args()
    assert 1 <= args.recordings <= 45 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    root = Path(os.environ['IPB_V1_RIDI_ROOT'])
    split = root / 'splits/train.txt'
    ids = [s.strip() for s in split.read_text().splitlines() if s.strip()]
    assert args.recordings <= len(ids) <= 128 and len(set(ids)) == len(ids)
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_bytes())
    assert len(inputs) == 3 and len({p['seed'] for p in inputs}) == 3
    files = [split] + [root / 'sequences' / (sid + '.h5') for sid in ids[:args.recordings]]
    files += [Path(p['checkpoint']) for p in inputs]
    assert all(p.stat().st_size < 64 * 1024 ** 2 for p in files)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in files}
    cfg = ViewConfig(window=200, rate=200., dims=3, frame='gravity_world',
                     orientation='reference', target='avg_velocity', remove_gravity=False)
    models, states = {}, {}
    for item in inputs:
        ckpt = load_checkpoint(item['checkpoint'], map_location='cpu')
        spec = InputSpec.from_dict(ckpt['input_spec'])
        assert all(getattr(spec, k) == getattr(cfg, k) for k in ViewConfig.__dataclass_fields__)
        assert spec.channels == InputSpec(dims=3).channels and ckpt['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(ckpt['model_cfg'].get('args') or {}))
        model.load_state_dict(ckpt['model'], strict=True)
        models[str(item['seed'])] = model.eval()
        states[str(item['seed'])] = {k: v.clone() for k, v in model.state_dict().items()}
    rows, groups, bindings = [], set(), []
    boost = np.array([.5, 0., 0.])
    for sid in ids[:args.recordings]:
        assert time.monotonic() - started < 180, 'Diagnostic time budget exceeded'
        seq = load_sequence(root / 'sequences' / (sid + '.h5'))
        view = SequenceView(seq, cfg)
        starts = view.starts(200, require_valid=True)
        starts = starts[(starts > 0) & (starts < len(seq) - cfg.window)]
        starts = starts[seq.valid_pose[starts - 1] & seq.valid_pose[starts + 1]]
        starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
        assert len(starts) >= 16, 'Insufficient preselected train windows'
        starts = starts[np.linspace(0, len(starts) - 1, 16, dtype=int)]
        x, y = view.imu_windows(starts), view.targets(starts)
        v0 = end_velocity(seq.position, seq.velocity, starts, cfg.rate)
        acceleration = x[:, 3:, :].transpose(0, 2, 1).astype(np.float64)
        acceleration -= np.array([0., 0., GRAVITY])
        delta = average_velocity_increment(acceleration, cfg.dt)
        shifted = boosted_view(seq, cfg, boost)
        input_same = np.array_equal(x, shifted.imu_windows(starts))
        target_error = float(np.max(np.abs(shifted.targets(starts) - y - boost)))
        shifted_v0 = end_velocity(shifted.seq.position, shifted.seq.velocity, starts, cfg.rate)
        boundary_error = float(np.max(np.abs(shifted_v0 - v0 - boost)))
        assert input_same and target_error < 1e-5 and boundary_error < 1e-5
        predictions = dict(zero_state=delta, oracle_state=v0 + delta,
                           shuffled_state=np.roll(v0, 8, axis=0) + delta,
                           oracle_persistence=v0)
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        measured = {name: metrics(torch.as_tensor(p), ty) for name, p in predictions.items()}
        for seed, model in models.items():
            measured['checkpoint_' + seed] = metrics(model(tx)['vel'], ty)
        rows.append(dict(metrics=measured, input_unchanged=input_same,
                         target_shift_max_error_mps=target_error,
                         boundary_shift_max_error_mps=boundary_error))
        groups.add(seq.group_id)
        bindings.append(dict(recording=sid, group_id=seq.group_id, starts=starts.tolist(),
                             initial_velocity_source='provided' if seq.velocity is not None else 'pose_central_difference'))
    assert all(torch.equal(v, states[seed][k]) for seed, model in models.items()
               for k, v in model.state_dict().items())
    assert all(digest(Path(p)) == h for p, h in before.items())
    private = dict(hashes=before, selection=bindings, per_recording=rows)
    Path('velocity-state-private.json').write_text(json.dumps(private, allow_nan=False))
    summary = {name: {key: dict(median=float(np.median([r['metrics'][name][key] for r in rows])),
                               min=float(min(r['metrics'][name][key] for r in rows)),
                               max=float(max(r['metrics'][name][key] for r in rows)))
                       for key in ('speed_ratio', 'relative_l2', 'speed_bias_mps')}
               for name in rows[0]['metrics']}
    paired = {name: dict(recordings=len(rows),
                        oracle_state_lower_relative_l2=sum(r['metrics']['oracle_state']['relative_l2'] < r['metrics'][name]['relative_l2'] for r in rows),
                        median_relative_l2_delta=float(np.median([r['metrics']['oracle_state']['relative_l2'] - r['metrics'][name]['relative_l2'] for r in rows])))
              for name in ('zero_state', 'shuffled_state', 'oracle_persistence')}
    report = dict(schema='ib.velocity-boundary-diagnostic.v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                  torch_version=torch.__version__, dataset='ridi', split='train', complete=True,
                  recordings=len(rows), existing_groups=len(groups), windows_per_recording=16,
                  total_windows=16 * len(rows), window=200, rate=200., threads=2,
                  selection='Train-list prefix; sixteen uniformly spaced valid moving nonoverlapping windows, with valid start-neighbor poses.',
                  optimizer_updates=0, weights_and_buffers_unchanged=True, input_files_unchanged=True,
                  galilean_control=dict(boost_mps=boost.tolist(), identical_inputs=sum(r['input_unchanged'] for r in rows),
                                        target_shift_max_error_mps=max(r['target_shift_max_error_mps'] for r in rows),
                                        boundary_shift_max_error_mps=max(r['boundary_shift_max_error_mps'] for r in rows)),
                  recording_summary=summary, paired_controls=paired,
                  cuda_initialized=torch.cuda.is_initialized(), elapsed_s=time.monotonic() - started,
                  limitations=['Reference initial velocity and reference orientation are privileged; no deployable accuracy claim.',
                               'Galilean transformation is a constructed identifiability control, not measured ambiguity in the observed dataset.',
                               'Reference initial velocity comes from provided velocity or neighboring reference poses.',
                               'Persistence is a reference-only autocorrelation control; integration adds no learned parameters.',
                               'Within-recording roll preserves initial-state values, not independence or exchangeability.',
                               'Recording medians are descriptive; recordings may share subjects and three seeds share the same data.',
                               'No validation/test evaluation, ATE/RTE claim, or attribution of the full-matrix shortening.'])
    assert not report['cuda_initialized']
    Path('velocity-state-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
