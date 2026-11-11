"""CPU readout intervention with frozen pretrained features and train-only fits."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from capacity_probe import metrics
from trajectory_gain_probe import readout


def objective(prediction, target, magnitude=False):
    loss = (prediction - target).square().mean()
    if magnitude:
        loss = loss + (prediction.norm(dim=-1) - target.norm(dim=-1)).square().mean() / 3
    return loss


def fit_head(initial, features, target, arm):
    head = copy.deepcopy(initial).requires_grad_(True)
    assert torch.equal(head.weight, initial.weight) and torch.equal(head.bias, initial.bias)
    if arm == 'least_squares':
        x = np.column_stack((features.numpy().astype(np.float64), np.ones(len(features))))
        y = target.numpy().astype(np.float64)
        original = np.vstack((initial.weight.detach().numpy().T, initial.bias.detach().numpy()))
        delta, _, rank, singular = np.linalg.lstsq(x, y - x @ original, rcond=1e-6)
        fitted = original + delta
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(fitted[:-1].T))
            head.bias.copy_(torch.from_numpy(fitted[-1]))
        detail = dict(updates=0, retained_rank=int(rank), design_columns=x.shape[1],
                      largest_singular=float(singular[0]), rcond=1e-6)
    elif arm in ('vector_mse', 'magnitude_aux'):
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0)
        for _ in range(256):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(head(features), target, magnitude=arm == 'magnitude_aux')
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step()
        detail = dict(updates=256, lr=.001, weight_decay=0, identical_initial_head=True)
    else:
        raise ValueError('unknown intervention arm')
    return head.requires_grad_(False), detail


@torch.no_grad()
def extract_features(model, inputs):
    return torch.cat([model.fc(model.transition(model.backbone(inputs[i:i+16])))
                      for i in range(0, len(inputs), 16)])


def main():
    from inertial_benchmark.data.format import load_sequence
    from inertial_benchmark.data.views import SequenceView, ViewConfig
    from inertial_benchmark.engine.predictor import Predictor
    from inertial_benchmark.models.ronin.model import RoNINResNet
    from inertial_benchmark.nn.base import InputSpec
    from inertial_benchmark.utils.torch_utils import load_checkpoint

    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    root, cfg = Path(os.environ['IPB_V1_RIDI_ROOT']), ViewConfig(dims=3)
    plan = json.loads(Path('head_intervention_plan.json').read_text())
    inputs = json.loads(Path('checkpoint-inputs-private.json').read_text())
    assert len(inputs) == 3 and {r['seed'] for r in inputs} == set(plan['seeds'])
    splits = {s: (root/'splits'/f'{s}.txt').read_text().split() for s in ('train', 'val')}
    assert len(splits['train']) == 45 and len(splits['val']) == 4
    assert not set(splits['train']) & set(splits['val'])
    files = [root/'splits'/f'{s}.txt' for s in splits]
    files += [root/'sequences'/f'{sid}.h5' for ids in splits.values() for sid in ids]
    files += [Path(row['checkpoint']) for row in inputs]
    files += [Path('head_intervention_plan.json'), Path('checkpoint-inputs-private.json')]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in files}
    xs, ys, train_groups, selections = [], [], set(), []
    for sid in splits['train']:
        seq = load_sequence(root/'sequences'/f'{sid}.h5')
        view = SequenceView(seq, cfg)
        starts = view.starts(200, require_valid=True)
        starts = starts[(starts > 0) & (starts < len(seq)-cfg.window)]
        starts = starts[seq.valid_pose[starts-1] & seq.valid_pose[starts+1]]
        starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
        assert len(starts) >= 16
        starts = starts[np.linspace(0, len(starts)-1, 16, dtype=int)]
        xs.append(view.imu_windows(starts)); ys.append(view.targets(starts))
        train_groups.add(seq.group_id)
        selections.append(dict(recording=sid, starts=starts.tolist()))
    x, y = torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))
    validation = [load_sequence(root/'sequences'/f'{sid}.h5') for sid in splits['val']]
    val_groups = {seq.group_id for seq in validation}
    assert len(train_groups) == 8 and len(val_groups) == 1 and not train_groups & val_groups
    report = dict(schema='ib.frozen-head-intervention.v1', observed_utc=datetime.now(timezone.utc).isoformat(),
                  dataset='ridi', profile='ronin_resnet18_3d', train_windows=len(x),
                  train_recordings=45, train_groups=8, validation_recordings=4, validation_groups=1,
                  group_overlap=0, fit_split='train', test_accessed=False, head_only=True,
                  steps_per_optimization_arm=256, final_step_preselected=True, seeds=[])
    measures = ('ate_m', 'rte_10s_m', 'position_error_p95_m', 'rte_10s_p95_m', 'plr')
    for item in inputs:
        assert time.monotonic()-started < 180
        checkpoint = load_checkpoint(item['checkpoint'], map_location='cpu')
        spec = InputSpec.from_dict(checkpoint['input_spec'])
        assert all(getattr(spec, k) == getattr(cfg, k) for k in ViewConfig.__dataclass_fields__)
        assert spec.channels == InputSpec(dims=3).channels and checkpoint['model_cfg']['arch'] == 'ronin_resnet'
        model = RoNINResNet(spec, **dict(checkpoint['model_cfg'].get('args') or {})).eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model'], strict=True)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        initial_head = model.head.linear
        features = extract_features(model, x)
        with torch.no_grad():
            check = torch.cat([model(x[i:i+16])['vel'] for i in range(0, len(x), 16)])
            assert torch.allclose(initial_head(features), check, atol=1e-6, rtol=1e-6)
        row = dict(seed=item['seed'], trainable_parameters=sum(p.numel() for p in initial_head.parameters()), arms={})
        for arm in plan['arms']:
            assert time.monotonic()-started < 180
            head, detail = ((initial_head, dict(updates=0)) if arm == 'identity'
                            else fit_head(initial_head, features, y, arm))
            with torch.no_grad():
                training = metrics(head(features), y)
            model.head.linear = head
            predictor = Predictor(model=model, device=torch.device('cpu'),
                                  overrides=dict(device='cpu', eval_stride=200, val_batch=32, amp=False))
            records = []
            for seq in validation:
                result = predictor.predict_sequence(seq)
                assert not result.skipped and np.isfinite(result.pos_pred).all()
                m = readout(result.pos_pred, seq.position, seq.valid_pose, spec.rate, result.pos_oracle)
                assert m['rte_10s_scaled'] == 0
                records.append(m)
            row['arms'][arm] = dict(fit=detail, train=training, validation_recordings=records,
                validation_medians={k: float(np.median([r[k] for r in records])) for k in measures})
            model.head.linear = initial_head
            assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
        baseline = row['arms']['identity']['validation_recordings']
        for arm, values in row['arms'].items():
            values['validation_recordings_worse'] = {
                k: sum(r[k] > b[k] + 1e-10 for r, b in zip(values['validation_recordings'], baseline))
                for k in measures if k != 'plr'}
        report['seeds'].append(row)
    assert all(digest(Path(p)) == value for p, value in before.items())
    report.update(complete=True, original_model_restored=True, frozen_features_and_buffers_unchanged=True,
                  source_checkpoint_and_data_files_unchanged=True, cuda_initialized=torch.cuda.is_initialized(),
                  elapsed_s=time.monotonic()-started, saves_weights=False, formal_matrix_changed=False)
    assert not report['cuda_initialized']
    Path('input-bindings-private.json').write_text(json.dumps(dict(hashes=before, selection=selections)))
    Path('head-intervention-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
