"""Train-only CPU memorization control; not a generalization or causal claim.

Run only on the Agent in a fresh directory with read-only framework/data mounts,
CUDA hidden, two CPU threads and an external wall timeout. No model is registered.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch import nn

from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView, ViewConfig
from inertial_benchmark.models.ronin.model import RoNINResNet
from inertial_benchmark.nn.base import InputSpec


def metrics(pred, target):
    speed, truth = pred.norm(dim=-1), target.norm(dim=-1)
    moving = (speed > 1e-8) & (truth > .2)
    cosine = ((pred * target).sum(-1) / (speed * truth).clamp_min(1e-12)).clamp(-1, 1)
    return dict(mse=float((pred - target).square().mean()),
                relative_l2=float(((pred - target).square().sum() / target.square().sum()).sqrt()),
                speed_ratio=float(speed.sum() / truth.sum()),
                speed_bias_mps=float((speed - truth).mean()),
                direction_deg=float(torch.rad2deg(cosine[moving].acos()).mean()) if moving.any() else None,
                direction_windows=int(moving.sum()))


@torch.no_grad()
def evaluate(model, x, y):
    """Same inputs/weights, dropout off; isolate running- vs batch-stat BN."""
    model.eval()
    result = {'running_bn': metrics(model(x)['vel'], y)}
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()
    result['batch_bn'] = metrics(model(x)['vel'], y)
    for k, v in model.named_buffers():
        v.copy_(buffers[k])
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=192)
    parser.add_argument('--wall-seconds', type=int, default=600)
    args = parser.parse_args()
    assert 1 <= args.steps <= 192 and 1 <= args.wall_seconds <= 600
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    source = Path(os.environ['IPB_V1_RIDI_ROOT'])
    split = source / 'splits/train.txt'
    sid = next(s.strip() for s in split.read_text().splitlines() if s.strip())
    path = source / 'sequences' / (sid + '.h5')
    assert path.stat().st_size < 64 * 1024 ** 2
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in (path, split)}
    seq = load_sequence(path)
    cfg = ViewConfig(window=200, rate=200., dims=3, frame='gravity_world',
                     orientation='reference', target='avg_velocity')
    view = SequenceView(seq, cfg)
    starts = view.starts(200, require_valid=True)
    starts = starts[np.linalg.norm(view.targets(starts), axis=-1) > .2]
    assert len(starts) >= 16, 'Insufficient preselected moving train windows'
    starts = starts[np.linspace(0, len(starts) - 1, 16, dtype=int)]
    x, y = torch.from_numpy(view.imu_windows(starts)), torch.from_numpy(view.targets(starts))
    assert x.shape == (16, 6, 200) and y.shape == (16, 3)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert abs(metrics(y * .5, y)['speed_ratio'] - .5) < 1e-6
    private = dict(input_hashes=before, train_recording=sid, starts=starts.tolist())
    Path('inputs-private.json').write_text(json.dumps(private))
    report = dict(schema='ib.train-capacity-probe.v1', training_only=True, dataset='ridi',
                  recordings=1, windows=16, architecture='ronin_resnet18_3d',
                  steps_per_arm=args.steps, threads=2, wall_budget_s=args.wall_seconds,
                  optimizer=dict(name='AdamW', lr=.001, weight_decay=0),
                  augmentation=[], seeds=[1735231569, 843628571, 951150512],
                  arms=[0., .5], runs=[], complete=False, formal_matrix_changed=False,
                  full_dataset_accuracy_established=False)
    save = lambda: Path('capacity-receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for seed in report['seeds']:
        torch.manual_seed(seed)
        model = RoNINResNet(InputSpec.from_dict(cfg.to_dict()))
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        report['parameters'] = sum(p.numel() for p in model.parameters())
        for dropout in report['arms']:
            model.load_state_dict(initial)
            assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
            torch.manual_seed(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            run = dict(seed=seed, dropout=dropout, steps=0, history=[], identical_initial_state=True)
            report['runs'].append(run)
            run['history'].append(dict(step=0, **evaluate(model, x, y)))
            for step in range(1, args.steps + 1):
                if time.monotonic() - started >= args.wall_seconds:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x)['vel'] - y).square().mean()
                assert torch.isfinite(loss), 'Nonfinite loss'
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                run['steps'] = step
                if step % 32 == 0 or step == args.steps:
                    run['history'].append(dict(step=step, **evaluate(model, x, y)))
                    save()
            if run['history'][-1]['step'] != run['steps']:
                run['history'].append(dict(step=run['steps'], **evaluate(model, x, y)))
            save()
            if run['steps'] < args.steps:
                break
        if report['runs'][-1]['steps'] < args.steps:
            break
    report.update(complete=len(report['runs']) == 6 and all(r['steps'] == args.steps for r in report['runs']),
                  input_files_unchanged=all(digest(Path(p)) == h for p, h in before.items()),
                  elapsed_s=time.monotonic() - started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  cuda_initialized=torch.cuda.is_initialized())
    assert report['input_files_unchanged'] and not report['cuda_initialized']
    save()
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
