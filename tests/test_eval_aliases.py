"""Scientific equivalence for opt-in exact final-test aliases (CPU only).

The native files below are synthetic engineering fixtures, not SubT accuracy
experiments. Real registered NIO LieEvents forwards/losses/reconstruction are
compared with the ordinary path, including post-constructor global RNG.
"""

import json
import random
import sys
import types
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytest
from synthetic import make_sequence

torch = pytest.importorskip('torch')
from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class ConvolutionCounter:
    def __init__(self):
        self.api1 = torch.conv1d
        self.api2 = torch.conv2d
        if (type(self.api1) is not types.BuiltinFunctionType
                or type(self.api2) is not types.BuiltinFunctionType):
            raise RuntimeError('Installed convolution APIs are not inspectable C builtins')
        self.by_api = {'torch.conv1d': 0, 'torch.conv2d': 0}
        self.active = False

    @property
    def count(self):
        return sum(self.by_api.values())

    def _observe(self, frame, event, arg):
        if event == 'c_call':
            if arg is self.api1:
                self.by_api['torch.conv1d'] += 1
            elif arg is self.api2:
                self.by_api['torch.conv2d'] += 1
        # No arguments, tensor data, frame or event trace is retained. The C
        # callable continues normally; sys.setprofile does not replace it.

    def __enter__(self):
        if self.active or sys.getprofile() is not None:
            raise RuntimeError('An external Python profiler is active')
        self.active = True
        sys.setprofile(self._observe)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.setprofile(None)
        finally:
            self.active = False


from inertial_benchmark.cfg import ConfigError, get_cfg  # noqa: E402
from inertial_benchmark.data.format import load_sequence, save_sequence  # noqa: E402
from inertial_benchmark.data.manifest import DatasetSpec, fingerprint  # noqa: E402
from inertial_benchmark.engine.benchmark import (  # noqa: E402
    BenchmarkItem, _run_item, load_plan, plan_item, scientific_run_identity,
)
from inertial_benchmark.engine.eval_aliases import (  # noqa: E402
    ALIAS_POLICY, ExactAliasSession, Ineligible, digest, file_sha256,
    model_signature, rng_signature, source_certificate,
)
from inertial_benchmark.engine.predictor import Predictor  # noqa: E402
from inertial_benchmark.engine.results import ARRAYS  # noqa: E402
from inertial_benchmark.engine.validator import Validator  # noqa: E402
from inertial_benchmark.nn.registry import build_model  # noqa: E402

SPLITS = ['test', 'test_ood_device_id', 'test_ood_placement', 'test_ood_position_source']
SMALL = {'group_sizes': [1, 1], 'base_plane': 4, 'trans_planes': 4, 'fc_dim': 8}


@pytest.fixture
def native(tmp_path):
    root = tmp_path / 'engineering-native'
    hashes = {}
    for i in range(2):
        seq = make_sequence(duration=1.5, seed=80 + i, sequence_id=f's{i}',
                            dataset='engineering-native', group_id=f'g{i}')
        # Reference gaps and independent input/target masks must also survive.
        seq.valid_pose[260:265] = False
        seq.valid_imu[270:275] = False
        path = save_sequence(root / 'sequences' / f's{i}.h5', seq)
        hashes[f's{i}'] = file_sha256(path)
    splits = {split: ['s0', 's1'] for split in SPLITS}
    (root / 'splits').mkdir()
    for split, ids in splits.items():
        (root / 'splits' / f'{split}.txt').write_text('\n'.join(ids) + '\n')
    (root / 'dataset.json').write_text(json.dumps({
        'name': 'engineering-native', 'fingerprint': fingerprint(hashes, splits),
        'sequences': hashes, 'scope': 'synthetic engineering fixture only'}))
    return DatasetSpec('engineering-native', root, test_splits=SPLITS)


def checkpoint(tmp_path, profile='nio_lie_events_ronin_3d', epoch=10):
    overrides = {'model': profile, 'device': 'cpu', 'workers': 0}
    if profile != 'imunet':
        overrides['model_args'] = SMALL
    cfg = get_cfg(overrides)
    model = build_model(cfg)
    path = tmp_path / f'{profile}-{epoch}.pt'
    torch.save({'model': model.state_dict(), 'model_cfg': model.model_cfg,
                'input_spec': model.input_spec.to_dict(), 'epoch': epoch,
                'cfg': cfg.to_dict()}, path)
    return path


def args(weights, native, split=SPLITS[0], **overrides):
    return get_cfg({'model': str(weights), 'data': str(native.root), 'split': split,
                    'device': 'cpu', 'workers': 0, 'val_batch': 2, 'eval_stride': 50,
                    'plots': False, 'save_predictions': True, 'efficiency': False,
                    **overrides})


def run_groups(weights, native, out, enabled, metric_dims=2):
    session = ExactAliasSession(native, SPLITS) if enabled else None
    results = []
    for split in SPLITS:
        result = Validator(args(weights, native, split, metric_dims=metric_dims), save_dir=out / split,
                           alias_session=session)(dataset=native)
        results.append(result)
    return results, rng_signature(), session


def reset_rng():
    random.seed(19)
    np.random.seed(29)
    torch.manual_seed(39)


@pytest.mark.parametrize('profile,metric_dims', [
    ('nio_lie_events_ronin', 2), ('nio_lie_events_ronin_3d', 2),
    ('nio_lie_events_ronin_3d', 3), ('nio_lie_events_tlio', 3),
    ('ronin_resnet18', 2), ('imunet', 2),
])
def test_registered_models_exact_results_constructor_rng_and_forward_savings(native, tmp_path, profile, metric_dims):
    weights = checkpoint(tmp_path, profile)
    reset_rng()
    with ConvolutionCounter() as old_trace:
        old, old_rng, _ = run_groups(weights, native, tmp_path / 'old', False, metric_dims)
    reset_rng()
    with ConvolutionCounter() as new_trace:
        new, new_rng, session = run_groups(weights, native, tmp_path / 'new', True, metric_dims)
    assert old_rng == new_rng
    assert session.last_reason == 'reused_complete_result', session.last_reason
    for before, after in zip(old, new):
        assert digest(before.sequences) == digest(after.sequences)
        assert digest(before.calibration) == digest(after.calibration)
        assert before.cfg == after.cfg
        assert digest(before.metrics) == digest(after.metrics)
        for a, b in zip(before.sequences, after.sequences):
            for field in ARRAYS:
                x, y = getattr(a, field), getattr(b, field)
                if x is None:
                    assert y is None
                else:
                    np.testing.assert_array_equal(x, y)
            assert digest(a.outputs) == digest(b.outputs)
            assert a.loss == b.loss
        provenance = json.loads((after.save_dir / 'eval-alias-provenance.json').read_text())
        assert provenance['reused'] == (after.split != SPLITS[0])
        assert provenance['rng_neutral'] and provenance['model_state_neutral']
        assert (after.save_dir / 'predictions/s0.npz').is_file()
        assert (after.save_dir / 'metrics.json').is_file()
    # Real source API calls, no model hooks/mocks or retained full trace.
    # Persist actual counts before asserting, including any observer failure.
    (tmp_path / 'actual-conv-api-counts.json').write_text(json.dumps({
        'observation': 'sys.setprofile real C-call builtin convolution source APIs',
        'old': old_trace.by_api, 'new': new_trace.by_api,
        'expected_ratio': 4, 'profile': profile, 'metric_dims': metric_dims}))
    # Every fresh constructor still ran; source forward ran4times vs1.
    assert old_trace.count == 4 * new_trace.count > 0
    new[1].sequences[0].vel_pred[0, 0] += 1
    np.testing.assert_array_equal(new[2].sequences[0].vel_pred, old[2].sequences[0].vel_pred)


def test_default_off_plan_and_identity_are_legacy(native, tmp_path):
    plan = {'models': ['ronin_resnet18'], 'datasets': [str(native.root)],
            'project': str(tmp_path), 'seeds': [3]}
    implicit = load_plan(plan)[0]
    explicit = load_plan({**plan, 'eval_alias_reuse': False})[0]
    assert implicit == explicit
    assert 'eval_alias_policy' not in plan_item(implicit)
    implicit.metadata = {'seeds_record_sha256': 'seed-record'}
    old = scientific_run_identity(implicit)
    implicit.eval_alias_reuse = True
    new = scientific_run_identity(implicit)
    assert {k: v for k, v in new.items() if k != 'eval_alias_policy'} == old
    assert new['eval_alias_policy'] == ALIAS_POLICY
    enabled = load_plan({**plan, 'eval_alias_reuse': True})[0]
    assert plan_item(enabled)['eval_alias_policy'] == ALIAS_POLICY
    with pytest.raises(ConfigError, match='boolean'):
        load_plan({**plan, 'eval_alias_reuse': 'false'})


def test_static_plan_policy_cannot_silently_share_completed_outputs(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    item = BenchmarkItem('rn', str(weights), str(native.root), 0,
                         root=tmp_path / 'runs', eval_alias_reuse=True)
    output = item.run_dir / 'test'
    output.mkdir(parents=True)
    (output / 'metrics.json').write_text('{}')
    with pytest.raises(RuntimeError, match='lack.*policy identity'):
        _run_item(item)
    (item.run_dir / 'eval-alias-policy.json').write_text(json.dumps(ALIAS_POLICY))
    item.eval_alias_reuse = False
    with pytest.raises(RuntimeError, match='policy changed'):
        _run_item(item)


@pytest.mark.parametrize('knob,value', [('eval_stride', 25), ('val_batch', 1),
                                      ('metric_dims', 3), ('min_speed', 0.7),
                                      ('save_predictions', False), ('max_plots', 1)])
def test_full_output_and_sampling_configuration_is_part_of_key(native, tmp_path, knob, value):
    weights = checkpoint(tmp_path)
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    first = session.prepare(predictor, cfg, 'test', 10, {})
    other = args(weights, native, **{knob: value})
    changed = session.prepare(Predictor(other), other, 'test', 10, {})
    assert first is not None and changed is not None, session.last_reason
    assert first.key != changed.key


def test_native_orientation_privileged_velocity_masks_and_order_are_bound(native, tmp_path):
    weights = checkpoint(tmp_path)
    cfg = args(weights, native)
    session = ExactAliasSession(native, SPLITS)
    first = session.prepare(Predictor(cfg), cfg, 'test', 10, {})
    assert first is not None, session.last_reason
    for field in ('orientation', 'velocity', 'valid_pose', 'valid_imu'):
        path = native.sequence_path('s0')
        sequence = load_sequence(path)
        changed = getattr(sequence, field)
        if changed.dtype == bool:
            changed[0] = not changed[0]
        elif field == 'orientation':
            changed[0] = np.array([0.99995, 0, 0, 0.01])
        else:
            changed[0, 0] += 0.1
        save_sequence(path, sequence)
        attempt = session.prepare(Predictor(cfg), cfg, 'test', 10, {})
        assert attempt is not None and attempt.key != first.key, session.last_reason
        first = attempt
    native.split_file(SPLITS[1]).write_text('s1\ns0\n')
    assert session.prepare(Predictor(cfg), cfg, SPLITS[1], 10, {}) is None


def test_checkpoint_epoch_phase_changes_key(native, tmp_path):
    weights = checkpoint(tmp_path, 'nio_lie_events_tlio')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    early = session.prepare(predictor, cfg, 'test', 0, {})
    late = session.prepare(predictor, cfg, 'test', 10, {})
    assert early and late and early.key != late.key


def test_class_output_policy_is_owned_scientific_state(native, tmp_path, monkeypatch):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    old = model_signature(predictor.model)
    monkeypatch.setattr(type(predictor.model), 'saved_outputs', ('different_output',))
    assert model_signature(predictor.model) != old


@pytest.mark.parametrize('artifact', ['metrics.json', 'sequences.csv', 'predictions/s0.npz',
                                   'eval-alias-complete.json'])
def test_tampered_or_missing_complete_artifact_falls_back(native, tmp_path, artifact):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    session = ExactAliasSession(native, SPLITS)
    cfg = args(weights, native)
    Validator(cfg, save_dir=tmp_path / 'source', alias_session=session)(dataset=native)
    assert session.entries, session.last_reason
    path = tmp_path / 'source' / artifact
    path.unlink()
    alias = args(weights, native, SPLITS[1])
    result = Validator(alias, save_dir=tmp_path / 'alias', alias_session=session)(dataset=native)
    assert result.sequences
    provenance = json.loads((result.save_dir / 'eval-alias-provenance.json').read_text())
    assert provenance['reused'] is False


def test_missing_requested_plots_does_not_commit_partial_source(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native, plots=True)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    attempt = session.prepare(predictor, cfg, 'test', 10, {})
    assert attempt, session.last_reason
    result = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    directory = tmp_path / 'partial'
    directory.mkdir()
    # This intentionally reproduces a failed plot save without fake metric files.
    result.save(directory, predictions=True, plots=False)
    (directory / 'args.yaml').write_text('')
    (directory / 'env.json').write_text('{}')
    session.finish(attempt, predictor, result, cfg, directory)
    assert not session.entries
    assert not (directory / 'eval-alias-complete.json').exists()


def test_unknown_state_hooks_callbacks_efficiency_and_cuda_are_refused(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    session = ExactAliasSession(native, SPLITS)
    predictor = Predictor(cfg)
    predictor.model.opaque = object()
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    del predictor.model.opaque
    hook = predictor.model.register_forward_hook(lambda *args: None)
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    hook.remove()
    assert session.prepare(predictor, cfg, 'test', 10, {'on_val_start': [lambda _: None]}) is None
    cfg.efficiency = True
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    cfg.efficiency = False
    predictor.device = torch.device('cuda')  # no CUDA allocation/initialization
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    assert 'separate qualification' in session.last_reason


def test_stochastic_sequence_and_custom_model_classes_have_no_certificate(native, tmp_path):
    # No forward, checkpoint or CUDA is needed to refuse these known classes.
    from inertial_benchmark.models.diffusionimu.model import DiffusionIMU
    from inertial_benchmark.nn.base import SequenceModel
    for cls in (DiffusionIMU, SequenceModel):
        with pytest.raises(Ineligible, match='no CPU purity'):
            source_certificate(object.__new__(cls))


def test_state_and_rng_mutation_block_commit_and_source_binding_changes(native, tmp_path, monkeypatch):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    attempt = session.prepare(predictor, cfg, 'test', 10, {})
    assert attempt, session.last_reason
    before = model_signature(predictor.model)
    result = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    assert model_signature(predictor.model) == before
    predictor.model.extra_counter = 1
    session.finish(attempt, predictor, result, cfg, tmp_path)
    assert not session.entries and 'not state/RNG neutral' in session.last_reason
    del predictor.model.extra_counter
    attempt = session.prepare(predictor, cfg, 'test', 10, {})
    torch.rand(1)
    session.finish(attempt, predictor, result, cfg, tmp_path)
    assert not session.entries and 'not state/RNG neutral' in session.last_reason
    from inertial_benchmark.engine import eval_aliases
    monkeypatch.setitem(eval_aliases._SOURCE_SHA256, 'engine/predictor.py', '0' * 64)
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    assert 'audited source changed' in session.last_reason


def test_session_cannot_load_prior_artifacts_and_memory_bound_falls_back(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    tiny = ExactAliasSession(native, SPLITS, max_bytes=1)
    result = Validator(args(weights, native), save_dir=tmp_path / 'source', alias_session=tiny)(dataset=native)
    assert not tiny.entries
    assert 'memory bound' in tiny.last_reason
    fresh = ExactAliasSession(native, SPLITS)
    cfg = args(weights, native, SPLITS[1])
    attempt = fresh.prepare(Predictor(cfg), cfg, SPLITS[1], 10, {})
    assert attempt, fresh.last_reason
    assert fresh.lookup(attempt, cfg, native.name, SPLITS[1]) is None
    assert result.sequences



def test_Module_type_parameter_source_structure_is_admitted():
    from inertial_benchmark.engine.eval_aliases import _class_attributes
    values = _class_attributes(torch.nn.Module)
    assert values['T_destination'] == ('torch.nn.Module.typing_only', 'T_destination',
                                      'builtins.dict[builtins.str,typing.Any]', (), False, False)
    model_signature(torch.nn.Module())


@pytest.mark.parametrize('override', [1, TypeVar('T_destination', bound=dict[str, Any])])
def test_subclass_type_parameter_override_is_not_ignored(override):
    class Shadow(torch.nn.Module):
        T_destination = override
    with pytest.raises(Ineligible, match='typing metadata was overridden'):
        model_signature(Shadow())


def test_Module_type_parameter_identity_mutation_is_refused(monkeypatch):
    monkeypatch.setattr(torch.nn.Module, 'T_destination',
                        TypeVar('T_destination', bound=dict[str, Any]))
    with pytest.raises(Ineligible, match='typing metadata was overridden'):
        model_signature(torch.nn.Module())


def test_instance_type_parameter_remains_unknown_state():
    model = torch.nn.Module()
    model.opaque = TypeVar('unreviewed_instance_state')
    with pytest.raises(Ineligible, match='unknown Python state: typing.TypeVar'):
        model_signature(model)



@pytest.mark.parametrize('kind', ['missing_key', 'missing_import_marker'])
def test_unqualified_Module_metadata_preserves_ordinary_fallback(native, tmp_path, monkeypatch, kind):
    from inertial_benchmark.engine import eval_aliases
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    if kind == 'missing_key':
        monkeypatch.delattr(torch.nn.Module, 'T_destination')
    else:
        monkeypatch.setattr(eval_aliases, '_MODULE_TYPE_PARAMETER', None)
    session = ExactAliasSession(native, SPLITS)
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    assert 'unavailable or unqualified' in session.last_reason
    # The metadata refusal cannot break the disabled normal evaluation path.
    ordinary = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    assert ordinary.evaluated and ordinary.sequences[0].vel_pred.size


class UnknownFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class UnknownDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize('kind', ['function', 'dispatch', 'pre_dispatch'])
def test_external_modes_are_refused_before_input_qualification(native, tmp_path, kind):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    if kind == 'function':
        mode = UnknownFunctionMode()
    elif kind == 'dispatch':
        mode = UnknownDispatchMode()
    else:
        from torch._subclasses.schema_check_mode import SchemaCheckMode
        mode = SchemaCheckMode()
        mode._dispatch_key = torch._C.DispatchKey.PreDispatch
    with mode:
        assert session.prepare(predictor, cfg, 'test', 10, {}) is None
        assert 'external Torch' in session.last_reason and 'mode is active' in session.last_reason
    # Ordinary state is usable after the mode exits; no test class is admitted.
    assert session.prepare(predictor, cfg, 'test', 10, {}) is not None, session.last_reason


def test_mode_entered_after_prepare_blocks_lookup_and_later_commit(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    session = ExactAliasSession(native, SPLITS)
    source_cfg = args(weights, native)
    source = Validator(source_cfg, save_dir=tmp_path / 'source', alias_session=session)(dataset=native)
    assert session.entries, session.last_reason
    cfg = args(weights, native, SPLITS[1])
    predictor = Predictor(cfg)
    attempt = session.prepare(predictor, cfg, SPLITS[1], 10, {})
    assert attempt, session.last_reason
    with UnknownFunctionMode():
        assert session.lookup(attempt, cfg, native.name, SPLITS[1]) is None
        assert 'function mode is active' in session.last_reason
    assert not session.entries and 'function mode is active' in attempt.invalid_reason
    destination = tmp_path / 'invalidated'
    destination.mkdir()
    session.finish(attempt, predictor, source, cfg, destination)
    assert not session.entries and not (destination / 'eval-alias-complete.json').exists()
    provenance = json.loads((destination / 'eval-alias-provenance.json').read_text())
    assert not provenance['eligible'] and not provenance['reused']


def test_mode_entered_during_evaluation_cannot_commit_complete_source(native, tmp_path):
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    session = ExactAliasSession(native, SPLITS)
    attempt = session.prepare(predictor, cfg, 'test', 10, {})
    assert attempt, session.last_reason
    result = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    destination = tmp_path / 'mode-commit'
    destination.mkdir()
    with UnknownDispatchMode():
        session.finish(attempt, predictor, result, cfg, destination)
        assert 'dispatch mode is active' in session.last_reason
    assert not session.entries and not (destination / 'eval-alias-complete.json').exists()


@pytest.mark.parametrize('missing', ['inspection_API', 'pre_dispatch_source'])
def test_unavailable_mode_inspection_preserves_default_off_evaluation(native, tmp_path, monkeypatch, missing):
    from inertial_benchmark.engine import eval_aliases
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    if missing == 'inspection_API':
        monkeypatch.setitem(eval_aliases._MODE_C_APIS, '_len_torch_function_stack', None)
    else:
        monkeypatch.setattr(eval_aliases, '_PRE_DISPATCH_SOURCE_SHA256', '0' * 64)
    session = ExactAliasSession(native, SPLITS)
    assert session.prepare(predictor, cfg, 'test', 10, {}) is None
    assert 'unqualified' in session.last_reason
    ordinary = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    assert ordinary.evaluated and ordinary.sequences[0].vel_pred.size


def test_passive_observer_does_not_replace_an_external_python_profiler():
    callback = lambda frame, event, arg: None
    assert sys.getprofile() is None
    sys.setprofile(callback)
    try:
        with pytest.raises(RuntimeError, match='external Python profiler'):
            with ConvolutionCounter():
                pass
        assert sys.getprofile() is callback
    finally:
        sys.setprofile(None)



def test_wrong_true_builtin_binding_is_not_a_mode_inspection_certificate(native, tmp_path, monkeypatch):
    from inertial_benchmark.engine import eval_aliases
    weights = checkpoint(tmp_path, 'ronin_resnet18')
    cfg = args(weights, native)
    predictor = Predictor(cfg)
    wrong = torch._C._len_torch_dispatch_stack
    assert type(wrong) is types.BuiltinFunctionType
    # Reproduce substitution before helper capture, not just a changed pointer.
    monkeypatch.setattr(torch._C, '_len_torch_function_stack', wrong)
    monkeypatch.setitem(eval_aliases._MODE_C_APIS, '_len_torch_function_stack', wrong)
    session = ExactAliasSession(native, SPLITS)
    with UnknownFunctionMode():
        assert session.prepare(predictor, cfg, 'test', 10, {}) is None
        assert 'inspection API unavailable or unqualified' in session.last_reason
    ordinary = Validator(cfg).evaluate(predictor, native.sequence_paths('test'), 10, native.name, 'test')
    assert ordinary.evaluated and ordinary.sequences[0].vel_pred.size
