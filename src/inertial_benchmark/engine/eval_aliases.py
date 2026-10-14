"""Default-off, CPU-only sharing of exact final-test aliases within one run.

This is deliberately not a persistent cache. A source must finish ordinary
evaluation and save all requested artifacts before its complete result can be
shared. Every alias still constructs and loads its own Predictor. No RNG state
is restored: constructor consumption remains part of the ordinary execution.
CUDA, stateful/unknown models, callbacks and efficiency measurements fall back.
"""

from __future__ import annotations

import copy
import ast
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from ..data.format import load_sequence
from ..data.views import SequenceView
from .results import RunResult

ALIAS_POLICY = {"schema": "ib.eval-alias.v2", "scope": "single_run_final_exact_alias",
                "qualification": "source_bound_empty_modes_cpu_only", "persistent": False}
_MODELS = {
    "inertial_benchmark.models.nio_lie_events.model.NIOLieEventsRoNIN",
    "inertial_benchmark.models.nio_lie_events.model.NIOLieEventsTLIO",
    "inertial_benchmark.models.ronin.model.RoNINResNet",
    "inertial_benchmark.models.tlio.model.TLIOResNet",
    "inertial_benchmark.models.imunet.model.IMUNet",
}
# Audited pure ordinary-eval implementations. A changed dependency requires a
# new source review and CPU qualification; it is never implicitly admitted.
_SOURCE_SHA256 = {
    'models/nio_lie_events/model.py': '63fa6577561a1f01618a640ebeac42b888554ca90f80ebaddb25413c4c47e782',
    'models/ronin/model.py': 'ccea19aeb4cbb9032427e959ea06d269a54ca39e55e077bfc83f9b1927fb1902',
    'models/tlio/model.py': '77ff87726b4257134d4498c51f523f0e5260c5874e4568e87bf10de6b9de35a2',
    'models/imunet/model.py': 'fbe26461f3a0f0e27cca8b110fdb640808c248d962c9ce08640f486fd50ef422',
    'nn/modules/lie_events.py': '96b171ef8521385d783026cc98f1e1742f8af3f40bb716a44adc73b0fd8e5051',
    'nn/modules/resnet1d.py': 'a884eb0eab381718628e68890762eed4f7a0c8b819a38105384062c43f288654',
    'nn/heads.py': '053a3a2490a36372b8e1a10d7c37e7874343c9c86490d2495987ff8021b7c476',
    'nn/base.py': 'b0f136b58e5fc476dd8e5ada9ff918ca79d493d6b10ba0702ac2aeaa5787789d',
    'nn/losses.py': '8456ee7dec8122005d51263e194647e07512c804a9328a6105dc0b7d76b43c2e',
    'nn/registry.py': 'f791644f0f14d313b14d605746318c7e04a69e5663351073bd0ce5cf8922a5f7',
    'engine/predictor.py': 'e7d2a1401dc0ba0011de7b3b032e8ab5f367cea2bad989979df3ed125e8764b3',
    'engine/results.py': 'c489f31fdd3f7b53537c80a618c5d92040f05b84271dacdb024d6b4fdcac19fc',
    'data/views.py': '70747c49e5b87b3f92099ae7d589f4e416701e5cfbeff611550afca584e83aac',
    'data/format.py': '25da7b232ea690a626e7d58bc0a97aa9f1bef5614819703fcb776a1721177334',
    'data/resample.py': '335696c3b78bfc1a5bb6efc38c3397ac3f7c7fdcb69ab4f74f72d0d80d4916b7',
    'utils/geometry.py': 'f60b42eddc2227ddabaa840ab03e7d76fe319bbadd9efb5c52aa0bb00d6eec81',
    'utils/torch_utils.py': '9c5f6ec9715c926324c96bbef317f64a8b13b04976a7b51ca1ff42da14aade6f',
    'metrics/__init__.py': '2991cf14d37f85f2430800350ab6e7e97b5169742f4588565b4193f203d40f2c',
    'metrics/trajectory.py': 'f6fbe18d4059817b6cae276b7c6c470b00db4a29464bb905deefb2b5c6976c4e',
    'metrics/uncertainty.py': '2ab9aa85412778becd1b738519206a89c12bb1be8132d243b325b7c201cebc76',
    'metrics/community.py': 'b8ef444a529ccda7a1f8309d4549a4665b3105c118a6ea68e6dd142c9a344ccb',
    'metrics/velocity.py': 'e10318234a63cbc10bbf0db957cb9dce72ffb7c61a2221c73cb661f4940fbe6d',
}
_PRESENTATION = {"split", "project", "name", "exist_ok", "verbose"}
# Exact, source-bound torch typing metadata, not executable model state.
_MODULE_TYPE_PARAMETER = vars(torch.nn.Module).get("T_destination")
_MODULE_TYPING_SOURCE_SHA256 = "56a1fa39f332238e3bad87f431903a875965abef92547afee6356c1390f2b713"
# Capture only references: unsupported Torch stays usable on the default path.
_MODE_C_APIS = {name: getattr(torch._C, name, None) for name in
               ('_len_torch_function_stack', '_len_torch_dispatch_stack', '_get_dispatch_mode')}
_PRE_DISPATCH_SOURCE_SHA256 = "c3d561d13aa4dede4525a5df8b47e38a54135653d1907101211c843df467a519"



class Ineligible(ValueError):
    """Conservative refusal; ordinary evaluation remains available."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _feed(h, value: Any) -> None:
    """Typed, bit-exact scientific values; opaque objects are never ignored."""
    def put(data):
        h.update(str(len(data)).encode() + b':' + data)
    if value is None or value is Ellipsis:
        put(str(value).encode())
    elif isinstance(value, (str, bytes, bool, int, float, Path, torch.dtype, torch.device)):
        put((type(value).__name__ + ':' + (value.hex() if type(value) is float
                                         else str(value))).encode())
    elif isinstance(value, np.generic):
        _feed(h, np.asarray(value))
    elif isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.layout != torch.strided:
            raise Ineligible('only ordinary CPU tensors are qualified')
        _feed(h, (str(value.dtype), tuple(value.shape), value.requires_grad))
        put(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise Ineligible('object arrays are not scientific values')
        _feed(h, (str(value.dtype), tuple(value.shape)))
        put(np.ascontiguousarray(value).tobytes())
    elif is_dataclass(value) and not isinstance(value, type):
        _feed(h, (type(value).__module__, type(value).__qualname__, len(fields(value))))
        for field in fields(value):
            _feed(h, (field.name, getattr(value, field.name)))
    elif isinstance(value, dict):
        put(('dict:' + str(len(value))).encode())
        if not all(isinstance(k, str) for k in value):
            raise Ineligible('non-string mapping keys')
        for key in sorted(value):
            _feed(h, (key, value[key]))
    elif isinstance(value, (list, tuple, set, frozenset)):
        put((type(value).__name__ + ':' + str(len(value))).encode())
        items = sorted(value, key=digest) if isinstance(value, (set, frozenset)) else value
        for item in items:
            _feed(h, item)
    else:
        raise Ineligible(f'unknown Python state: {type(value).__module__}.{type(value).__qualname__}')


def digest(value: Any) -> str:
    h = hashlib.sha256()
    _feed(h, value)
    return h.hexdigest()


def rng_signature() -> str:
    """Inspect CPU global generators without initializing or touching CUDA."""
    return digest((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def _code_signature(code) -> str:
    constants = tuple(_code_signature(v) if isinstance(v, types.CodeType) else v
                      for v in code.co_consts)
    return digest((code.co_code, constants, code.co_names, code.co_varnames,
                   code.co_freevars, code.co_cellvars, code.co_argcount,
                   code.co_kwonlyargcount, code.co_posonlyargcount, code.co_flags))


def _check_method(fn, compiled: dict) -> None:
    if not inspect.isfunction(fn):
        return
    if not hasattr(fn.__code__, 'co_qualname'):
        raise Ineligible('runtime code qualification needs Python 3.11 or later')
    filename = Path(fn.__code__.co_filename)
    if not filename.is_file():
        raise Ineligible('generated or replaced model method')
    key = str(filename.resolve())
    if key not in compiled:
        codes = {}
        def walk(code):
            codes[code.co_qualname] = _code_signature(code)
            for constant in code.co_consts:
                if isinstance(constant, types.CodeType):
                    walk(constant)
        walk(compile(filename.read_bytes(), str(filename), 'exec', dont_inherit=True))
        compiled[key] = codes
    if compiled[key].get(fn.__code__.co_qualname) != _code_signature(fn.__code__):
        raise Ineligible('runtime method differs from its source')


def _empty_torch_modes() -> dict:
    """Reject every external mode; no observer or test class gets permission."""
    try:
        for name, api in _MODE_C_APIS.items():
            if (type(api) is not types.BuiltinFunctionType
                    or api.__name__ != name or api.__module__ != 'torch._C'
                    or getattr(torch._C, name, None) is not api):
                raise Ineligible('Torch mode inspection API unavailable or unqualified')
        if _MODE_C_APIS['_len_torch_function_stack']() != 0:
            raise Ineligible('external Torch function mode is active')
        if _MODE_C_APIS['_len_torch_dispatch_stack']() != 0:
            raise Ineligible('external Torch dispatch mode is active')
        keys = getattr(torch._C, '_TorchDispatchModeKey', None)
        members = getattr(keys, '__members__', None)
        if type(members) is not dict or not members:
            raise Ineligible('Torch infrastructure mode inspection unavailable')
        for key in members.values():
            if _MODE_C_APIS['_get_dispatch_mode'](key) is not None:
                raise Ineligible('external Torch infrastructure mode is active')
        # PreDispatch has separate infrastructure/schema slots which the C user
        # stack does not cover. Inspect source-bound state without invoking modes.
        ops = torch._ops
        if file_sha256(Path(ops.__file__)) != _PRE_DISPATCH_SOURCE_SHA256:
            raise Ineligible('Torch pre-dispatch inspection source is unqualified')
        cls = getattr(ops, '_ModeStackStateForPreDispatch', None)
        state = getattr(ops, '_mode_stack_state_for_pre_dispatch', None)
        if (type(cls) is not type or type(state) is not cls
                or cls.__module__ != ops.__name__
                or cls.__qualname__ != '_ModeStackStateForPreDispatch'):
            raise Ineligible('Torch pre-dispatch state is unqualified')
        compiled = {}
        for method in vars(cls).values():
            if inspect.isfunction(method):
                _check_method(method, compiled)
        values = vars(state)
        slot_key = '_ModeStackStateForPreDispatch__infra_modes'
        if set(values) != {slot_key, '_schema_check_mode'}:
            raise Ineligible('Torch pre-dispatch state structure is unqualified')
        slots = values[slot_key]
        if type(slots) is not list or len(slots) != 2:
            raise Ineligible('Torch pre-dispatch slots are unqualified')
        if any(value is not None for value in slots) or values['_schema_check_mode'] is not None:
            raise Ineligible('external Torch pre-dispatch mode is active')
        return {'function': 0, 'dispatch': 0, 'pre_dispatch': 0,
                'infrastructure_keys': sorted(members),
                'pre_dispatch_source_sha256': _PRE_DISPATCH_SOURCE_SHA256}
    except Ineligible:
        raise
    except (AttributeError, OSError, ValueError, TypeError, RuntimeError) as exc:
        raise Ineligible('Torch mode inspection unavailable or unqualified') from exc


def source_certificate(model) -> dict:
    """Explicit CPU candidate list plus source and runtime method binding."""
    name = type(model).__module__ + '.' + type(model).__name__
    if name not in _MODELS:
        raise Ineligible('model class has no CPU purity qualification')
    mode_certificate = _empty_torch_modes()
    if (_MODULE_TYPE_PARAMETER is None
            or vars(torch.nn.Module).get('T_destination') is not _MODULE_TYPE_PARAMETER):
        raise Ineligible('Torch Module typing metadata is unavailable or unqualified')
    _class_attributes(torch.nn.Module)  # opt-in only; binds actual typing/source structure
    root = Path(__file__).resolve().parents[1]
    bindings = {}
    compiled = {}
    checked = set()
    constants = {}
    for relative, expected in _SOURCE_SHA256.items():
        actual = file_sha256(root / relative)
        if actual != expected:
            raise Ineligible(f'audited source changed: {relative}')
        bindings[relative] = actual
    # Validate module-level scientific helpers as well as model methods. An
    # instance or global monkeypatch must not acquire a certificate merely by
    # keeping the original file on disk.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        try:
            relative = str(Path(filename).resolve().relative_to(root))
        except ValueError:
            continue
        if relative not in bindings:
            continue
        text = Path(filename).read_bytes()
        top = compile(text, filename, 'exec', dont_inherit=True)
        for code in top.co_consts:
            if not isinstance(code, types.CodeType) or code.co_name.startswith('<'):
                continue
            actual = vars(module).get(code.co_name)
            if inspect.isfunction(actual):
                if _code_signature(inspect.unwrap(actual).__code__) != _code_signature(code):
                    raise Ineligible('scientific helper differs from its source')
                checked.add(id(actual))
            elif not isinstance(actual, type):
                raise Ineligible('scientific helper was replaced')
            else:
                for method_code in code.co_consts:
                    if not isinstance(method_code, types.CodeType) or method_code.co_name.startswith('<'):
                        continue
                    member = vars(actual).get(method_code.co_name)
                    if isinstance(member, (classmethod, staticmethod)):
                        member = member.__func__
                    if isinstance(member, property):
                        member = member.fget
                    if not inspect.isfunction(member) or _code_signature(inspect.unwrap(member).__code__) != _code_signature(method_code):
                        raise Ineligible('scientific class method was replaced')
                    checked.add(id(member))
        for node in ast.parse(text).body:
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, ast.AnnAssign) else [])
            for target_node in targets:
                if not isinstance(target_node, ast.Name):
                    continue
                value = getattr(module, target_node.id, None)
                try:
                    constants[relative + ':' + target_node.id] = digest(value)
                except Ineligible:
                    # Registries/types are checked by method and selected-loss
                    # binding, not serialized as opaque Python objects.
                    continue
                try:
                    literal = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                # Registries such as METRIC_INFO are deliberately extended by
                # their own module after declaration. Bind their actual content;
                # only immutable literal constants require declaration equality.
                if not isinstance(literal, (dict, list, set)) and digest(literal) != digest(value):
                    raise Ineligible('scientific literal constant changed')
            if not isinstance(node, ast.ImportFrom):
                continue
            target = importlib.util.resolve_name('.' * node.level + (node.module or ''),
                                                 module.__package__) if node.level else node.module
            if not target or not target.startswith('inertial_benchmark.'):
                continue
            imported = sys.modules.get(target)
            for alias in node.names:
                if alias.name != '*' and getattr(module, alias.asname or alias.name, None) is not getattr(imported, alias.name, None):
                    raise Ineligible('imported scientific dependency was replaced')
    from ..nn import losses
    loss_cls = losses.LOSSES.get(model.loss_name)
    if getattr(loss_cls, '__module__', None) != losses.__name__:
        raise Ineligible('custom loss has no purity qualification')
    for member in vars(loss_cls).values():
        _check_method(member, compiled)
    for module in model.modules():
        cls = type(module)
        if not cls.__module__.startswith(('inertial_benchmark.', 'torch.nn.')):
            raise Ineligible('unknown child module')
        for base in cls.__mro__:
            if base is object:
                continue
            for attribute, member in vars(base).items():
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if isinstance(member, property):
                    member = member.fget
                if inspect.isfunction(member) and base.__module__.startswith('inertial_benchmark.'):
                    fn = inspect.unwrap(member)
                    if fn.__module__ != base.__module__ or fn.__qualname__ != base.__qualname__ + '.' + attribute:
                        raise Ineligible('model method belongs to a different implementation')
                if id(member) not in checked:
                    _check_method(member, compiled)
                    checked.add(id(member))
    # Hook dictionaries outside the model can also affect every forward.
    from torch.nn.modules import module as nn_module
    if any(value for key, value in vars(nn_module).items()
           if key.startswith('_global_') and 'hook' in key and isinstance(value, dict)):
        raise Ineligible('global module hooks are active')
    return {'model_class': name, 'device_qualification': 'cpu',
            'torch_mode_certificate': mode_certificate,
            'loss_class': loss_cls.__qualname__,
            'source_sha256': bindings, 'torch_version': str(torch.__version__),
            'scientific_constants_sha256': digest(constants),
            'runtime_code_sha256': digest(compiled),
            'execution_path_sha256': {name: file_sha256(Path(__file__).parent / name)
                                     for name in ('eval_aliases.py', 'validator.py')}}


def _class_attributes(cls) -> dict:
    values = {}
    for key, value in vars(cls).items():
        if key == 'T_destination':
            # This exact Module key is only the state_dict destination type
            # parameter (Torch source line2173). Do not extend this exception to
            # model instance attributes, subclasses or unrelated TypeVars.
            if cls is not torch.nn.Module or value is not _MODULE_TYPE_PARAMETER:
                raise Ineligible('Torch Module typing metadata was overridden')
            if file_sha256(Path(inspect.getfile(cls))) != _MODULE_TYPING_SOURCE_SHA256:
                raise Ineligible('Torch Module typing declaration source changed')
            if type(value) is not TypeVar:
                raise Ineligible('Torch Module typing metadata is not the declared TypeVar')
            bound = value.__bound__
            if (value.__name__ != 'T_destination'
                    or type(bound) is not types.GenericAlias or bound.__origin__ is not dict
                    or bound.__args__ != (str, Any) or value.__constraints__ != ()
                    or value.__covariant__ or value.__contravariant__
                    or getattr(value, '__infer_variance__', False)):
                raise Ineligible('Unexpected Torch Module TypeVar structure')
            # No evaluation of forward refs, get_type_hints, callbacks or code.
            values[key] = ('torch.nn.Module.typing_only', 'T_destination',
                           'builtins.dict[builtins.str,typing.Any]', (), False, False)
        elif ((not key.startswith('__') or key == '__constants__')
              and not (callable(value) or isinstance(value, (property, classmethod, staticmethod))
                       or inspect.isdatadescriptor(value))):
            values[key] = value
    return values


def model_signature(model) -> str:
    state = []
    for name, module in model.named_modules():
        attrs = {}
        for key, value in vars(module).items():
            if key in ('_parameters', '_buffers', '_modules'):
                continue
            if 'hook' in key and value:
                raise Ineligible('model hooks are active')
            attrs[key] = value
        class_attrs = []
        for cls in type(module).__mro__:
            values = _class_attributes(cls)
            class_attrs.append((cls.__module__, cls.__qualname__, values))
        state.append((name, type(module).__module__, type(module).__qualname__, attrs,
                      class_attrs,
                      dict(module._parameters), dict(module._buffers),
                      {k: p.grad for k, p in module._parameters.items() if p is not None}))
    return digest(state)


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _artifacts(result: RunResult, args, directory: Path) -> dict:
    paths = ['metrics.json', 'sequences.csv', 'args.yaml', 'env.json']
    if args.save_predictions:
        paths.extend(f'predictions/{s.sequence_id}.npz' for s in result.evaluated)
    if args.plots and result.evaluated:
        paths.extend(f'plots/traj_{s.sequence_id}.png'
                     for s in sorted(result.evaluated, key=lambda s: s.sequence_id)[:int(args.max_plots)])
        paths.extend('plots/' + n + '.png' for n in
                     ('trajectories', 'error_cdf', 'error_over_time', 'ate_box', 'length_ratio'))
    return {p: file_sha256(directory / p) for p in paths}


@dataclass
class _Attempt:
    key: str
    before_state: str
    before_rng: str
    certificate: dict
    ids: list
    file_stats: list
    source: Any = None
    invalid_reason: str = ""


class ExactAliasSession:
    """Owned by one benchmark item; entries live only until that item returns."""

    def __init__(self, spec, splits, max_bytes: int = 256 << 20):
        self.spec = spec
        groups = {}
        for split in splits:
            if str(split).startswith('test') and spec.has_split(split):
                ids = tuple(spec.sequence_ids(split))
                groups[ids] = groups.get(ids, 0) + 1
        self.duplicates = {ids for ids, count in groups.items() if count > 1 and ids}
        self.max_bytes = int(max_bytes)
        self.entries = {}
        self.last_reason = 'no_matching_complete_source'

    def prepare(self, predictor, args, split: str, epoch: int, callbacks: dict):
        """Fail closed; key includes actual native arrays, windows and extras."""
        try:
            if not str(split).startswith('test') or tuple(self.spec.sequence_ids(split)) not in self.duplicates:
                raise Ineligible('not a final-test ordered alias group')
            if predictor.device.type != 'cpu':
                raise Ineligible('CUDA alias reuse has no separate qualification')
            if args.efficiency or any(callbacks.values()) or any(predictor.callbacks.values()):
                raise Ineligible('callbacks or efficiency require ordinary execution')
            certificate = source_certificate(predictor.model)
            before_state, before_rng = model_signature(predictor.model), rng_signature()
            ids = self.spec.sequence_ids(split)
            stats, native = [], []
            for sid, path in zip(ids, self.spec.sequence_paths(split)):
                path = path.resolve()
                stat = path.stat()
                token = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                view = SequenceView(load_sequence(path), predictor.view_cfg)
                h = hashlib.sha256()
                _feed(h, (view.seq, view.cfg, view.yaw_offset, view.imu, view.q,
                          view.yaw, view.valid_input, view.valid_target, view.valid))
                starts = view.starts(int(args.eval_stride), require_valid=False)
                _feed(h, (starts, view.window_valid_input(starts), view.window_valid_target(starts)))
                for first in range(0, len(starts), 256):
                    _feed(h, view.windows(starts[first:first + 256]))
                native.append((sid, token, file_sha256(path), h.hexdigest()))
                if token != (str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns):
                    raise Ineligible('native input changed during verification')
                stats.append(token)
            cfg = {k: v for k, v in args.to_dict().items() if k not in _PRESENTATION}
            key = digest((ALIAS_POLICY, certificate, before_state, cfg, epoch,
                          str(Path(args.model).resolve()), file_sha256(Path(args.model)),
                          self.spec.fingerprint(), file_sha256(self.spec.root / 'dataset.json'),
                          ids, native, str(predictor.device), predictor.amp,
                          torch.get_num_threads(), torch.get_num_interop_threads(),
                          torch.are_deterministic_algorithms_enabled(),
                          torch.is_deterministic_algorithms_warn_only_enabled(),
                          torch.backends.mkldnn.enabled, torch.backends.mkldnn.deterministic,
                          torch.get_float32_matmul_precision(), torch.get_default_dtype()))
            if before_rng != rng_signature() or before_state != model_signature(predictor.model):
                raise Ineligible('input qualification changed RNG or model state')
            self.last_reason = 'no_matching_complete_source'
            return _Attempt(key, before_state, before_rng, certificate, ids, stats)
        except (Ineligible, OSError, ValueError, TypeError, RuntimeError) as exc:
            self.last_reason = str(exc)
            return None

    def lookup(self, attempt: _Attempt, args, dataset: str, split: str):
        entry = self.entries.get(attempt.key)
        if entry is None:
            return None
        try:
            _empty_torch_modes()
            manifest = entry['manifest']
            if file_sha256(entry['path']) != entry['manifest_sha256']:
                raise Ineligible('complete source manifest changed')
            for relative, expected in manifest['artifacts'].items():
                if file_sha256(entry['path'].parent / relative) != expected:
                    raise Ineligible('complete source artifact changed')
            if digest((entry['sequences'], entry['calibration'])) != manifest['numerical_result_sha256']:
                raise Ineligible('in-memory result changed')
            sequences = copy.deepcopy(entry['sequences'])
            attempt.source = entry
            return RunResult(sequences, dataset=dataset, split=split,
                             model=entry['model'], cfg=args.to_dict(),
                             calibration=copy.deepcopy(entry['calibration']))
        except (Ineligible, OSError, MemoryError) as exc:
            attempt.invalid_reason = str(exc)
            self.entries.pop(attempt.key, None)
            self.last_reason = str(exc)
            return None

    def finish(self, attempt, predictor, result, args, directory: Path) -> None:
        """Commit only after normal save; missing plots and changed state reject."""
        if attempt is None:
            _atomic_json(directory / 'eval-alias-provenance.json', {
                'schema': 'ib.eval-alias.fallback.v1', 'policy': ALIAS_POLICY,
                'split': result.split, 'reused': False, 'eligible': False,
                'reason': self.last_reason})
            return
        try:
            if attempt.invalid_reason:
                raise Ineligible(attempt.invalid_reason)
            _empty_torch_modes()
            if model_signature(predictor.model) != attempt.before_state or rng_signature() != attempt.before_rng:
                raise Ineligible('evaluation is not state/RNG neutral')
            if [s.sequence_id for s in result.sequences] != attempt.ids or any(s.skipped for s in result.sequences):
                raise Ineligible('incomplete ordered sequence result')
            for path, size, modified, changed in attempt.file_stats:
                stat = Path(path).stat()
                if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (size, modified, changed):
                    raise Ineligible('native input changed during evaluation')
            numerical = digest((result.sequences, result.calibration))
            artifacts = _artifacts(result, args, directory)
            source = attempt.source
            size = sum(a.nbytes for s in result.sequences for a in vars(s).values()
                       if isinstance(a, np.ndarray))
            size += sum(a.nbytes for s in result.sequences for a in s.outputs.values()
                        if isinstance(a, np.ndarray))
            if size > self.max_bytes:
                raise Ineligible('full result exceeds session memory bound')
            manifest = {'schema': 'ib.eval-alias.complete.v1', 'policy': ALIAS_POLICY,
                        'key': attempt.key, 'split': result.split, 'ordered_ids': attempt.ids,
                        'numerical_result_sha256': numerical, 'artifacts': artifacts,
                        'certificate': attempt.certificate, 'rng_neutral': True,
                        'model_state_neutral': True, 'reused': source is not None}
            if source is not None:
                manifest['source'] = {'split': source['manifest']['split'],
                                      'complete_manifest': str(source['path']),
                                      'complete_manifest_sha256': source['manifest_sha256']}
            _atomic_json(directory / 'eval-alias-provenance.json', manifest)
            complete = directory / 'eval-alias-complete.json'
            _atomic_json(complete, manifest)  # last, after all requested artifacts
            if source is None:
                # A bounded full numerical result, never a metrics-only cache.
                self.entries.clear()
                self.entries[attempt.key] = {'sequences': copy.deepcopy(result.sequences),
                                            'model': result.model, 'calibration': copy.deepcopy(result.calibration),
                                            'manifest': manifest, 'path': complete,
                                            'manifest_sha256': file_sha256(complete)}
            self.last_reason = 'reused_complete_result' if source else 'complete_source_committed'
        except (Ineligible, OSError, ValueError, TypeError, MemoryError, RuntimeError) as exc:
            self.entries.pop(attempt.key, None)
            self.last_reason = str(exc)
            _atomic_json(directory / 'eval-alias-provenance.json', {
                'schema': 'ib.eval-alias.fallback.v1', 'policy': ALIAS_POLICY,
                'split': result.split, 'reused': attempt.source is not None,
                'eligible': False, 'reason': self.last_reason})
