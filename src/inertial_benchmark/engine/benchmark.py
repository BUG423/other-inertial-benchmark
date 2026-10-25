"""基准矩阵：模型 × 数据集 × 种子，逐项训练并在测试划分上评测，已完成项自动跳过。

YAML 示例::

    name: main
    project: runs/benchmark          # 输出到 <project>/<name>/<label>/<dataset>/seed<k>/
    models:
      - ronin_resnet18                 # 名称 / 模型 YAML / checkpoint
      - {label: ronin_official, model: ronin_resnet18, overrides: {recipe: official}}
    datasets: [ronin, ridi]            # 或 {data: ..., overrides: {...}}
    seeds: [0, 1, 2]
    splits: [val]                      # 缺省使用数据集 YAML 的 test_splits；[val] = 只评 val
    overrides: {train_windows_budget: 3.2e+7, device: 0}
    report: true                       # 结束后汇总到 <project>/<name>/report
    continue_on_error: false           # true 时单项失败只记录并继续

完成标志：``train/metrics.json``（训练完成）与 ``<split>/metrics.json``（该划分评测完成）。
训练中断（有 ``last.pt`` 无 ``metrics.json``）时自动断点续训。

``overrides.train_windows_budget``（见 ``budget.py``）让每个 run 固定“看过的训练窗口总数”，
epoch 数由框架按数据集规模换算——窗口数差一个数量级的数据集之间这才是等算力预算。
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import stat
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from ..cfg import INPUT_KEYS, ConfigError, get_cfg, is_checkpoint
from ..data.manifest import resolve_dataset
from ..metrics import MAIN_METRICS
from ..utils import CFG_DIR, LOGGER, json_load, json_save, to_builtin, yaml_load

PathLike = Union[str, Path]
PRIVATE_REMOTE_ROOT = os.environ.get('IPB_V1_REMOTE_ROOT', '<agent-project-root>').rstrip('/') + '/'
PRIVATE_LOCK_ROOT = Path(os.environ.get('IPB_V1_LOCK_ROOT', '<agent-runtime-root>/campaigns'))
# 基准 YAML 允许的顶层键
PLAN_KEYS = ("name", "project", "models", "datasets", "seeds", "splits", "overrides", "report",
             "continue_on_error", "native_registry", "seed_record", "policy", "admission",
             "eval_alias_reuse")
MATRIX_STATE_MUTABLE_FIELDS = frozenset((
    'id', 'state', 'error', 'dependency_reason', 'resource_reason',
    'model_cpu_pending_reason', 'checkpoint', 'checkpoint_sha256',
    'checkpoint_retention', 'checkpoint_source', 'cross_domain_transfer',
    'result', 'result_artifacts', 'run_identity_path', 'run_identity_sha256',
))
# 评测时不需要的训练专用键
TRAIN_ONLY = ("epochs", "train_windows_budget", "budget_max_epochs",
              "batch", "lr", "optimizer", "momentum", "weight_decay", "scheduler",
              "lr_final", "step_size", "gamma", "plateau_patience", "warmup_epochs", "grad_clip",
              "patience", "min_epochs", "gap_windows", "val_interval", "fitness", "fitness_stat", "save_period", "augment",
              "stride",
              "resume", "pretrained", "recipe", "loss", "loss_switch_epoch", "model_args")


@dataclass
class BenchmarkItem:
    label: str
    model: Any
    data: str
    seed: int
    overrides: dict = field(default_factory=dict)
    splits: Optional[list] = None
    root: Path = Path(".")
    metadata: dict = field(default_factory=dict)
    eval_alias_reuse: bool = False
    # Exact qualification evidence from ``taskview_receipt.splits``.  The ops
    # runner clears ``metadata`` after independently checking admission, so the
    # split receipt travels separately into the shared execution path.
    split_receipts: Optional[dict] = None
    # Ephemeral, already-snapshotted completion evidence.  These are deliberately
    # not stored in metadata/run identity or serialized to campaign artifacts.
    completed_verification: Optional[dict] = field(default=None, repr=False)
    transfer_origin: Optional[dict] = field(default=None, repr=False)
    transfer_origin_verification: Optional[dict] = field(default=None, repr=False)
    coverage_rows: Optional[list] = field(default=None, repr=False)

    @property
    def run_dir(self) -> Path:
        return self.root / self.label / Path(str(self.data)).name / f"seed{self.seed}"


def _entry(item: Any, key: str) -> tuple:
    if isinstance(item, Mapping):
        value = item.get(key)
        if value is None:
            raise ConfigError(f"benchmark entry {item} lacks '{key}'")
        label = item.get("label") or Path(str(value)).stem
        return label, value, dict(item.get("overrides") or {})
    return Path(str(item)).stem, item, {}


def resolve_benchmark_cfg(cfg: Union[PathLike, Mapping]) -> Union[Path, Mapping]:
    """YAML 路径；不存在时依次尝试 ``cfg/benchmarks/<name>.yaml``（``cfg=main`` 的简写）。"""
    if isinstance(cfg, Mapping):
        return cfg
    path = Path(cfg)
    if path.exists():
        return path
    candidate = CFG_DIR / "benchmarks" / f"{path.stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"benchmark config {cfg} not found (also looked for {candidate})")


def _sha(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_row_id(profile: str, dataset: str, seed: int, input_track: str) -> str:
    value = dict(profile=profile, dataset=dataset, seed=seed, input_track=input_track)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def matrix_state(admission: PathLike, row: dict) -> dict:
    """One per-row state store shared by the public CLI and the ops queue."""
    identifier = matrix_row_id(row['profile'], row['dataset'], row['seed'], row['input_track'])
    if row.get('id', identifier) != identifier:
        raise RuntimeError('coverage row id does not match profile/dataset/seed/track')
    path = Path(admission).parent / 'run-states' / f'{identifier}.json'
    if not path.exists():
        return dict(row)
    state = json_load(path)
    # The coverage/taskview record is the qualification authority.  A stale or
    # hand-edited run-state may contribute execution progress, but it must not
    # replace scientific qualification fields when the two stores are merged.
    keys = ('profile', 'architecture', 'dataset', 'data_root', 'domain', 'seed', 'input_track',
            'classic', 'dims', 'main_xyz', 'native_xyz', 'learned_cuda_training', 'trainable',
            'checkpoint_training_source', 'scientific_input_spec', 'sampling_spec', 'scientific_loss',
            'dataset_fingerprint', 'dataset_manifest_sha256', 'split_sha256',
            'profile_config_sha256', 'seeds_record_sha256', 'taskview_receipt', 'model_cpu_accepted',
            'model_cpu_receipt_sha256')
    changed_overlap = any(
        key not in MATRIX_STATE_MUTABLE_FIELDS and state[key] != row[key]
        for key in set(state) & set(row)
    )
    if state.get('id', identifier) != identifier or changed_overlap or any(
            state.get(key) != row.get(key) for key in keys):
        raise RuntimeError('saved matrix state scientific row identity changed')
    merged = {**row, **state}
    merged.update({key: row[key] for key in keys if key in row})
    merged['id'] = identifier
    return merged


def save_matrix_state(admission: PathLike, row: dict) -> None:
    identifier = matrix_row_id(row['profile'], row['dataset'], row['seed'], row['input_track'])
    path = Path(admission).parent / 'run-states' / f'{identifier}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    json_save(temporary, {**row, 'id': identifier})
    os.replace(temporary, path)


def retain_completed_checkpoint(admission: PathLike, row: dict, run_dir: PathLike,
                                policy: Mapping, *, coverage_rows: Optional[list] = None,
                                transfer_origin: Optional[Mapping] = None,
                                transfer_origin_context: Optional[Mapping] = None) -> None:
    """Persist verified completion before releasing this cell's resumable last state."""
    verify_completed_state(
        row, coverage_rows, transfer_origin, transfer_origin_context)
    if row.get('state') != 'completed':
        raise RuntimeError('checkpoint retention requires completed training and every test split')
    save_matrix_state(admission, row)
    if policy.get('checkpoint_retention') != 'completed_best_only' or not row.get('trainable'):
        return
    last = Path(run_dir) / 'train' / 'weights' / 'last.pt'
    best = Path(row['checkpoint'])
    if best.resolve() != (last.parent / 'best.pt').resolve():
        raise RuntimeError('completed checkpoint must belong to this training cell before cleanup')
    released = last.stat().st_size if last.exists() else 0
    if last.exists():
        last.unlink()
    row['checkpoint_retention'] = {'policy': 'completed_best_only', 'best_sha256': row['checkpoint_sha256'],
                                   'last_removed': True, 'released_last_bytes': released}
    save_matrix_state(admission, row)


def verify_dataset_identity(root: PathLike, record: Mapping) -> None:
    native = Path(root)
    if (_sha(native / 'dataset.json') != record['dataset_manifest_sha256']
            or json_load(native / 'dataset.json').get('fingerprint') != record['dataset_fingerprint']
            or {p.name: _sha(p) for p in sorted((native / 'splits').glob('*.txt'))} != record['split_sha256']):
        raise RuntimeError('native manifest/fingerprint/split bytes changed after qualification')


def _snapshot_regular_file(path: PathLike, label: str, *, keep_bytes: bool) -> dict:
    """Read one trusted small file once and bind its bytes to its pathname."""
    path = Path(path)
    raw_parent = path.parent
    if raw_parent.is_symlink() or path.is_symlink():
        raise RuntimeError(f'completed run {label} path must not contain a symlink')
    try:
        root = raw_parent.resolve(strict=True)
        root_before = root.stat(follow_symlinks=False)
        canonical = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f'completed run {label} is missing') from exc
    if canonical.parent != root:
        raise RuntimeError(f'completed run {label} path escapes its trusted parent')
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as exc:
        raise RuntimeError(f'completed run {label} cannot be safely read') from exc
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f'completed run {label} is not a regular file')
        digest = hashlib.sha256()
        chunks = [] if keep_bytes else None
        size = 0
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after) or size != after.st_size:
        raise RuntimeError(f'completed run {label} changed during validation')
    try:
        current = path.stat(follow_symlinks=False)
        root_after = root.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f'completed run {label} pathname changed') from exc
    directory_identity = lambda value: (value.st_dev, value.st_ino, value.st_mode)
    if (not stat.S_ISREG(current.st_mode) or identity(current) != identity(after)
            or directory_identity(root_before) != directory_identity(root_after)
            or path.is_symlink() or path.resolve(strict=True).parent != root):
        raise RuntimeError(f'completed run {label} pathname changed')
    snapshot = {'path': canonical, 'sha256': digest.hexdigest(), 'size': size}
    if chunks is not None:
        snapshot['bytes'] = b''.join(chunks)
    return snapshot


def _completed_file_context(row: Mapping) -> dict:
    checkpoint = _snapshot_regular_file(
        row.get('checkpoint', ''), 'checkpoint', keep_bytes=False)
    if checkpoint['sha256'] != row.get('checkpoint_sha256'):
        raise RuntimeError('completed run checkpoint changed; refusing implicit reuse')
    identity = _snapshot_regular_file(
        row.get('run_identity_path', ''), 'run_identity_path', keep_bytes=True)
    if identity['sha256'] != row.get('run_identity_sha256'):
        raise RuntimeError('completed run run_identity_path changed; refusing implicit reuse')
    try:
        payload = json.loads(identity['bytes'].decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError('completed run identity is not valid UTF-8 JSON') from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError('completed run identity must be a JSON object')
    return {'checkpoint': checkpoint, 'run_identity': dict(payload),
            'run_identity_file': identity}


def _verify_transfer_provenance(row: Mapping, coverage_rows: Optional[list] = None,
                                transfer_origin: Optional[Mapping] = None,
                                transfer_origin_context: Optional[Mapping] = None) -> None:
    """Bind a test-only cell to the exact qualified training cell and checkpoint."""
    if row.get('trainable') is True:
        if row.get('checkpoint_source') is not None or row.get('cross_domain_transfer') is not None:
            raise RuntimeError('trainable completed run carries transfer-only provenance')
        return
    source_dataset = row.get('checkpoint_training_source')
    # Static/legacy non-matrix rows have neither field and no transfer claim to verify.
    if source_dataset is None and row.get('checkpoint_source') is None:
        return
    if not isinstance(source_dataset, str) or not source_dataset:
        raise RuntimeError('transfer completed run lacks checkpoint_training_source')
    source = row.get('checkpoint_source')
    if not isinstance(source, Mapping) or set(source) != {'run_id', 'path', 'sha256'}:
        raise RuntimeError('transfer completed run checkpoint_source is not a strict provenance object')
    expected_id = matrix_row_id(
        row.get('profile'), source_dataset, row.get('seed'), row.get('input_track'))
    if source.get('run_id') != expected_id:
        raise RuntimeError('transfer completed run checkpoint_source run_id mismatch')
    checkpoint = Path(str(row.get('checkpoint', '')))
    source_path = Path(str(source.get('path', '')))
    try:
        same_path = source_path.resolve(strict=True) == checkpoint.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError('transfer completed run checkpoint_source path is missing') from exc
    if not same_path:
        raise RuntimeError('transfer completed run checkpoint_source path mismatch')
    if source.get('sha256') != row.get('checkpoint_sha256'):
        raise RuntimeError('transfer completed run checkpoint_source sha256 mismatch')
    cross_domain = row.get('cross_domain_transfer')
    if type(cross_domain) is not bool:
        raise RuntimeError('transfer completed run cross_domain_transfer must be boolean')
    if coverage_rows is not None:
        authoritative = next((candidate for candidate in coverage_rows
                              if candidate.get('profile') == row.get('profile')
                              and candidate.get('dataset') == source_dataset
                              and candidate.get('seed') == row.get('seed')
                              and candidate.get('input_track') == row.get('input_track')), None)
        if authoritative is None or authoritative.get('id', expected_id) != expected_id:
            raise RuntimeError('transfer completed run has no authoritative source row')
        origin = transfer_origin or authoritative
        if (origin.get('id', expected_id) != expected_id
                or origin.get('profile') != authoritative.get('profile')
                or origin.get('dataset') != authoritative.get('dataset')
                or origin.get('seed') != authoritative.get('seed')
                or origin.get('input_track') != authoritative.get('input_track')):
            raise RuntimeError('transfer completed run source row identity mismatch')
        if origin.get('state') != 'completed':
            raise RuntimeError('transfer completed run authoritative source is not completed')
        if origin.get('trainable') is not True:
            raise RuntimeError('transfer completed run authoritative source is not trainable')
        origin_context = transfer_origin_context
        if origin_context is None:
            origin_context = verify_completed_state(origin, coverage_rows)
        if not isinstance(origin_context, Mapping):
            raise RuntimeError('transfer completed run lacks verified source context')
        try:
            origin_path = Path(str(origin['checkpoint'])).resolve(strict=True)
        except (KeyError, FileNotFoundError) as exc:
            raise RuntimeError('transfer completed run source checkpoint is missing') from exc
        context_checkpoint = origin_context.get('checkpoint')
        if (not isinstance(context_checkpoint, Mapping)
                or context_checkpoint.get('path') != origin_path
                or context_checkpoint.get('sha256') != origin.get('checkpoint_sha256')
                or str(row.get('checkpoint')) != str(origin_path)
                or source.get('path') != str(origin_path)
                or source_path.resolve(strict=True) != origin_path
                or source.get('sha256') != origin.get('checkpoint_sha256')):
            raise RuntimeError(
                'transfer completed run checkpoint_source differs from authoritative origin')
        origin_identity = origin_context.get('run_identity')
        if not isinstance(origin_identity, Mapping):
            raise RuntimeError('transfer completed run source identity is not verified')
        identity_fields = ('profile', 'seed', 'input_track', 'dataset_fingerprint',
                           'profile_config_sha256', 'scientific_input_spec')
        if any(origin.get(key) is not None and origin_identity.get(key) != origin.get(key)
               for key in identity_fields):
            raise RuntimeError('transfer completed run source scientific identity mismatch')
        expected_cross_domain = origin.get('domain') != row.get('domain')
        if cross_domain != expected_cross_domain:
            raise RuntimeError('transfer completed run cross_domain_transfer mismatch')


def verify_completed_state(row: Mapping, coverage_rows: Optional[list] = None,
                           transfer_origin: Optional[Mapping] = None,
                           transfer_origin_context: Optional[Mapping] = None) -> Optional[dict]:
    if row.get('state') != 'completed':
        return None
    context = _completed_file_context(row)
    _verify_transfer_provenance(
        row, coverage_rows, transfer_origin, transfer_origin_context)
    spec, expected = _expected_eval_splits(row)
    taskview = row.get('taskview_receipt') or {}
    if taskview.get('status') != 'passed':
        raise RuntimeError('completed campaign row lacks passed taskview qualification')
    receipts = taskview.get('splits')
    split_snapshots = {}
    qualified_applicable, qualified_na = _qualified_eval_splits(
        spec, expected, receipts, row=row, snapshots=split_snapshots)
    statuses = (row.get('result') or {}).get('splits')
    if not isinstance(statuses, Mapping) or list(statuses) != expected:
        raise RuntimeError('completed run status omits, invents, or reorders an evaluation split')
    if any(statuses[split] not in ('evaluated', 'skipped (done)')
           for split in qualified_applicable):
        raise RuntimeError('completed run has an applicable split without an evaluation result')
    for split in qualified_na:
        if statuses[split] != 'N/A (not_applicable_missing_split)':
            raise RuntimeError('completed run N/A split status is inconsistent')

    run_root = Path(str(row.get('run_identity_path', ''))).parent
    artifacts = row.get('result_artifacts')
    # Legacy completed states predate hash bindings.  They remain reusable only
    # after the same strict, read-only validation of every currently qualified
    # result.  We deliberately do not mutate/backfill their state here.
    if artifacts is None:
        for split in qualified_applicable:
            _validate_result_split(
                row, split, run_dir=run_root, split_snapshot=split_snapshots[split])
        return context
    if not isinstance(artifacts, Mapping) or artifacts.get('schema') != 'ib.completed-results.v1':
        raise RuntimeError('completed run has an unsupported evaluation result binding')
    bound_applicable = artifacts.get('applicable_splits')
    bound_not_applicable = artifacts.get('not_applicable_splits')
    split_records = artifacts.get('splits')
    if (not isinstance(bound_applicable, list) or not bound_applicable
            or not isinstance(bound_not_applicable, Mapping) or not isinstance(split_records, Mapping)):
        raise RuntimeError('completed run has no applicable verified evaluation splits')
    if (bound_applicable != qualified_applicable or list(split_records) != qualified_applicable
            or bound_not_applicable != {split: 'not_applicable_missing_split'
                                       for split in qualified_na}):
        raise RuntimeError('completed run evaluation split manifest is inconsistent')
    for split in qualified_applicable:
        record = split_records.get(split)
        if not isinstance(record, Mapping):
            raise RuntimeError(f'completed run lacks result binding for split {split!r}')
        _validate_result_split(
            row, split, record=record, split_snapshot=split_snapshots[split])
    return context


def _safe_split_name(split: Any) -> str:
    if (not isinstance(split, str) or not split or split in ('.', '..')
            or '/' in split or '\\' in split or '\x00' in split
            or Path(split).name != split):
        raise RuntimeError(f'unsafe evaluation split name: {split!r}')
    return split


def _expected_eval_splits(row: Mapping, spec: Optional[Any] = None) -> tuple:
    """Return the exact non-official evaluation contract from the dataset spec."""
    spec = spec or resolve_dataset(row.get('data_root') or row.get('data'))
    expected = []
    for split in spec.test_splits:
        split = _safe_split_name(split)
        if split.startswith('official_'):
            continue
        if split in expected:
            raise RuntimeError(f'dataset repeats evaluation split {split!r}')
        expected.append(split)
    if not expected:
        raise RuntimeError('dataset declares no non-official evaluation splits')
    return spec, expected


def _split_snapshot(row: Optional[Mapping], spec: Any, split: str) -> Optional[dict]:
    """Read one split once and bind the parsed IDs to its qualified bytes."""
    split = _safe_split_name(split)
    raw_root = Path(spec.root) / 'splits'
    path = spec.split_file(split)
    hashes = row.get('split_sha256') if row is not None else None
    bound_when_qualified = isinstance(hashes, Mapping) and path.name in hashes

    def missing() -> None:
        if bound_when_qualified:
            raise RuntimeError(
                f'evaluation split {split!r} disappeared after qualified split_sha256')
        return None

    if raw_root.is_symlink():
        raise RuntimeError('dataset split root must not be a symlink')
    try:
        root = raw_root.resolve(strict=True)
        root_before = root.stat(follow_symlinks=False)
    except FileNotFoundError:
        return missing()
    if not stat.S_ISDIR(root_before.st_mode):
        raise RuntimeError('dataset split root is not a directory')
    if path.is_symlink():
        raise RuntimeError(f'evaluation split {split!r} resolves outside the dataset split root')
    if not path.exists():
        return missing()
    try:
        canonical = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f'evaluation split {split!r} pathname changed') from exc
    if canonical.parent != root:
        raise RuntimeError(f'evaluation split {split!r} resolves outside the dataset split root')
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as exc:
        raise RuntimeError(f'evaluation split {split!r} cannot be safely read') from exc
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f'evaluation split {split!r} is not a regular split file')
        payload = stream.read()
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RuntimeError(f'evaluation split {split!r} changed during validation')
    try:
        current = path.stat(follow_symlinks=False)
        root_after = root.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f'evaluation split {split!r} pathname changed') from exc
    directory_identity = lambda value: (value.st_dev, value.st_ino, value.st_mode)
    if (not stat.S_ISREG(current.st_mode) or identity(current) != identity(after)
            or directory_identity(root_before) != directory_identity(root_after)
            or path.is_symlink() or path.resolve(strict=True).parent != root):
        raise RuntimeError(f'evaluation split {split!r} pathname changed')
    digest = hashlib.sha256(payload).hexdigest()
    if row is not None:
        if not isinstance(hashes, Mapping) or hashes.get(path.name) != digest:
            raise RuntimeError(
                f'evaluation split {split!r} changed from qualified split_sha256')
    try:
        text = payload.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise RuntimeError(f'evaluation split {split!r} is not UTF-8') from exc
    identifiers = []
    for line in text.splitlines():
        value = line.split('#', 1)[0].strip()
        if value:
            identifiers.append(value)
    return {'ids': identifiers, 'sha256': digest, 'path': str(canonical)}


def _split_applicable(spec: Any, split: str, receipts: Optional[Mapping],
                      row: Optional[Mapping] = None,
                      snapshots: Optional[dict] = None) -> bool:
    """Return the qualified applicability of one evaluation split.

    Dynamic campaigns must use their taskview receipt rather than infer from a
    directory left by an older run.  Static plans have no receipt, so both a
    split file and at least one declared sequence are required.
    """
    split = _safe_split_name(split)
    snapshot = _split_snapshot(row, spec, split)
    if snapshots is not None:
        snapshots[split] = snapshot
    ids = snapshot['ids'] if snapshot is not None else None
    if receipts is not None:
        if split not in receipts or not isinstance(receipts[split], Mapping):
            raise RuntimeError(f'taskview receipt lacks evaluation split {split!r}')
        status = receipts[split].get('status')
        if status == 'not_applicable_missing_split':
            if ids:
                raise RuntimeError(f'evaluation split {split!r} is nonempty but taskview marks it N/A')
            return False
        if status != 'passed':
            raise RuntimeError(f'evaluation split {split!r} taskview is not passed: {status!r}')
        if not ids:
            raise RuntimeError(f'qualified applicable split {split!r} is now missing or empty')
        _int_field(receipts[split].get('valid_windows'), 'taskview valid_windows', minimum=1)
        return True
    return bool(ids)


def _qualified_eval_splits(spec: Any, expected: list,
                           receipts: Optional[Mapping], *, require_applicable: bool = True,
                           row: Optional[Mapping] = None, snapshots: Optional[dict] = None
                           ) -> tuple[list, list]:
    if not isinstance(receipts, Mapping):
        raise RuntimeError('completed campaign row lacks taskview split qualification')
    auxiliary = {'train', 'val'} - set(expected)
    receipt_eval = [split for split in receipts if split not in auxiliary]
    if receipt_eval != expected:
        raise RuntimeError('taskview receipt does not exactly cover dataset evaluation splits')
    applicable, not_applicable = [], []
    for split in expected:
        (applicable if _split_applicable(spec, split, receipts, row, snapshots)
         else not_applicable).append(split)
    if require_applicable and not applicable:
        raise RuntimeError('coverage cell has no applicable nonempty evaluation split')
    return applicable, not_applicable


def _int_field(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f'evaluation result {name} must be an integer >= {minimum}')
    return value


def _validate_result_split(row: Mapping, split: str, run_dir: Optional[PathLike] = None,
                           record: Optional[Mapping] = None,
                           split_snapshot: Optional[Mapping] = None) -> dict:
    """Validate and optionally bind the two small, authoritative result files."""
    if (run_dir is None) == (record is None):
        raise ValueError('provide exactly one of run_dir or record')
    split = _safe_split_name(split)
    run_root = Path(run_dir) if record is None else Path(str(row.get('run_identity_path', ''))).parent
    if run_root.is_symlink():
        raise RuntimeError('evaluation run root must not be a symlink')
    canonical_root = run_root.resolve()
    split_dir = run_root / split
    if split_dir.is_symlink() or split_dir.resolve().parent != canonical_root:
        raise RuntimeError(f'evaluation split {split!r} output resolves outside the run root')
    try:
        root_before = canonical_root.stat(follow_symlinks=False)
        split_before = split_dir.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f'applicable split {split!r} output directory is missing') from exc
    if not stat.S_ISDIR(root_before.st_mode) or not stat.S_ISDIR(split_before.st_mode):
        raise RuntimeError(f'evaluation split {split!r} output parent is not a directory')
    metrics_path = split_dir / 'metrics.json'
    sequences_path = split_dir / 'sequences.csv'
    canonical_paths = (metrics_path.resolve(), sequences_path.resolve())
    if record is not None and (Path(str(record.get('metrics_path', ''))) != canonical_paths[0]
            or Path(str(record.get('sequences_path', ''))) != canonical_paths[1]):
        raise RuntimeError(f'completed split {split!r} result paths changed')

    def read_once(path: Path, label: str) -> tuple[bytes, str]:
        if path.is_symlink():
            raise RuntimeError(f'applicable split {split!r} {label} must not be a symlink')
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        except FileNotFoundError as exc:
            raise RuntimeError(f'applicable split {split!r} lacks {label}') from exc
        except OSError as exc:
            raise RuntimeError(f'applicable split {split!r} cannot safely read {label}') from exc
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f'applicable split {split!r} {label} is not a regular file')
            payload = stream.read()
            after = os.fstat(stream.fileno())
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if identity(before) != identity(after) or len(payload) != after.st_size:
            raise RuntimeError(f'applicable split {split!r} {label} changed during validation')
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f'applicable split {split!r} {label} pathname changed') from exc
        if (not stat.S_ISREG(current.st_mode)
                or identity(current) != identity(after)
                or path.is_symlink() or path.resolve().parent != split_dir.resolve()):
            raise RuntimeError(f'applicable split {split!r} {label} pathname changed')
        return payload, hashlib.sha256(payload).hexdigest()

    metrics_bytes, metrics_sha = read_once(metrics_path, 'metrics.json')
    sequences_bytes, sequences_sha = read_once(sequences_path, 'sequences.csv')
    try:
        root_after = canonical_root.stat(follow_symlinks=False)
        split_after = split_dir.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise RuntimeError(f'evaluation split {split!r} output directory changed') from exc
    directory_identity = lambda value: (value.st_dev, value.st_ino, value.st_mode)
    if (directory_identity(root_before) != directory_identity(root_after)
            or directory_identity(split_before) != directory_identity(split_after)
            or split_dir.is_symlink() or split_dir.resolve().parent != canonical_root):
        raise RuntimeError(f'evaluation split {split!r} output directory changed')
    if record is not None:
        if metrics_sha != record.get('metrics_sha256'):
            raise RuntimeError(f'completed split {split!r} metrics.json changed')
        if sequences_sha != record.get('sequences_sha256'):
            raise RuntimeError(f'completed split {split!r} sequences.csv changed')
    try:
        metrics = json.loads(metrics_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'evaluation split {split!r} metrics.json is invalid') from exc
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f'evaluation split {split!r} metrics.json is not an object')
    if metrics.get('mode') != 'val':
        raise RuntimeError(f'evaluation split {split!r} is not a mode=val result')
    expected = {'model': row.get('profile'), 'dataset': row.get('dataset'),
                'split': split, 'seed': row.get('seed')}
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f'evaluation split {split!r} {key} identity mismatch')
    dims = row.get('dims')
    if dims not in (2, 3):
        raise RuntimeError('coverage row lacks a valid 2D/3D evaluation dimension')
    axes = list('xyz'[:dims])
    evidence = (metrics.get('metric_dims_requested'), metrics.get('metric_dims_effective'),
                metrics.get('predicted_axes'))
    if ((metrics.get('protocol') or {}).get('metric_dims') != dims
            or evidence != (dims, dims, axes)):
        raise RuntimeError(f'evaluation split {split!r} dimension evidence mismatch')
    num_sequences = _int_field(metrics.get('num_sequences'), 'num_sequences', minimum=1)
    num_skipped = _int_field(metrics.get('num_skipped'), 'num_skipped')
    aggregates = {}
    for name in ('metrics', 'median', 'std', 'count'):
        value = metrics.get(name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f'evaluation split {split!r} {name} must be a mapping')
        aggregates[name] = value
    metric_keys = set(aggregates['metrics'])
    if (not metric_keys or any(set(aggregates[name]) != metric_keys
                               for name in ('median', 'std', 'count'))):
        raise RuntimeError(f'evaluation split {split!r} aggregate metric keys are inconsistent or empty')

    try:
        with io.StringIO(sequences_bytes.decode('utf-8'), newline='') as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise RuntimeError(f'evaluation split {split!r} sequences.csv is not UTF-8') from exc
    required = {'sequence_id', 'skipped', 'metric_dims_requested',
                'metric_dims_effective', 'predicted_axes'}
    if (not fieldnames or len(fieldnames) != len(set(fieldnames))
            or not required <= set(fieldnames)):
        raise RuntimeError(f'evaluation split {split!r} sequences.csv lacks required headers')
    if any(None in entry for entry in rows):
        raise RuntimeError(f'evaluation split {split!r} sequences.csv has malformed extra fields')
    if len(rows) != num_sequences + num_skipped:
        raise RuntimeError(f'evaluation split {split!r} CSV row count disagrees with metrics.json')
    evaluated = [entry for entry in rows if not entry.get('skipped')]
    if len(evaluated) != num_sequences:
        raise RuntimeError(f'evaluation split {split!r} evaluated CSV rows disagree with metrics.json')
    identifiers = [entry.get('sequence_id') for entry in rows]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f'evaluation split {split!r} sequence IDs are empty or duplicated')
    if split_snapshot is None:
        split_snapshot = _split_snapshot(
            row, resolve_dataset(row.get('data_root') or row.get('data')), split)
    expected_ids = split_snapshot['ids'] if split_snapshot is not None else None
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError(f'evaluation split {split!r} dataset sequence IDs are empty or duplicated')
    # Dataset order is not a result invariant: Validator may schedule/load rows
    # differently.  Completeness is exact set equality; CSV uniqueness above
    # makes the comparison cardinality-preserving.
    if set(identifiers) != set(expected_ids):
        raise RuntimeError(f'evaluation split {split!r} CSV sequence IDs do not exactly match the split')
    for entry in evaluated:
        try:
            csv_dims = (int(entry['metric_dims_requested']), int(entry['metric_dims_effective']))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f'evaluation split {split!r} CSV dimension evidence is invalid') from exc
        if csv_dims != (dims, dims) or entry['predicted_axes'] != ','.join(axes):
            raise RuntimeError(f'evaluation split {split!r} CSV dimension evidence mismatch')

    def finite_number(value: Any) -> bool:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value)))

    if not metric_keys <= set(fieldnames):
        raise RuntimeError(f'evaluation split {split!r} CSV lacks aggregate metric columns')
    for key in metric_keys:
        count = _int_field(aggregates['count'][key], f'count[{key}]')
        if count > num_sequences:
            raise RuntimeError(f'evaluation split {split!r} count[{key}] exceeds num_sequences')
        values, value_rows = [], []
        for index, entry in enumerate(evaluated):
            raw = entry.get(key)
            if raw is None or not str(raw).strip():
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f'evaluation split {split!r} CSV metric {key!r} is not numeric') from exc
            if not math.isfinite(value):
                raise RuntimeError(
                    f'evaluation split {split!r} CSV metric {key!r} is not finite')
            values.append(value)
            value_rows.append(index)
        if count != len(values):
            raise RuntimeError(f'evaluation split {split!r} count[{key}] disagrees with CSV')
        for name in ('metrics', 'median', 'std'):
            value = aggregates[name][key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise RuntimeError(f'evaluation split {split!r} {name}[{key}] must be numeric or null')
        if count and (not finite_number(aggregates['metrics'][key])
                      or not finite_number(aggregates['median'][key])):
            raise RuntimeError(f'evaluation split {split!r} aggregate {key} is non-finite')
        if not count and any(aggregates[name][key] is not None
                             for name in ('metrics', 'median', 'std')):
            raise RuntimeError(f'evaluation split {split!r} aggregate {key} disagrees with zero count')
        if count == 1 and aggregates['std'][key] is not None:
            raise RuntimeError(f'evaluation split {split!r} std[{key}] disagrees with one sample')
        if count >= 2 and not finite_number(aggregates['std'][key]):
            raise RuntimeError(f'evaluation split {split!r} std[{key}] is non-finite')
        if count:
            mean = float(np.mean(values))
            if key == 'loss' and 'num_windows' in fieldnames:
                weights = []
                for index in value_rows:
                    try:
                        weight = float(evaluated[index].get('num_windows', ''))
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f'evaluation split {split!r} loss num_windows is not numeric') from exc
                    if not math.isfinite(weight) or weight < 0:
                        raise RuntimeError(
                            f'evaluation split {split!r} loss num_windows is invalid')
                    weights.append(weight)
                if sum(weights) > 0:
                    mean = float(np.average(values, weights=weights))
            expected_aggregates = {
                'metrics': mean,
                'median': float(np.median(values)),
                'std': float(np.std(values, ddof=1)) if count > 1 else None,
            }
            for name, target in expected_aggregates.items():
                actual = aggregates[name][key]
                if target is None:
                    if actual is not None:
                        raise RuntimeError(
                            f'evaluation split {split!r} {name}[{key}] disagrees with CSV')
                elif (not finite_number(actual)
                      or not math.isclose(float(actual), target,
                                          rel_tol=1e-9, abs_tol=1e-12)):
                    raise RuntimeError(
                        f'evaluation split {split!r} {name}[{key}] disagrees with CSV')
    if not any(key in MAIN_METRICS and aggregates['count'][key] > 0
               and finite_number(aggregates['metrics'][key]) for key in metric_keys):
        raise RuntimeError(f'evaluation split {split!r} has no finite publishable main metric')

    bound = {'metrics_path': str(canonical_paths[0]), 'metrics_sha256': metrics_sha,
             'sequences_path': str(canonical_paths[1]),
             'sequences_sha256': sequences_sha, 'num_sequences': num_sequences,
             'num_skipped': num_skipped, 'csv_rows': len(rows)}
    if record is not None and any(record.get(key) != value for key, value in bound.items()):
        raise RuntimeError(f'completed split {split!r} result binding changed')
    return bound


def bind_completed_results(row: Mapping, run_dir: PathLike, splits: list,
                           receipts: Optional[Mapping]) -> dict:
    """Validate every applicable split and return state-file hash bindings."""
    if (row.get('taskview_receipt') or {}).get('status') != 'passed':
        raise RuntimeError('completed campaign row lacks passed taskview qualification')
    spec, expected = _expected_eval_splits(row)
    if list(splits) != expected:
        raise RuntimeError('requested evaluation splits do not exactly match dataset test_splits')
    snapshots = {}
    applicable, na_splits = _qualified_eval_splits(
        spec, expected, receipts, row=row, snapshots=snapshots)
    not_applicable = {split: 'not_applicable_missing_split' for split in na_splits}
    records = {}
    for split in applicable:
        records[split] = _validate_result_split(
            row, split, run_dir=run_dir, split_snapshot=snapshots[split])
    return {'schema': 'ib.completed-results.v1', 'applicable_splits': applicable,
            'not_applicable_splits': not_applicable, 'splits': records}


def verify_capacity_policy(policy: Mapping, source_sha: str, hardware: Optional[Mapping] = None) -> dict:
    path = Path(policy.get('capacity_receipt', ''))
    if not path.is_file() or _sha(path) != policy.get('capacity_receipt_sha256'):
        raise RuntimeError('frozen capacity receipt missing or changed')
    receipt = json_load(path)
    if receipt.get('formal_ready') is not True or receipt.get('source_manifest_sha256') != source_sha:
        raise RuntimeError('capacity lifecycle evidence is incomplete or belongs to a different source')
    bindings = {'batch': 'common_batch_candidate', 'val_batch': 'common_val_batch',
                'gap_windows': 'common_gap_windows_candidate', 'cache': 'common_cache',
                'workers': 'common_workers', 'amp': 'common_amp',
                'stride': 'common_training_stride', 'eval_stride': 'common_eval_stride'}
    if any(policy['overrides'].get(key) != receipt.get(field) for key, field in bindings.items()):
        raise RuntimeError('frozen recipe differs from actually measured capacity resource knobs')
    if hardware is not None:
        fields = ('name', 'total_memory_bytes', 'torch_version', 'cuda_version')
        if any(hardware.get(key) != receipt.get('hardware', {}).get(key) for key in fields):
            raise RuntimeError('actual GPU/environment differs from measured capacity hardware')
    return receipt


def scientific_run_identity(item: BenchmarkItem) -> dict:
    meta = item.metadata
    keys = ('dataset_fingerprint', 'dataset_manifest_sha256', 'split_sha256',
            'profile_config_sha256', 'scientific_input_spec', 'sampling_spec', 'input_track',
            'policy_sha256', 'source_manifest_sha256', 'transfer_checkpoint_sha256')
    identity = {**{key: meta.get(key) for key in keys}, 'profile': item.label, 'seed': item.seed,
                'seed_record_sha256': meta['seeds_record_sha256'], 'overrides': item.overrides}
    if item.eval_alias_reuse:
        from .eval_aliases import ALIAS_POLICY
        identity['eval_alias_policy'] = dict(ALIAS_POLICY)
    return identity


def dependency_order(items: list) -> list:
    """Evaluate each transfer directly after its exact origin when available."""
    by_key = {(i.label, i.metadata.get('dataset_name'), i.seed,
               i.metadata.get('input_track')): i for i in items}
    ordered, visiting, seen = [], set(), set()
    def visit(item):
        key = (item.label, item.metadata.get('dataset_name'), item.seed, item.metadata.get('input_track'))
        if key in seen:
            return
        if key in visiting:
            raise RuntimeError('cyclic matrix checkpoint dependency')
        visiting.add(key)
        origin = item.metadata.get('checkpoint_training_source') if not item.metadata.get('trainable') else None
        parent = by_key.get((item.label, origin, item.seed, item.metadata.get('input_track'))) if origin else None
        if parent is not None:
            visit(parent)
        visiting.remove(key); seen.add(key); ordered.append(item)
    for item in items:
        visit(item)
    return ordered


def _plan_path(spec: Mapping, key: str, environment: str) -> Path:
    value = os.environ.get(environment) or spec.get(key)
    if not value:
        raise ConfigError(f"dynamic benchmark requires {key} or {environment}")
    return Path(value).expanduser().resolve()


def _dynamic_entries(spec: Mapping) -> tuple:
    """Resolve active profiles and accepted independent natives without downloading data."""
    from ..cfg import model_yaml_names
    models = model_yaml_names() if spec.get("models") == "all_active_profiles" else spec.get("models")
    registry_path = _plan_path(spec, "native_registry", "IPB_NATIVE_REGISTRY")
    registry = json_load(registry_path)
    datasets, source_ids = [], set()
    for row in registry.get("datasets", []):
        if row.get("accepted") is not True:
            continue
        source = row.get("independent_source_id", row["name"])
        if source in source_ids:
            raise ConfigError(f"native aliases duplicate independent source {source}")
        source_ids.add(source)
        native = Path(row["root"]).resolve()
        manifest = json_load(native / "dataset.json")
        if (manifest.get("engineering_only") or not manifest.get("sequences")
                or manifest.get("fingerprint") != row["fingerprint"]):
            raise ConfigError(f"accepted native identity changed: {row['name']}")
        datasets.append({**row, "root": str(native), "manifest_sha256": _sha(native / "dataset.json"),
                         "split_sha256": {p.name: _sha(p) for p in sorted((native / "splits").glob("*.txt"))}})
    seed_path = _plan_path(spec, "seed_record", "IPB_RANDOM_SEED_RECORD")
    record = json_load(seed_path)
    seeds = record.get("seeds", [])
    if (record.get("schema") != "ib.benchmark.random-seeds.v1"
            or record.get("entropy_source") != "operating_system_random_via_python_secrets"
            or record.get("sample_once") is not True or len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(s) is not int or not 0 <= s < 2 ** 31 for s in seeds)):
        raise ConfigError("main requires the same three unique once-sampled OS-random seeds")
    policy_path = _plan_path(spec, "policy", "IPB_FROZEN_POLICY")
    policy = json_load(policy_path) if policy_path.exists() else {}
    policy_overrides = policy.get("overrides", {}) if policy.get("status") == "frozen" else {}
    admission_path = _plan_path(spec, "admission", "IPB_MATRIX_ADMISSION")
    source_root = Path(__file__).resolve().parents[3]
    source_file = source_root / "source_manifest.json"
    source_hash = None
    if source_file.is_file():
        manifest = json_load(source_file)
        for name, expected in {**manifest.get('files', {}),
                               **manifest.get('operational_acceptance_helpers', {})}.items():
            if _sha(source_root / name) != expected:
                raise RuntimeError(f'managed benchmark source changed: {name}')
        source_hash = _sha(source_file)
    return models, datasets, seeds, policy_overrides, {
        "dynamic": True, "registry_path": str(registry_path), "registry_sha256": _sha(registry_path),
        "seed_record_path": str(seed_path), "seeds_record_sha256": _sha(seed_path),
        "policy_path": str(policy_path), "policy_sha256": _sha(policy_path) if policy_path.exists() else None,
        "policy_state": policy.get("status", "pending_capacity_measurement_and_freeze"),
        "admission_path": str(admission_path), "source_root": str(source_root),
        "source_manifest_sha256": source_hash}


def load_plan(cfg: Union[PathLike, Mapping], overrides: Optional[dict] = None) -> list:
    """解析基准 YAML，返回 :class:`BenchmarkItem` 列表（模型 × 数据集 × 种子）。"""
    cfg = resolve_benchmark_cfg(cfg)
    spec = dict(yaml_load(cfg)) if not isinstance(cfg, Mapping) else dict(cfg)
    unknown = set(spec) - set(PLAN_KEYS)
    if unknown:
        raise ConfigError(f"unknown benchmark keys: {sorted(unknown)}")
    alias_reuse = spec.get('eval_alias_reuse', False)
    if type(alias_reuse) is not bool:
        raise ConfigError('eval_alias_reuse must be a boolean')
    name = spec.get("name") or (Path(str(cfg)).stem if not isinstance(cfg, Mapping) else "bench")
    common = {**(spec.get("overrides") or {}), **(overrides or {})}
    # 输出根目录：命令行的 project= 优先于 YAML（同一份矩阵可以写到别的盘而不改配置）
    root = Path(common.get("project") or spec.get("project") or "runs/benchmark") / name
    items = []
    dynamic = spec.get("models") == "all_active_profiles" or spec.get("datasets") == "accepted_native"
    if dynamic:
        from .budget import is_sequence_model
        models, datasets, seeds, policy_overrides, shared = _dynamic_entries(spec)
        for m in models or []:
            label, model, m_over = _entry(m, "model")
            resolved = get_cfg({**common, **m_over, "model": model})
            sequence_model = is_sequence_model(model)
            privileged = resolved.orientation == "reference" or "init_velocity" in resolved.extra_inputs
            for d in datasets:
                data_spec = resolve_dataset(d["root"])
                train = data_spec.has_split("train") and bool(data_spec.sequence_ids("train"))
                val = data_spec.has_split("val") and bool(data_spec.sequence_ids("val"))
                state = "pending_admission" if train and val else (
                    "test_only_checkpoint_required" if not train else "unsupported_missing_validation")
                source = ("euroc" if d["name"] == "blackbird" else d.get("checkpoint_training_source"))
                for seed in seeds:
                    over = {**common, **m_over, **policy_overrides, "seed": seed,
                            "metric_dims": int(resolved.dims), "device": "cpu" if sequence_model else "0"}
                    metadata = {**shared, "dataset_name": d["name"], "domain": d.get("domain", "unknown"),
                                "independent_source_id": d.get("independent_source_id", d["name"]),
                                "dataset_fingerprint": d["fingerprint"], "dataset_manifest_sha256": d["manifest_sha256"],
                                "split_sha256": d["split_sha256"], "trainable": bool(train and val),
                                "native_xyz": int(resolved.dims) == 3, "sequence_calibration": sequence_model,
                                "input_track": "reference_oracle" if privileged else "deployment",
                                "state": state, "checkpoint_training_source": source,
                                "scientific_input_spec": {key: getattr(resolved, key) for key in INPUT_KEYS},
                                "sampling_spec": {'stride': int(over.get('stride', resolved.stride)),
                                                  'eval_stride': int(over.get('eval_stride', resolved.eval_stride))},
                                "profile_config_sha256": _sha(CFG_DIR / 'models' / f'{Path(str(model)).stem}.yaml')}
                    metadata['coverage_row_id'] = matrix_row_id(label, d['name'], seed, metadata['input_track'])
                    splits = [s for s in data_spec.test_splits if not s.startswith("official_")]
                    items.append(BenchmarkItem(label, model, d["root"], seed, over, splits, root,
                                               metadata, alias_reuse))
        if not items:
            raise ConfigError("dynamic benchmark has no accepted independent native datasets or active profiles")
        return items
    for m in spec.get("models") or []:
        label, model, m_over = _entry(m, "model")
        for d in spec.get("datasets") or []:
            _, data, d_over = _entry(d, "data")
            for seed in spec.get("seeds") or [0]:
                over = {**common, **m_over, **d_over, "seed": int(seed)}
                get_cfg({**over, "model": model, "data": data})  # 提前发现配置错误
                items.append(BenchmarkItem(label, model, data, int(seed), over,
                                           spec.get("splits"), root, eval_alias_reuse=alias_reuse))
    if not items:
        raise ConfigError("benchmark needs at least one model and one dataset")
    return items


def plan_item(item: BenchmarkItem) -> dict:
    """``dry_run`` 用的计划条目：run 目录、每轮窗口数与等窗口预算换算出的 epoch 数。

    epoch 数在这里就能算出来（惰性扫描 train 划分的有效窗口数，不载入序列），因此
    ``dry_run=true`` 能在真正开跑之前给出“每个 run 跑几轮”。
    """
    from .budget import count_train_windows, is_sequence_model, resolve_budget

    cfg = get_cfg({**item.overrides, "model": item.model, "data": item.data})
    spec = resolve_dataset(item.data)
    status = {"label": item.label, "data": str(item.data), "seed": item.seed,
              "dir": str(item.run_dir), "train": "planned", "epochs": int(cfg.epochs)}
    if item.metadata:
        status.update(coverage=item.metadata, train=item.metadata["state"])
    sequence_model = not is_checkpoint(item.model) and is_sequence_model(item.model)
    if sequence_model:
        status["train"] = "planned (calibrate once)"
    if int(cfg.train_windows_budget or 0) and not sequence_model:
        try:
            windows = count_train_windows(cfg, spec=spec)
            budget = resolve_budget(cfg, windows) or {}
            status.update(epochs=budget.get("epochs", cfg.epochs), train_budget=budget)
        except Exception as exc:  # noqa: BLE001 - 规划失败不应阻断其余条目的计划
            status["epochs"] = None
            status["error"] = f"{type(exc).__name__}: {exc}"
    status["splits"] = list(item.splits or spec.test_splits)
    if item.eval_alias_reuse:
        from .eval_aliases import ALIAS_POLICY
        status['eval_alias_policy'] = dict(ALIAS_POLICY)
    return status


def _admit_dynamic(item: BenchmarkItem) -> Optional[str]:
    """Admission is explicit; an unqualified row remains visible and does not train."""
    meta = item.metadata
    if not meta:
        return None
    if not meta.get('source_manifest_sha256'):
        return 'awaiting_frozen_source_identity'
    if meta["state"] == "unsupported_missing_validation":
        return "unsupported_missing_validation"
    policy_path = Path(meta["policy_path"])
    policy = json_load(policy_path) if policy_path.exists() else {}
    if policy.get("status") != "frozen" or _sha(policy_path) != meta.get("policy_sha256"):
        return "awaiting_frozen_common_policy"
    required = ("epochs", "batch", "optimizer", "lr", "scheduler", "weight_decay", "val_interval",
                "patience", "min_epochs", "val_batch", "amp", "cache", "workers", "gap_windows",
                "train_windows_budget")
    overrides = policy.get("overrides", {})
    if (any(k not in overrides for k in required) or int(overrides["train_windows_budget"]) != 0
            or overrides.get("loss") is not None):
        raise ConfigError("frozen primary recipe must use common fixed epochs and preserve scientific losses")
    try:
        verify_capacity_policy(policy, meta['source_manifest_sha256'])
    except RuntimeError:
        return "awaiting_actual_gpu_lifecycle_capacity"
    admission = Path(meta["admission_path"])
    if not admission.is_file():
        return "awaiting_taskview_and_model_cpu_acceptance"
    admission_record = json_load(admission)
    if admission_record.get('source', {}).get('source_manifest_sha256') != meta['source_manifest_sha256']:
        return 'awaiting_current_source_acceptance'
    records = admission_record.get("coverage", [])
    row = next((r for r in records if r.get("profile") == item.label
                and r.get("dataset") == meta["dataset_name"] and r.get("seed") == item.seed
                and r.get("input_track") == meta["input_track"]), None)
    if (not row or row.get("dataset_fingerprint") != meta["dataset_fingerprint"]
            or row.get('sampling_spec') != meta.get('sampling_spec')
            or row.get("taskview_receipt", {}).get("status") != "passed"
            or not row.get("model_cpu_accepted")):
        return "awaiting_taskview_and_model_cpu_acceptance"
    verify_dataset_identity(item.data, meta)
    row = matrix_state(admission, row)
    item.coverage_rows = records
    training = None
    training_context = None
    if not meta["trainable"]:
        origin = meta.get("checkpoint_training_source")
        if not origin:
            return "test_only_transfer_source_unspecified"
        training = next((r for r in records if r.get("profile") == item.label
                         and r.get("dataset") == origin and r.get("seed") == item.seed
                         and r.get("input_track") == meta["input_track"]), None)
        if training:
            training = matrix_state(admission, training)
            if training.get('state') == 'completed':
                training_context = verify_completed_state(training, records)
    completed_context = verify_completed_state(
        row, records, training, training_context)
    meta['admitted_row'] = row
    item.completed_verification = completed_context
    item.transfer_origin = training
    item.transfer_origin_verification = training_context
    if not meta["trainable"]:
        if (not training or training.get('state') != 'completed' or not training.get("checkpoint")
                or not Path(training["checkpoint"]).is_file()):
            return "awaiting_same_profile_track_seed_transfer_checkpoint"
        if training_context is None:
            return 'awaiting_verified_transfer_run_identity'
        identity = training_context['run_identity']
        if (identity.get('source_manifest_sha256') != meta['source_manifest_sha256']
                or identity.get('seed_record_sha256') != meta['seeds_record_sha256']
                or identity.get('seed') != item.seed or identity.get('policy_sha256') != meta['policy_sha256']
                or identity.get('profile') != item.label or identity.get('input_track') != meta['input_track']
                or identity.get('profile_config_sha256') != meta['profile_config_sha256']
                or identity.get('dataset_fingerprint') != training.get('dataset_fingerprint')
                or identity.get('scientific_input_spec') != meta['scientific_input_spec']):
            return 'awaiting_matching_transfer_scientific_identity'
        item.model = str(training_context['checkpoint']['path'])
        checkpoint_sha = training_context['checkpoint']['sha256']
        checkpoint_source = {'run_id': training['id'], 'path': str(item.model),
                             'sha256': checkpoint_sha}
        row['checkpoint_source'] = checkpoint_source
        row['cross_domain_transfer'] = training.get('domain') != row.get('domain')
        meta['checkpoint_source'] = checkpoint_source
        meta['cross_domain_transfer'] = row['cross_domain_transfer']
        meta["transfer_checkpoint_sha256"] = checkpoint_sha
    elif not meta["sequence_calibration"]:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("main learned training requires actual CUDA; no automatic CPU fallback")
        verify_capacity_policy(policy, meta['source_manifest_sha256'], {
            'name': torch.cuda.get_device_name(0),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda})
    return None


def train_summary(train_dir: Path) -> dict:
    """从训练目录读出 epoch、预算、选模标量与 train/eval 守卫摘要。"""
    args_file, metrics_file = train_dir / "args.yaml", train_dir / "metrics.json"
    out: dict = {}
    if not metrics_file.exists():
        return out
    meta = json_load(metrics_file)
    args = yaml_load(args_file) if args_file.exists() else {}
    key = args.get("fitness", "ate")
    stat = args.get("fitness_stat", "mean")
    values = meta.get("metrics" if stat == "mean" else stat) or {}
    out["epochs"] = meta.get("epochs", args.get("epochs"))
    out["best_epoch"] = meta.get("best_epoch")
    out["fitness"] = f"{stat} {key}"
    out["val_fitness"] = values.get(key)
    out["train_budget"] = meta.get("train_budget") or {}
    gap = meta.get("train_eval_gap") or {}
    verdict = gap.get("verdict") or {}
    if verdict:
        out["train_eval_gap"] = verdict.get("dropout_probe_delta")
        out["train_eval_gap_level"] = verdict.get("level")
        out["train_eval_speed_ratio"] = verdict.get("speed_ratio_eval")
    elif gap.get("ratio") is not None:  # 兼容旧 run
        out["train_eval_gap"] = gap["ratio"]
    return out


def run_item(item: BenchmarkItem) -> dict:
    """The public entry and ops dispatcher share the same exclusive cell ID."""
    if not item.metadata:
        return _run_item(item)
    meta = item.metadata
    identifier = matrix_row_id(item.label, meta['dataset_name'], item.seed, meta['input_track'])
    lock_path = Path(meta['admission_path']).parent / 'run-locks' / f'{identifier}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open('a+') as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            blocked = _admit_dynamic(item)
            if blocked:
                return {'label': item.label, 'data': str(item.data), 'seed': item.seed,
                        'dir': str(item.run_dir), 'train': blocked, 'splits': {}, 'error': blocked}
            row = meta['admitted_row']
            if row.get('state') == 'completed':
                context = item.completed_verification
                if context is None:
                    context = verify_completed_state(
                        row, item.coverage_rows, item.transfer_origin,
                        item.transfer_origin_verification)
                if context['run_identity'] != scientific_run_identity(item):
                    raise RuntimeError('completed shared run requested scientific identity changed')
                return {**row['result'], 'train': 'skipped (completed shared state)',
                        'dir': str(Path(row['run_identity_path']).parent),
                        'shared_completed_run': True, 'checkpoint': row['checkpoint']}
            with ExitStack() as resources:
                # AutoDL legacy live-source deployers use the same resource files.
                if str(meta.get('source_root', '')).startswith(PRIVATE_REMOTE_ROOT):
                    kind = 'cpu-resource.lock' if meta.get('sequence_calibration') else 'gpu-0.lock'
                    path = PRIVATE_LOCK_ROOT / kind
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handle = resources.enter_context(path.open('a+'))
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return _run_item(item)
    except BlockingIOError:
        return {'label': item.label, 'data': str(item.data), 'seed': item.seed,
                'dir': str(item.run_dir), 'train': 'pending_resource', 'splits': {},
                'error': 'matching run ID or resource is already owned by another worker'}


def _run_item(item: BenchmarkItem) -> dict:
    """训练（或跳过/续训）并在各测试划分上评测一项。"""
    status = {"label": item.label, "data": str(item.data), "seed": item.seed,
              "dir": str(item.run_dir), "train": None, "splits": {}}
    blocked = _admit_dynamic(item)
    if blocked:
        return {**status, "train": blocked, "coverage": item.metadata, "error": blocked}
    spec = resolve_dataset(item.data)
    splits = list(item.splits or spec.test_splits)
    for split in splits:
        _safe_split_name(split)
    receipts = item.split_receipts
    if receipts is None and item.metadata:
        receipts = item.metadata.get('admitted_row', {}).get('taskview_receipt', {}).get('splits')
    if receipts is not None:
        contract = item.metadata.get('admitted_row', {}) if item.metadata else {'data_root': item.data}
        _, expected = _expected_eval_splits(contract, spec)
        if splits != expected:
            raise RuntimeError('requested evaluation splits do not exactly match dataset test_splits')
        _qualified_eval_splits(
            spec, expected, receipts, require_applicable=True,
            row=item.metadata.get('admitted_row') if item.metadata else None)
    # Static plans also need an opt-in identity boundary. Legacy disabled plans
    # create no new file; they cannot silently reuse an enabled plan's outputs.
    alias_identity = item.run_dir / 'eval-alias-policy.json'
    if item.eval_alias_reuse or alias_identity.exists():
        from .eval_aliases import ALIAS_POLICY
        expected = dict(ALIAS_POLICY) if item.eval_alias_reuse else None
        if alias_identity.exists():
            if json_load(alias_identity) != expected:
                raise RuntimeError('evaluation alias policy changed; use a new benchmark run directory')
        elif any(item.run_dir.glob('*/metrics.json')):
            raise RuntimeError('existing outputs lack the requested evaluation alias policy identity')
        else:
            json_save(alias_identity, expected)
    if item.metadata:
        identity_path = item.run_dir / "run-identity.json"
        identity = scientific_run_identity(item)
        if identity_path.exists() and json_load(identity_path) != identity:
            raise RuntimeError("resume scientific source/data/seed/recipe identity changed")
        if not identity_path.exists() and any(item.run_dir.glob("*/metrics.json")):
            raise RuntimeError("existing outputs lack a scientific identity; refusing implicit reuse")
        json_save(identity_path, identity)
        json_save(item.run_dir / 'inventory-audit.json', item.metadata)
    train_dir = item.run_dir / "train"
    weights = train_dir / "weights" / "best.pt"
    if is_checkpoint(item.model):
        weights, status["train"] = Path(item.model), "pretrained checkpoint"
    elif (train_dir / "metrics.json").exists() and weights.exists():
        status["train"] = "skipped (done)"
    else:
        over = {**item.overrides, "model": item.model, "data": item.data}
        last = train_dir / "weights" / "last.pt"
        if last.exists():
            over = {"resume": str(last), **{k: item.overrides[k] for k in ("device", "workers")
                                           if k in item.overrides}}
            status["train"] = "resumed"
        else:
            status["train"] = "trained"
        from .trainer import Trainer
        Trainer(overrides=over, save_dir=train_dir).train()

    alias_session = None
    if item.eval_alias_reuse:
        from .eval_aliases import ExactAliasSession
        alias_session = ExactAliasSession(spec, splits)
    for split in splits:
        if not _split_applicable(spec, split, receipts):
            status["splits"][split] = "N/A (not_applicable_missing_split)"
            continue
        out = item.run_dir / split
        if (out / "metrics.json").exists():
            status["splits"][split] = "skipped (done)"
            continue
        over = {k: v for k, v in item.overrides.items()
                if k not in TRAIN_ONLY and k not in INPUT_KEYS}
        cfg = get_cfg({**over, "model": str(weights), "data": item.data, "split": split,
                       "mode": "val"})
        from .validator import Validator
        if alias_session is None:
            Validator(cfg, save_dir=out, label=item.label)()
        else:
            Validator(cfg, save_dir=out, label=item.label, alias_session=alias_session)()
        status["splits"][split] = "evaluated"
    status.update(train_summary(train_dir))
    status["test"] = any(s.startswith("test") and v == "evaluated"
                         for s, v in status["splits"].items())
    if item.metadata:
        verify_dataset_identity(item.data, item.metadata)
        admitted = item.metadata['admitted_row']
        artifacts = bind_completed_results(admitted, item.run_dir, list(splits), receipts)
        if item.transfer_origin_verification is not None:
            checkpoint_snapshot = item.transfer_origin_verification['checkpoint']
        else:
            checkpoint_snapshot = _snapshot_regular_file(
                weights, 'checkpoint', keep_bytes=False)
        row = {**admitted, 'state': 'completed', 'result': status,
               'run_identity_path': str(identity_path), 'run_identity_sha256': _sha(identity_path),
               'checkpoint': str(checkpoint_snapshot['path']),
               'checkpoint_sha256': checkpoint_snapshot['sha256'],
               'result_artifacts': artifacts}
        retain_completed_checkpoint(item.metadata['admission_path'], row, item.run_dir,
                                    json_load(item.metadata['policy_path']),
                                    coverage_rows=item.coverage_rows,
                                    transfer_origin=item.transfer_origin,
                                    transfer_origin_context=item.transfer_origin_verification)
    return status


def _log_record(event: str, payload: Mapping) -> None:
    """一行结构化进度记录（便于 ``grep`` 与自动监控）；空字段不写出。"""
    body = {k: v for k, v in payload.items() if v is not None}
    LOGGER.info(f"benchmark {event}: " + json.dumps(to_builtin(body), ensure_ascii=False,
                                                    sort_keys=False))


def run_benchmark(cfg: Union[PathLike, Mapping], overrides: Optional[dict] = None,
                  dry_run: bool = False, report: Optional[bool] = None,
                  continue_on_error: Optional[bool] = None) -> list:
    """执行完整矩阵；返回每项的状态，并写出 ``<root>/benchmark.json``。

    每个 run 的开始与结束各写一行结构化日志（``benchmark run start`` /
    ``benchmark run done``），含数据集、模型、种子、epoch 数、用时、val 选模标量与是否评测
    过 test。``continue_on_error=true``（或 YAML 同名键）时，单个 run 失败只记录进
    ``benchmark.json`` 并继续后续条目——长跑矩阵不应被一个组合拖垮；返回的状态里带
    ``error`` 字段，CLI 以退出码 1 报出。
    """
    cfg = resolve_benchmark_cfg(cfg)
    items = load_plan(cfg, overrides)
    if any(item.metadata for item in items):
        items = dependency_order(items)
    root = items[0].root
    spec = dict(yaml_load(cfg)) if not isinstance(cfg, Mapping) else dict(cfg)
    tolerate = bool(spec.get("continue_on_error", False) if continue_on_error is None
                    else continue_on_error)
    LOGGER.info(f"benchmark: {len(items)} runs → {root}")
    statuses = []
    for k, item in enumerate(items, 1):
        head = {"i": k, "n": len(items), "model": item.label, "dataset": str(item.data),
                "seed": item.seed, "dir": str(item.run_dir)}
        if dry_run:
            status = plan_item(item)
            _log_record("run planned", {**head, "epochs": status.get("epochs"),
                                        "splits": status.get("splits")})
            statuses.append(status)
            continue
        _log_record("run start", head)
        t0 = time.time()
        try:
            status = run_item(item)
        except Exception as exc:  # noqa: BLE001 - 见 docstring：长跑矩阵可选择跳过失败项
            if not tolerate:
                raise
            status = {"label": item.label, "data": str(item.data), "seed": item.seed,
                      "dir": str(item.run_dir), "train": "failed", "splits": {},
                      "error": f"{type(exc).__name__}: {exc}"}
            LOGGER.error(f"benchmark: {item.label} × {item.data} × seed {item.seed} failed: "
                         f"{type(exc).__name__}: {exc}")
        status["seconds"] = round(time.time() - t0, 1)
        statuses.append(status)
        json_save(root / "benchmark.json", {"plan": spec, "runs": statuses})
        _log_record("run done", {**head, "train": status.get("train"),
                                 "epochs": status.get("epochs"),
                                 "seconds": status["seconds"],
                                 "fitness": status.get("fitness"),
                                 "val_fitness": status.get("val_fitness"),
                                 "test": status.get("test"),
                                 "splits": status.get("splits"),
                                 "error": status.get("error")})
    do_report = spec.get("report", True) if report is None else report
    if do_report and not dry_run:
        from ..utils.reports import build_report

        build_report(root, root / "report")
    return statuses
