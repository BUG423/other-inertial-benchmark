"""Server-only phase-A tests; synthetic protocol fixtures are not NIO admission."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from inertial_benchmark.data.format import REQUIRED_ATTRS, Sequence, save_sequence
from inertial_benchmark.data.dataset import InertialDataset
from inertial_benchmark.data.manifest import DatasetSpec, fingerprint, imu_content_hash
from torch.utils.data import default_collate
from inertial_benchmark.data.unlabelled import (
    RecordingIdentity, ReferenceAttitudeContract, SourcePartitionGuard,
    TargetAccessAudit, TargetInputConfig, UnlabelledInertialDataset,
)
from inertial_benchmark.engine.training_protocol import (
    INFERENCE_SCHEMA, DomainProvenance, ProtocolIdentity, StageRecipe,
    capture_rng, preserve_rng_and_modes, restore_rng,
)
from inertial_benchmark.nn.base import InputSpec
from research.staging.algorithm_ports.motiontransformer.model import MotionTransformerAdaptation
from research.staging.algorithm_ports.motiontransformer.phase_a_protocol import (
    IncompletePairError, MotionPhaseAProtocol, ordered_ids_sha256,
)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_native_attrs(dataset, sid, group, *, labelled):
    """Canonical v1 metadata for generated test data, not a physical recording.

    This follows tests/synthetic.v1_attrs: no absolute acquisition time or
    device attitude is invented. The target deliberately has no pose labels;
    its synthetic reference-q permission is in the separate coverage receipt.
    """
    attrs = {
        "schema_version": "1.0", "dataset": dataset, "sequence_id": sid,
        "sample_rate_hz": 200., "source_sample_rate_hz": 200.,
        "resampling": "none (synthetic)", "timestamp_type": "relative",
        "start_time_unix": float("nan"), "world_frame": "gravity_aligned_z_up",
        "orientation_convention": "body_to_world_wxyz",
        "accelerometer_type": "specific_force", "body_frame": "device",
        "position_source": "analytic_constant_velocity" if labelled else "none",
        "orientation_source": "synthetic-reference", "device_orientation_source": "none",
        "subject_id": "synthetic_no_person", "device_id": "synthetic_no_physical_device",
        "placement": "unknown", "group_id": group, "source_license": "generated",
        "source_files": ["test_phase_a_protocol.py:generated_in_memory"],
        "converter": "synthetic_motion_phase_a_fixture@1",
        "synthetic_engineering_fixture_not_formal_science": True,
    }
    assert set(REQUIRED_ATTRS) <= attrs.keys()
    return attrs


def publish_manifest(root, dataset, rows, splits):
    for split, ids in splits.items():
        (root / "splits" / (split + ".txt")).write_text("\n".join(ids) + "\n")
    payload = {"schema_version": "1.0", "dataset": dataset, "sequences": rows, "splits": splits,
               "fingerprint": fingerprint({sid: row["sha256"] for sid, row in rows.items()}, splits)}
    (root / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def publish_admission(root, metadata, selected, *, mode="reference_attitude_assisted", coverage=None, guard=None):
    path = root / "input-admission.json"
    proof = {"schema": "ipb-native-train-input-admission-v1", "status": "accepted_train_inputs",
             "native_manifest_sha256": file_sha(root / "dataset.json"),
             "dataset_fingerprint": metadata["fingerprint"],
             "canonical_train_split_sha256": file_sha(root / "splits/train.txt"),
             "input_mode": mode, "synthetic_engineering_fixture_not_formal_science": True,
             "selected_inputs": {sid: {"imu_sha256": metadata["sequences"][sid]["imu_sha256"],
                 **({"reference_coverage_receipt": str(coverage)} if coverage else {})} for sid in selected}}
    if guard is not None:
        proof["source_partition_guard_sha256"] = hashlib.sha256(json.dumps(asdict(guard), sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(proof, indent=2) + "\n")
    return path


def make_source(root, count, dims, seed):
    root.mkdir(parents=True, exist_ok=True)
    (root / "sequences").mkdir(exist_ok=True); (root / "splits").mkdir(exist_ok=True)
    if not (root / "dataset.json").exists():
        n = 400 + (count - 1) * 200
        rows = {}
        for sid, omega in (("source-train", 0.), ("source-val", .1), ("source-test", .2)):
            t = np.arange(n) / 200.
            gyro = np.full((n, 3), omega, dtype="f4")
            acc = np.tile(np.array([0, 0, 9.81], dtype="f4"), (n, 1))
            velocity = np.tile(np.array([1.1, -.3, .2], dtype="f4"), (n, 1))
            sequence = Sequence(t, gyro, acc, np.tile(np.array([1, 0, 0, 0], dtype="f4"), (n, 1)),
                t[:, None] * velocity, np.ones(n, bool), np.ones(n, bool), velocity=velocity,
                attrs=synthetic_native_attrs("source-native", sid, sid, labelled=True))
            file = root / "sequences" / (sid + ".h5"); save_sequence(file, sequence)
            rows[sid] = {"file": "sequences/" + sid + ".h5", "sha256": file_sha(file),
                         "imu_sha256": imu_content_hash(gyro, acc), "group_id": sid}
        publish_manifest(root, "source-native", rows,
            {"train": ["source-train"], "val": ["source-val"], "test": ["source-test"]})
    native = DatasetSpec("source-native", root)
    metadata = native.manifest()
    guard = SourcePartitionGuard(
        tuple(RecordingIdentity("source-native", sid, sid, "recording", imu_sha256=metadata["sequences"][sid]["imu_sha256"])
              for sid in ("source-train",)),
        tuple(RecordingIdentity("source-native", sid, sid, "recording", imu_sha256=metadata["sequences"][sid]["imu_sha256"])
              for sid in ("source-val", "source-test")))
    assert native.sequence_ids("train") == ["source-train"]
    assert native.sequence_ids("val") == ["source-val"] and native.sequence_ids("test") == ["source-test"]
    dataset = InertialDataset(native.sequence_paths("train"), InputSpec(window=400, rate=200, dims=dims),
        stride=200, augment=[], training=True, cache=True, seed=seed)
    materialized = default_collate([dataset[i] for i in range(len(dataset))])
    dataset.imu, dataset.target = materialized["imu"], materialized["target"]
    admission = publish_admission(root, metadata, dataset.sequence_ids, guard=guard)
    return dataset, guard, metadata, admission


def make_target(root, count, guard):
    root.mkdir(parents=True, exist_ok=True)
    (root / "sequences").mkdir(exist_ok=True); (root / "splits").mkdir(exist_ok=True)
    n = 400 + (count - 1) * 200
    if not (root / "dataset.json").exists():
        rng = np.random.default_rng(8801)
        gyro, acc = rng.normal(size=(n, 3)).astype("f4"), rng.normal(size=(n, 3)).astype("f4")
        file = root / "sequences/target-train.h5"
        with h5py.File(file, "w") as f:
            attrs = synthetic_native_attrs("target-native", "target-train", "target-recording", labelled=False)
            # HDF5 attrs cannot store a Python list of arbitrary Unicode strings.
            # Sequence.save_sequence uses the same JSON encoding for source_files.
            attrs["source_files"] = json.dumps(attrs["source_files"])
            f.attrs.update(attrs)
            f["timestamp"] = np.arange(n) / 200.; f["imu/gyroscope"] = gyro; f["imu/accelerometer"] = acc
            f["valid/imu"] = np.ones(n, bool)
            f["pose/orientation"] = np.tile(np.array([1, 0, 0, 0], dtype="f4"), (n, 1))
            f["pose/position"] = np.full((n, 3), np.nan); f["pose/velocity"] = np.full((n, 3), np.nan)
            f["valid/pose"] = np.zeros(n, bool)
        publish_manifest(root, "target-native", {"target-train": {"file": "sequences/target-train.h5",
            "sha256": file_sha(file), "imu_sha256": imu_content_hash(gyro, acc), "group_id": "target-recording"}},
            {"train": ["target-train"], "val": ["never-open-val"], "test": ["never-open-test"]})
    metadata = json.loads((root / "dataset.json").read_text())
    coverage = root / "reference-input-coverage.json"
    proof = {"status": "accepted_reference_attitude_coverage", "native_manifest_sha256": file_sha(root / "dataset.json"),
             "sequence_id": "target-train", "sample_count": n, "intervals": [[0, n]],
             "orientation_source": "synthetic-reference", "synthetic_engineering_fixture_not_formal_science": True,
             "orientation_sha256": hashlib.sha256(repr((n, 4)).encode() + np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (n, 1)).tobytes()).hexdigest(),
             "timestamp_sha256": hashlib.sha256(repr((n,)).encode() + np.asarray(np.arange(n) / 200., dtype="<f8").tobytes()).hexdigest()}
    coverage.write_text(json.dumps(proof, indent=2) + "\n")
    dataset = UnlabelledInertialDataset(root,
        TargetInputConfig(window=400, mode="reference_attitude_assisted"), source_guard=guard,
        group_kinds={"target-train": "recording"}, stride=200,
        attitude={"target-train": ReferenceAttitudeContract("target-train", n,
            "synthetic-reference", ((0, n),), file_sha(coverage))}, audit=TargetAccessAudit())
    admission = publish_admission(root, metadata, dataset.sequence_ids, coverage=coverage)
    return dataset, metadata, admission


def make_protocol(root, *, hidden=8, dims=3, source_count=6, target_count=6, seed=1073):
    source, guard, sm, sa = make_source(root / "source", source_count, dims, seed)
    target, tm, ta = make_target(root / "target", target_count, guard)
    spec = InputSpec(window=400, rate=200, dims=dims)
    identity = ProtocolIdentity(MotionPhaseAProtocol.protocol_name,
        DomainProvenance(sm["fingerprint"], file_sha(root / "source/splits/train.txt"),
                         ordered_ids_sha256(source.sequence_ids), file_sha(sa), len(source)),
        DomainProvenance(tm["fingerprint"], target.train_split_sha256,
                         ordered_ids_sha256(target.sequence_ids), file_sha(ta), len(target)),
        file_sha(Path(__file__).resolve().parents[4] / "source_manifest.json"),
        hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest(),
        seed, "reference_attitude_assisted")
    model = MotionTransformerAdaptation(spec, hidden=hidden)
    result = MotionPhaseAProtocol(model, source, target, identity, source_guard=guard,
        source_canonical_train_ids=("source-train",), source_native_root=root / "source",
        source_admission_path=sa, target_admission_path=ta, runtime_source_root=Path(__file__).resolve().parents[4],
        source_batch=2, target_batch=3,
        recipe=StageRecipe(lr=0.0001, epochs=2))
    result.test_admission_paths = (root / "source", sa, ta)
    return result


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b


def test_default_width_two_real_D_G_pairs_all_five_losses(tmp_path):
    protocol = make_protocol(tmp_path, hidden=128, source_count=4)
    assert protocol.model.core.hidden == 128 and protocol.model.input_spec.window == 400
    initial = {k: v.clone() for k, v in protocol.model.state_dict().items()}
    for expected in (1, 2):
        result = protocol.step()
        assert result.generator_steps == result.discriminator_steps == result.pair == expected
        assert result.source_windows == 2 and result.target_windows == 3
        assert set(result.objectives) == {"total", "discriminator", "gan", "reconstruction", "prediction", "cycle", "perceptual"}
        assert all(np.isfinite(v) for v in result.objectives.values())
    for stage, names in protocol.parameter_names.items():
        assert any(not torch.equal(initial[n], protocol.model.state_dict()[n]) for n in names)
        steps = [int(v["step"]) for v in protocol.optimizers[stage].state.values() if "step" in v]
        assert steps and min(steps) == max(steps) == 2


def test_target_never_reads_labels_posevalid_or_heldout(tmp_path, monkeypatch):
    original = h5py.Dataset.__getitem__
    reads = []
    def checked(dataset, item):
        if "target" in Path(dataset.file.filename).parts:
            reads.append(dataset.name)
        if "target" in Path(dataset.file.filename).parts:
            assert dataset.name not in ("/pose/position", "/pose/velocity", "/valid/pose")
        return original(dataset, item)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    protocol = make_protocol(tmp_path)
    protocol.step()
    assert reads and set(reads) <= {"/timestamp", "/imu/gyroscope", "/imu/accelerometer", "/valid/imu", "/pose/orientation"}
    assert all("never-open" not in event["path"] for event in protocol.target_dataset.audit.events)


def test_partial_epoch_atomic_checkpoint_exact_both_optimizers_sampler_RNG_metrics(tmp_path):
    torch.manual_seed(2201); np.random.seed(2201); random.seed(2201)
    original = make_protocol(tmp_path / "a")
    original.step(); original.step()
    original.save(tmp_path / "complete-pair.pt")
    saved = torch.load(tmp_path / "complete-pair.pt", weights_only=True)
    expected_pair = original.step()
    expected = original.state_dict()
    # Construction deliberately consumes RNG; restore must be the last operation.
    restored = make_protocol(tmp_path / "a")
    restored.load_state_dict(saved)
    actual_pair = restored.step()
    assert expected_pair == actual_pair
    same(expected, restored.state_dict())


def test_target_cycles_independently_commits_source_once(tmp_path):
    protocol = make_protocol(tmp_path, target_count=3)
    for _ in range(3): protocol.step()
    assert protocol.stream.source.exhausted and protocol.stream.source.cycle == 0
    assert protocol.stream.target.cycle == 2
    assert protocol.aggregates["source_windows"] == 6 and protocol.aggregates["target_windows"] == 9
    assert protocol.step() is None


def test_both_schedulers_completed_epoch_resume_exact_without_double_advance(tmp_path):
    protocol = make_protocol(tmp_path / "a", source_count=2)
    protocol.step(); protocol.complete_epoch(); saved = protocol.state_dict()
    expected_lrs = {k: v.get_last_lr() for k, v in protocol.schedulers.items()}
    restored = make_protocol(tmp_path / "a", source_count=2); restored.load_state_dict(saved)
    assert expected_lrs == {k: v.get_last_lr() for k, v in restored.schedulers.items()}
    with pytest.raises(ValueError): restored.complete_epoch()
    restored.start_epoch()
    assert restored.stream.source.epoch == restored.stream.target.epoch == 1
    assert restored.source_dataset.epoch == 1


def test_D_only_failure_poison_blocks_every_api_and_recovers_from_last_pair(tmp_path, monkeypatch):
    protocol = make_protocol(tmp_path)
    saved = protocol.state_dict()
    real = protocol.model.core.generator_objective
    monkeypatch.setattr(protocol.model.core, "generator_objective", lambda *args: (_ for _ in ()).throw(RuntimeError("after D")))
    with pytest.raises(IncompletePairError): protocol.step()
    assert protocol.stream.pairs == protocol.stream.source.cursor == protocol.stream.target.cursor == 0
    assert all(p.requires_grad for p in protocol.model.core.discriminator_parameters())
    for call in (protocol.step, protocol.state_dict, protocol.inference, protocol.complete_epoch,
                 lambda: protocol.evaluation_loss(torch.zeros(2, 6, 400), torch.zeros(2, 3))):
        with pytest.raises(IncompletePairError): call()
    monkeypatch.setattr(protocol.model.core, "generator_objective", real)
    protocol.load_state_dict(saved)
    assert protocol.step().pair == 1


@pytest.mark.parametrize("field", ["target", "recipe", "source"])
def test_changed_domain_or_recipe_checkpoint_refuses_before_weights_mutate(tmp_path, field):
    protocol = make_protocol(tmp_path)
    saved = protocol.state_dict(); before = copy.deepcopy(protocol.model.state_dict())
    if field == "recipe": saved["recipe"]["lr"] *= 2
    else:
        saved["identity"][field]["dataset_fingerprint"] = "9" * 64
        changed = replace(protocol.identity, **{field: replace(getattr(protocol.identity, field), dataset_fingerprint="9" * 64)})
        sr, sa, ta = protocol.test_admission_paths
        source_meta = json.loads((sr / "dataset.json").read_text())
        guard = SourcePartitionGuard(
            tuple(RecordingIdentity("source-native", sid, sid, "recording", imu_sha256=source_meta["sequences"][sid]["imu_sha256"])
                  for sid in ("source-train",)),
            tuple(RecordingIdentity("source-native", sid, sid, "recording", imu_sha256=source_meta["sequences"][sid]["imu_sha256"])
                  for sid in ("source-val", "source-test")))
        with pytest.raises(ValueError, match="fingerprint"):
            MotionPhaseAProtocol(protocol.model, protocol.source_dataset, protocol.target_dataset,
                changed, source_guard=guard, source_canonical_train_ids=("source-train",),
                source_native_root=sr, source_admission_path=sa, target_admission_path=ta,
                runtime_source_root=Path(__file__).resolve().parents[4], source_batch=2, target_batch=3)
    with pytest.raises(ValueError): protocol.load_state_dict(saved)
    same(before, protocol.model.state_dict())


def test_native_inference_state_output_rng_equal_and_no_training_or_target_data(tmp_path):
    protocol = make_protocol(tmp_path); protocol.step()
    before_rng = capture_rng(); bundle = protocol.inference(0).payload()
    same(before_rng, capture_rng())
    assert bundle["schema"] == INFERENCE_SCHEMA and bundle["inference_export"]
    assert not set(bundle) & {"optimizers", "schedulers", "stream", "rng", "target_data", "data", "aggregates"}
    binary = io.BytesIO(); torch.save(bundle, binary); binary.seek(0)
    restored_bundle = torch.load(binary, weights_only=True)
    spec = InputSpec.from_dict(restored_bundle["input_spec"])
    restored = MotionTransformerAdaptation(spec, hidden=restored_bundle["architecture"]["hidden"],
        layers=restored_bundle["architecture"]["layers"], inference_domain=restored_bundle["inference_domain"])
    restored.load_state_dict(restored_bundle["model"], strict=True)
    x = protocol.source_dataset.imu[:2]
    protocol.model.inference_domain = 0
    with torch.no_grad(): same(protocol.model(x), restored(x))
    with pytest.raises(ValueError): protocol.load_state_dict(restored_bundle)


def test_scoped_source_evaluation_has_no_target_read_and_preserves_modes_noise(tmp_path):
    protocol = make_protocol(tmp_path); protocol.step()
    events = len(protocol.target_dataset.audit.events); rng = capture_rng()
    protocol.model.train(); protocol.model.core.decoder.eval()
    modes = [m.training for m in protocol.model.modules()]
    result = protocol.evaluation_loss(protocol.source_dataset.imu[:2], protocol.source_dataset.target[:2])
    assert set(result) == {"source_polar_velocity_diagnostic"} and np.isfinite(next(iter(result.values())))
    same(rng, capture_rng()); assert modes == [m.training for m in protocol.model.modules()]
    assert len(protocol.target_dataset.audit.events) == events


@pytest.mark.parametrize("kwargs", [{"amp": True}, {"workers": 2}])
def test_unqualified_AMP_or_prefetch_rejected(kwargs):
    with pytest.raises(ValueError): StageRecipe(**kwargs)


def test_native_payload_rejects_raw_target_data_and_half_pair_evidence(tmp_path):
    protocol = make_protocol(tmp_path)
    native = protocol.inference()
    with pytest.raises(ValueError): replace(native, architecture={"target_test": torch.zeros(5)}).payload()
    with pytest.raises(ValueError): replace(native, input_spec={"target_data": "/test"}).payload()
    with pytest.raises(ValueError): replace(native, training_evidence={"completed_pairs": 1, "generator_steps": 0, "discriminator_steps": 1}).payload()


def test_complete_source_canonical_and_heldout_guard_is_required(tmp_path):
    protocol = make_protocol(tmp_path)
    guard = SourcePartitionGuard(
        (RecordingIdentity("source-native", "another-train", "source-group", "recording", imu_sha256="1" * 64),),
        (RecordingIdentity("source-native", "source-val", "held-out", "recording", imu_sha256="2" * 64),))
    with pytest.raises(ValueError, match="canonical"):
        MotionPhaseAProtocol(protocol.model, protocol.source_dataset, protocol.target_dataset,
            protocol.identity, source_guard=guard, source_canonical_train_ids=("source-train",),
            source_native_root=protocol.test_admission_paths[0], source_admission_path=protocol.test_admission_paths[1],
            target_admission_path=protocol.test_admission_paths[2],
            runtime_source_root=Path(__file__).resolve().parents[4], source_batch=2)


def test_target_label_key_injection_is_rejected_and_poisoned(tmp_path, monkeypatch):
    protocol = make_protocol(tmp_path)
    original = UnlabelledInertialDataset.__getitem__
    def injected(dataset, index):
        return {**original(dataset, index), "target": np.zeros(3)}
    monkeypatch.setattr(UnlabelledInertialDataset, "__getitem__", injected)
    with pytest.raises(IncompletePairError): protocol.step()
    assert protocol.stream.pairs == 0
    with pytest.raises(IncompletePairError): protocol.state_dict()


def test_source_validation_only_and_no_partial_epoch_scheduler(tmp_path):
    protocol = make_protocol(tmp_path)
    with pytest.raises(ValueError): protocol.complete_epoch()
    for _ in range(3): protocol.step()
    with pytest.raises(ValueError, match="source"): protocol.complete_epoch(1., validation_domain="target")


def test_CPU_rng_roundtrip_never_initializes_CUDA(monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: (_ for _ in ()).throw(AssertionError("CPU initialized CUDA")))
    state = capture_rng()
    expected = (torch.randn(3), np.random.normal(size=3), random.random())
    restore_rng(state)
    actual = (torch.randn(3), np.random.normal(size=3), random.random())
    assert torch.equal(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1]); assert expected[2] == actual[2]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("dims", [2, 3])
def test_true_source_polar_labels_retains_XYZ_vertical_and_exact_XY_slice(tmp_path, dims):
    protocol = make_protocol(tmp_path, dims=dims)
    result = protocol.step()
    assert result.generator_steps == result.discriminator_steps == 1
    labels = protocol.model.source_labels(protocol.source_dataset.target[:2])
    if dims == 3:
        assert torch.equal(labels[:, 2], protocol.source_dataset.target[:2, 2])
        assert protocol.model.core.predictor.readout.weight.grad[2].abs().sum() > 0
    else:
        assert labels.shape == (2, 2)


def test_changed_target_reference_q_with_unchanged_IMU_rejected(tmp_path):
    protocol = make_protocol(tmp_path)
    target = protocol.target_dataset
    with h5py.File(target.paths[0], "r+") as handle:
        handle["pose/orientation"][:] = np.tile(np.array([0, 0, 0, 1], dtype="f4"), (len(handle["timestamp"]), 1))
    from research.staging.algorithm_ports.motiontransformer.phase_a_protocol import _target_binding
    with pytest.raises(ValueError, match="reference-attitude"):
        _target_binding(target, protocol.test_admission_paths[2], protocol.identity.target)
