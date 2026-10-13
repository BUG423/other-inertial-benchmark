"""Unregistered Motion phase-A protocol for a future hook in the single Trainer.

400 native samples at 200 Hz become 200 pair-averaged samples at 100 Hz.
Labels are adapted world-velocity polar components, not original displacement.
This leaf does not orchestrate training, register a model, or alter old files.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np
import h5py
import hashlib
import json
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import default_collate

from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.splits import read_split
from inertial_benchmark.data.samplers import CommittedBatchSampler, UnpairedBatchStream
from inertial_benchmark.data.unlabelled import SourcePartitionGuard, UnlabelledInertialDataset
from inertial_benchmark.engine.training_protocol import (
    TRAINING_SCHEMA, NativeProtocolInference, PairResult, ProtocolIdentity,
    StageRecipe, TrainingProtocol, atomic_save, capture_rng, immutable_copy,
    preserve_rng_and_modes, restore_rng,
)
from inertial_benchmark.utils.torch_utils import build_optimizer, build_scheduler
from .model import PAPER_LOSS_WEIGHTS, MotionTransformerAdaptation, frozen_discriminators, squared_distance


def ordered_ids_sha256(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_admission(root, admission_path, provenance):
    """Bind operator-admitted metadata before training; no numeric target labels.

    The accepted target reader still opens only its own input allowlist. This
    separate binder consumes immutable admission/manifest metadata, not QC loss
    values. Reference coverage remains an explicit upstream scientific admission.
    """
    root = Path(root).resolve(strict=True)
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    admission = json.loads(Path(admission_path).read_text())
    allowed = {"schema", "status", "native_manifest_sha256", "dataset_fingerprint",
               "canonical_train_split_sha256", "input_mode", "selected_inputs",
               "synthetic_engineering_fixture_not_formal_science", "source_partition_guard_sha256"}
    if set(admission) - allowed or any(set(v) - {"imu_sha256", "reference_coverage_receipt"}
                                      for v in admission.get("selected_inputs", {}).values()):
        raise ValueError("input admission must contain provenance only, never target labels/QC values")
    if (admission.get("schema") != "ipb-native-train-input-admission-v1"
            or admission.get("status") != "accepted_train_inputs"
            or admission.get("native_manifest_sha256") != _sha(manifest_path)
            or admission.get("dataset_fingerprint") != manifest["fingerprint"]
            or provenance.dataset_fingerprint != manifest["fingerprint"]
            or provenance.canonical_train_split_sha256 != _sha(root / "splits/train.txt")
            or admission.get("canonical_train_split_sha256") != provenance.canonical_train_split_sha256
            or provenance.input_admission_sha256 != _sha(admission_path)):
        raise ValueError("domain fingerprint/split/input certificate differs from actual native metadata")
    return root, manifest, admission


def _source_binding(dataset, root, admission_path, provenance, guard, canonical_ids):
    root, manifest, admission = _native_admission(root, admission_path, provenance)
    guard_sha = hashlib.sha256(json.dumps(asdict(guard), sort_keys=True).encode()).hexdigest()
    if admission.get("source_partition_guard_sha256") != guard_sha:
        raise ValueError("input certificate must bind this full source hash/group permission guard")
    if set(admission["selected_inputs"]) != set(dataset.sequence_ids):
        raise ValueError("source input certificate selected IDs differ from actual dataset")
    actual_train = tuple(read_split(root / "splits/train.txt"))
    held_out = set(read_split(root / "splits/val.txt"))
    for split in sorted((root / "splits").glob("test*.txt")):
        held_out.update(read_split(split))
    if (tuple(canonical_ids) != actual_train or tuple(r.sequence_id for r in guard.train) != actual_train
            or {r.sequence_id for r in guard.held_out} != held_out):
        raise ValueError("complete guard must bind actual canonical source train/val/test IDs")
    for recording in (*guard.train, *guard.held_out):
        row = manifest["sequences"][recording.sequence_id]
        if (recording.dataset != manifest["dataset"] or recording.imu_sha256 != row["imu_sha256"]
                or recording.group_id != row["group_id"]):
            raise ValueError("source guard IMU hash/group belongs to another native recording")
    for k, sid in enumerate(dataset.sequence_ids):
        row = manifest["sequences"][sid]
        path = (root / row.get("file", f"sequences/{sid}.h5")).resolve(strict=True)
        if not path.is_relative_to(root) or _sha(path) != row["sha256"]:
            raise ValueError("source labelled data path/bytes changed")
        observed, canonical = dataset.sequence(k), load_sequence(path)
        for name in ("timestamp", "gyroscope", "accelerometer", "position", "orientation", "velocity",
                     "valid_imu", "valid_pose", "device_orientation", "valid_device_orientation"):
            x, y = getattr(observed, name), getattr(canonical, name)
            if (x is None) != (y is None):
                raise ValueError("source labels/input differ from canonical file")
            if x is not None:
                np.testing.assert_array_equal(x, y, err_msg="source cached arrays differ from canonical file")
        if admission["selected_inputs"][sid]["imu_sha256"] != row["imu_sha256"]:
            raise ValueError("source input certificate uses another recording")
    return admission



def _allowed_input_digest(dataset, path, field, dtype):
    """Hash only granted input content through the accepted reader allowlist."""
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        shape = handle[field].shape
        digest.update(repr(shape).encode())
        for start in range(0, shape[0], 65536):
            values = np.ascontiguousarray(dataset._read(handle, path, field, slice(start, min(start + 65536, shape[0]))), dtype=dtype)
            digest.update(values.tobytes())
    return digest.hexdigest()

def _target_binding(dataset, admission_path, provenance):
    root, manifest, admission = _native_admission(dataset.root, admission_path, provenance)
    if set(admission["selected_inputs"]) != set(dataset.sequence_ids):
        raise ValueError("target input certificate selected IDs differ from actual reader")
    if admission.get("input_mode") != dataset.view.mode:
        raise ValueError("target privilege certificate differs from reader mode")
    for path, recording in zip(dataset.paths, dataset.identities):
        sid = recording.sequence_id
        row, entry = manifest["sequences"][sid], admission["selected_inputs"][sid]
        canonical = (root / row.get("file", f"sequences/{sid}.h5")).resolve(strict=True)
        if path.resolve() != canonical or not canonical.is_relative_to(root):
            raise ValueError("target reader resolves another raw recording")
        if recording.imu_sha256 != row["imu_sha256"] or entry["imu_sha256"] != recording.imu_sha256:
            raise ValueError("target actual scanned IMU differs from manifest/admission")
        contract = dataset._attitude[sid]
        receipt = json.loads(Path(entry["reference_coverage_receipt"]).read_text())
        if set(receipt) - {"status", "native_manifest_sha256", "sequence_id", "sample_count",
                           "intervals", "orientation_source", "orientation_sha256", "timestamp_sha256",
                           "synthetic_engineering_fixture_not_formal_science"}:
            raise ValueError("reference coverage certificate cannot carry numeric target labels")
        if (contract.evidence_sha256 != _sha(entry["reference_coverage_receipt"])
                or receipt.get("status") != "accepted_reference_attitude_coverage"
                or receipt.get("native_manifest_sha256") != _sha(root / "dataset.json")
                or receipt.get("sequence_id") != sid or receipt.get("sample_count") != contract.sample_count
                or tuple(tuple(v) for v in receipt.get("intervals", [])) != contract.intervals
                or receipt.get("orientation_source") != contract.orientation_source
                or receipt.get("orientation_sha256") != _allowed_input_digest(dataset, path, "pose/orientation", "<f4")
                or receipt.get("timestamp_sha256") != _allowed_input_digest(dataset, path, "timestamp", "<f8")):
            raise ValueError("actual reference-attitude coverage certificate does not match reader")
    return admission


class IncompletePairError(RuntimeError):
    """Weights may contain a D-only update; restore the last complete checkpoint."""


class MotionPhaseAProtocol(TrainingProtocol):
    protocol_name = "motiontransformer-world-velocity-unpaired-phase-A-v1"

    def __init__(self, model: MotionTransformerAdaptation, source_dataset,
                 target_dataset: UnlabelledInertialDataset, identity: ProtocolIdentity,
                 *, source_guard: SourcePartitionGuard, source_canonical_train_ids: tuple,
                 source_native_root: Path, source_admission_path: Path, target_admission_path: Path,
                 runtime_source_root: Path,
                 source_batch: int, target_batch: Optional[int] = None,
                 recipe: StageRecipe = StageRecipe()) -> None:
        if not isinstance(model, MotionTransformerAdaptation):
            raise TypeError("phase-A needs the reviewed unregistered Motion adaptation")
        runtime_source_root = Path(runtime_source_root).resolve(strict=True)
        if (not Path(__file__).resolve().is_relative_to(runtime_source_root)
                or identity.source_manifest_sha256 != _sha(runtime_source_root / "source_manifest.json")):
            raise ValueError("protocol source identity must bind its actual immutable code manifest")
        if identity.protocol != self.protocol_name:
            raise ValueError("protocol identity does not describe phase-A adaptation")
        if not isinstance(target_dataset, UnlabelledInertialDataset):
            raise TypeError("target must use the accepted input-only canonical-train reader")
        if getattr(source_dataset, "training", False) is not True:
            raise ValueError("source labels must come from an admitted training dataset")
        if (not isinstance(source_guard, SourcePartitionGuard)
                or tuple(item.sequence_id for item in source_guard.train) != tuple(source_canonical_train_ids)
                or not set(source_dataset.sequence_ids) <= set(source_canonical_train_ids)):
            raise ValueError("complete canonical source-train and held-out hash/group guard required")
        _source_binding(source_dataset, source_native_root, source_admission_path,
                        identity.source, source_guard, source_canonical_train_ids)
        _target_binding(target_dataset, target_admission_path, identity.target)
        for recording in target_dataset.identities:
            source_guard.check(recording)  # Revalidate with this exact run's full source guard.
        spec = model.input_spec
        if hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest() != identity.input_contract_sha256:
            raise ValueError("declared input contract differs from actual model InputSpec")
        source_view = getattr(source_dataset, "view", None)
        if source_view is None or any(getattr(source_view, name) != getattr(spec, name)
                for name in ("window", "rate", "frame", "orientation", "remove_gravity", "target", "dims", "output_steps", "overlap", "history", "history_stride", "extra_inputs")):
            raise ValueError("source task view must match the actual phase-A InputSpec")
        if (spec.window != 400 or spec.rate != 200 or model.decimation != 2
                or target_dataset.view.window != spec.window or target_dataset.view.rate != spec.rate
                or target_dataset.view.mode != "reference_attitude_assisted"
                or identity.input_privilege != target_dataset.view.mode
                or target_dataset.view.remove_gravity != spec.remove_gravity):
            raise ValueError("phase-A domains need matching admitted gravity-world 400@200 input")
        if (len(source_dataset) != identity.source.window_count
                or len(target_dataset) != identity.target.window_count
                or target_dataset.train_split_sha256 != identity.target.canonical_train_split_sha256
                or ordered_ids_sha256(target_dataset.sequence_ids) != identity.target.selected_train_ids_sha256
                or ordered_ids_sha256(source_dataset.sequence_ids) != identity.source.selected_train_ids_sha256):
            raise ValueError("admitted domain counts/split/ordered-ID identity differs")
        self.model, self.source_dataset, self.target_dataset = model, source_dataset, target_dataset
        self.identity, self.recipe = identity, recipe
        target_batch = source_batch if target_batch is None else target_batch
        self.stream = UnpairedBatchStream(
            CommittedBatchSampler(len(source_dataset), source_batch, seed=identity.experiment_seed,
                                  stream="source", workers=recipe.workers),
            CommittedBatchSampler(len(target_dataset), target_batch, seed=identity.experiment_seed,
                                  stream="target", workers=recipe.workers))
        self.source_dataset.set_epoch(0)
        modules = {
            "generator": nn.ModuleList([model.core.encoder, model.core.generators,
                                         model.core.decoder, model.core.predictor]),
            "discriminator": model.core.discriminators,
        }
        if any(not p.requires_grad for module in modules.values() for p in module.parameters()):
            raise ValueError("every scientific generator/discriminator parameter must remain trainable")
        names = {id(p): name for name, p in model.named_parameters()}
        self.parameter_names = {stage: tuple(names[id(p)] for p in module.parameters())
                                for stage, module in modules.items()}
        g, d = (set(self.parameter_names[key]) for key in ("generator", "discriminator"))
        if not g or not d or g & d or g | d != set(names.values()):
            raise ValueError("G/D ownership must be disjoint and complete")
        self.stage_parameters = {stage: tuple(module.parameters()) for stage, module in modules.items()}
        self.optimizers = {stage: build_optimizer(module, recipe.optimizer, recipe.lr,
            recipe.momentum, recipe.weight_decay) for stage, module in modules.items()}
        for stage, optimizer in self.optimizers.items():
            owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if owned != {id(p) for p in self.stage_parameters[stage]}:
                raise ValueError("optimizer does not own the entire declared stage partition")
        self.schedulers = {stage: build_scheduler(opt, recipe) for stage, opt in self.optimizers.items()}
        self._poisoned = False
        self._epoch_complete = False
        self.aggregates = {"pairs": 0, "source_windows": 0, "target_windows": 0,
                           "weighted_objective_sums": {}}

    def _healthy(self) -> None:
        if self._poisoned:
            raise IncompletePairError("incomplete D/G pair: restore the last atomic committed checkpoint")

    def _clip(self, stage: str) -> None:
        maximum = self.recipe.grad_clip if self.recipe.grad_clip > 0 else math.inf
        torch.nn.utils.clip_grad_norm_(self.stage_parameters[stage], maximum, error_if_nonfinite=True)

    def _batches(self, pair):
        source = default_collate([self.source_dataset[i] for i in pair.source.indices])
        target = default_collate([self.target_dataset[i] for i in pair.target.indices])
        if set(target) - {"imu", "timestamp", "input_mask", "seq", "start"}:
            raise ValueError("target training batch cannot contain labels/pose/velocity")
        if (not source["mask"].bool().all() or not target["input_mask"].bool().all()
                or source.get("extra")):
            raise ValueError("only valid six-axis/no-extra source and target windows are admitted")
        device = next(self.model.parameters()).device
        x = source["imu"].to(device=device, dtype=torch.float32)
        y = source["target"].to(device=device, dtype=torch.float32)
        z = target["imu"].to(device=device, dtype=torch.float32)
        if not torch.isfinite(y).all():
            raise ValueError("source labels must be finite")
        return self.model.prepare(x), self.model.source_labels(y), self.model.prepare(z)

    def step(self) -> Optional[PairResult]:
        self._healthy()
        if self._epoch_complete:
            raise ValueError("advance the completed epoch explicitly before another pair")
        pair = self.stream.reserve_pair()
        if pair is None:
            return None
        # Once a pair starts, any error poisons this transaction. No expensive
        # per-batch model/optimizer copy is needed; restore the last committed file.
        self._poisoned = True
        try:
            source, labels, target = self._batches(pair)
            self.model.train()
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            d_loss = self.model.core.discriminator_loss(source, target)
            if not torch.isfinite(d_loss):
                raise FloatingPointError("nonfinite discriminator objective")
            d_loss.backward()
            self._clip("discriminator")
            self.optimizers["discriminator"].step()
            self.optimizers["discriminator"].zero_grad(set_to_none=True)
            with frozen_discriminators(self.model.core):
                total, terms = self.model.core.generator_objective(source, labels, target)
                if not torch.isfinite(total) or not all(torch.isfinite(v) for v in terms.values()):
                    raise FloatingPointError("nonfinite joint five-objective generator loss")
                total.backward()
            self._clip("generator")
            self.optimizers["generator"].step()
            objectives = {"discriminator": float(d_loss.detach()), "total": float(total.detach()),
                          **{key: float(v.detach()) for key, v in terms.items()}}
            ns, nt = len(pair.source.indices), len(pair.target.indices)
            aggregate = immutable_copy(self.aggregates)
            aggregate["pairs"] += 1
            aggregate["source_windows"] += ns
            aggregate["target_windows"] += nt
            sums = aggregate["weighted_objective_sums"]
            for key, value in objectives.items():
                sums[key] = sums.get(key, 0.) + value * ns
            g, d = self.stream.generator_steps + 1, self.stream.discriminator_steps + 1
            self.stream.commit_pair(pair, generator_steps=g, discriminator_steps=d)
            self.aggregates = aggregate
            self._poisoned = False
        except BaseException as error:
            raise IncompletePairError("pair did not commit; restore the last complete checkpoint") from error
        return PairResult(self.stream.pairs, ns, nt, g, d, objectives)

    def complete_epoch(self, validation_metric: Optional[float] = None,
                       *, validation_domain: str = "source") -> None:
        self._healthy()
        if not self.stream.source.exhausted or self.stream._pending is not None or self._epoch_complete:
            raise ValueError("epoch completion requires a fully committed source pass exactly once")
        if validation_domain != "source":
            raise ValueError("only source validation selects training scheduler/best")
        if self.recipe.scheduler == "plateau" and (
                validation_metric is None or not math.isfinite(validation_metric)):
            raise ValueError("plateau needs a finite source-val metric")
        self._poisoned = True
        try:
            for scheduler in self.schedulers.values():
                scheduler.step(validation_metric) if self.recipe.scheduler == "plateau" else scheduler.step()
            self._epoch_complete = True
            self._poisoned = False
        except BaseException as error:
            raise IncompletePairError("scheduler stages did not finish atomically; restore checkpoint") from error

    def start_epoch(self) -> None:
        self._healthy()
        if not self._epoch_complete:
            raise ValueError("complete the current epoch before advancing")
        self._poisoned = True
        try:
            self.stream.start_epoch(self.stream.source.epoch + 1)
            self.source_dataset.set_epoch(self.stream.source.epoch)
            self._epoch_complete = False
            self.aggregates = {"pairs": 0, "source_windows": 0, "target_windows": 0,
                               "weighted_objective_sums": {}}
            self._poisoned = False
        except BaseException as error:
            raise IncompletePairError("epoch advancement interrupted; restore checkpoint") from error

    def state_dict(self) -> dict:
        self._healthy()
        if any(not p.requires_grad for parameters in self.stage_parameters.values() for p in parameters):
            raise IncompletePairError("temporary discriminator freezing cannot be checkpointed")
        if self.stream._pending is not None:
            raise IncompletePairError("only complete-pair boundaries may be serialized")
        return immutable_copy({"schema": TRAINING_SCHEMA, "identity": asdict(self.identity),
            "identity_sha256": self.identity.sha256, "recipe": asdict(self.recipe),
            "parameter_names": self.parameter_names, "model": self.model.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in self.schedulers.items()},
            "stream": self.stream.state_dict(), "epoch_complete": self._epoch_complete,
            "inference_domain": self.model.inference_domain,
            "module_modes": {n: m.training for n, m in self.model.named_modules()},
            "parameter_requires_grad": {n: p.requires_grad for n, p in self.model.named_parameters()},
            "aggregates": self.aggregates, "rng": capture_rng()})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (state.get("schema") != TRAINING_SCHEMA or state.get("inference_export")
                or state.get("identity") != asdict(self.identity)
                or state.get("identity_sha256") != self.identity.sha256
                or state.get("recipe") != asdict(self.recipe)
                or state.get("parameter_names") != self.parameter_names):
            raise ValueError("training checkpoint source/domain/recipe/ownership identity mismatch")
        if (set(state.get("optimizers", {})) != set(self.optimizers)
                or set(state.get("schedulers", {})) != set(self.schedulers)
                or type(state.get("epoch_complete")) is not bool):
            raise ValueError("checkpoint must contain both stages and epoch completion")
        modules, parameters = dict(self.model.named_modules()), dict(self.model.named_parameters())
        modes, gradients = state.get("module_modes", {}), state.get("parameter_requires_grad", {})
        if (set(modes) != set(modules) or set(gradients) != set(parameters)
                or any(type(v) is not bool for v in (*modes.values(), *gradients.values()))
                or not all(gradients.values())
                or state.get("inference_domain") not in (0, 1)):
            raise ValueError("checkpoint module/domain mode contract is incomplete")
        aggregates, stream = state["aggregates"], state["stream"]
        source, target = stream["source"], stream["target"]
        if (aggregates["source_windows"] != source["cursor"]
                or aggregates["target_windows"] != target["cycle"] * self.stream.target.limit + target["cursor"]
                or aggregates["pairs"] != math.ceil(source["cursor"] / self.stream.source.batch_size)
                or any(not math.isfinite(v) for v in aggregates["weighted_objective_sums"].values())):
            raise ValueError("partial epoch aggregates differ from committed sampler exposure")
        self._poisoned = True
        self.stream.load_state_dict(state["stream"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.inference_domain = state["inference_domain"]
        for name, mode in modes.items():
            modules[name].training = mode
        for name, flag in gradients.items():
            parameters[name].requires_grad_(flag)
        for stage in self.optimizers:
            self.optimizers[stage].load_state_dict(state["optimizers"][stage])
            self.schedulers[stage].load_state_dict(state["schedulers"][stage])
        self._epoch_complete = state["epoch_complete"]
        if self._epoch_complete and not self.stream.source.exhausted:
            raise ValueError("completed checkpoint epoch has unconsumed source windows")
        self.aggregates = immutable_copy(state["aggregates"])
        self.source_dataset.set_epoch(self.stream.source.epoch)
        restore_rng(state["rng"])  # LAST: construction/loading cannot shift next latent draw.
        self._poisoned = False

    def save(self, path) -> None:
        atomic_save(path, self.state_dict())

    def evaluation_loss(self, imu: torch.Tensor, source_target: torch.Tensor) -> dict:
        self._healthy()
        with preserve_rng_and_modes(self.model), torch.no_grad():
            self.model.eval()
            source = self.model.prepare(imu)
            labels = self.model.source_labels(source_target)
            polar = self.model.core.predict(source, 0)
            return {"source_polar_velocity_diagnostic": float(squared_distance(polar, labels))}

    def inference(self, domain: int = 1) -> NativeProtocolInference:
        self._healthy()
        if self.stream._pending is not None:
            raise IncompletePairError("cannot export an uncommitted pair")
        return NativeProtocolInference(self.identity, immutable_copy(self.model.state_dict()),
            self.model.input_spec.to_dict(), {"type": "motiontransformer_phase_A_staging",
            "hidden": self.model.core.hidden, "layers": self.model.core.layers,
            "decimation": self.model.decimation, "label_semantics": "adapted_world_velocity_polar",
            "loss_name": self.model.default_loss, "gan_printed_minimax": True,
            **{name: value for name, value in zip(("w_reconstruction", "w_prediction", "w_cycle", "w_perceptual"), PAPER_LOSS_WEIGHTS)}},
            domain, {"completed_pairs": self.stream.pairs,
                     "generator_steps": self.stream.generator_steps,
                     "discriminator_steps": self.stream.discriminator_steps})
