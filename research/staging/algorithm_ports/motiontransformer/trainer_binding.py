"""Unregistered, opt-in Motion binding for the single shared Trainer.

Only reviewed selected canonical train recordings are used. The target reader
consumes IMU and explicitly attested reference attitude; it never loads labels.
Source validation uses domain 0 and a named polar-velocity diagnostic. This is
the phase-A world-velocity/Z adaptation, not the paper body-displacement track.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from inertial_benchmark.data.dataset import InertialDataset
from inertial_benchmark.data.splits import read_split
from inertial_benchmark.data.unlabelled import (
    ReferenceAttitudeContract, SourcePartitionGuard, TargetInputConfig,
    UnlabelledInertialDataset,
)
from inertial_benchmark.data.views import ViewConfig
from inertial_benchmark.engine.protocol_binding import ProtocolTrainerBinding, ProtocolSetupContext
from inertial_benchmark.engine.training_protocol import DomainProvenance, ProtocolIdentity, StageRecipe
from inertial_benchmark.nn.base import InputSpec
from .model import MotionTransformerAdaptation, squared_distance
from .phase_a_protocol import (
    MotionPhaseAProtocol, _source_binding, _target_binding, ordered_ids_sha256,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SourcePolarEvaluation(nn.Module):
    """Same XYZ forward; loss is a source-only diagnostic, never GAN training."""

    def __init__(self, model: MotionTransformerAdaptation) -> None:
        super().__init__()
        self.model = model
        self.input_spec = model.input_spec
        self.model_cfg = model.model_cfg
        self.loss_name = "source_polar_velocity_diagnostic"
        self.loss_kwargs = {}

    def forward(self, imu: torch.Tensor) -> dict:
        if self.model.inference_domain != 0:
            raise ValueError("source fitness cannot evaluate the target domain")
        return self.model(imu)

    def loss(self, out: dict, batch: dict, epoch: int = 0) -> tuple:
        if self.model.inference_domain != 0 or "target_domain_imu" in batch:
            raise ValueError("source diagnostic is separate from joint training objectives")
        labels = self.model.source_labels(batch["target"])
        polar = torch.stack([out["speed"], out["heading"]], -1)
        if self.input_spec.dims == 3:
            polar = torch.cat([polar, out["vertical"][:, None]], -1)
        mask = batch["mask"].bool().reshape(len(labels), -1).all(-1)
        if not mask.any():
            raise ValueError("source diagnostic needs actual valid source labels")
        loss = squared_distance(polar[mask], labels[mask])
        return loss, {"source_polar_velocity_diagnostic": loss.detach()}


@dataclass(frozen=True)
class MotionTrainerInputs:
    source_root: Path
    target_root: Path
    runtime_source_root: Path
    source_manifest_sha256: str
    source_admission: Path
    source_admission_sha256: str
    target_admission: Path
    target_admission_sha256: str
    permission_proof: Path
    permission_proof_sha256: str
    permission_review: Path
    permission_review_sha256: str
    source_guard: SourcePartitionGuard
    source_ids: tuple
    target_ids: tuple
    source_val_ids: tuple
    target_group_kinds: dict
    target_batch: int = 3
    hidden: int = 128
    layers: int = 1

    def check_permission_documents(self) -> dict:
        for name in ("source_admission", "target_admission", "permission_proof", "permission_review"):
            if _sha(getattr(self, name)) != getattr(self, name + "_sha256"):
                raise ValueError(f"{name} bytes changed from the reviewed request")
        proof = json.loads(Path(self.permission_proof).read_text())
        review = json.loads(Path(self.permission_review).read_text())
        synthetic = review.get("schema") == "ipb-synthetic-engineering-permission-v1"
        if not synthetic and (review.get("schema") != "ib.root.Motion-permission-A1.actual-review.v1"
                              or review.get("proof_sha256") != self.permission_proof_sha256
                              or not str(review.get("status", "")).startswith("accepted_")):
            raise ValueError("a root-reviewed real input-byte proof is required")
        if synthetic and (proof.get("synthetic_engineering_fixture_not_formal_science") is not True
                          or review.get("synthetic_engineering_fixture_not_formal_science") is not True):
            raise ValueError("synthetic permission cannot claim real native admission")
        if (set(self.source_ids) != set(proof["source_proof"])
                or set(self.target_ids) != set(proof["target_proof"])
                or set(review["source_labelled_ids"]) != set(self.source_ids)
                or {review["target_id"]} != set(self.target_ids)):
            raise ValueError("selected train IDs exceed the actually reviewed representative proof")
        if (not self.source_ids or not self.target_ids or not self.source_val_ids
                or len(set(self.source_ids)) != len(self.source_ids)
                or len(set(self.target_ids)) != len(self.target_ids)
                or set(self.target_group_kinds) != set(self.target_ids)):
            raise ValueError("selected source/target/val IDs must be explicit and nonempty")
        return proof


class MotionTrainerFactory:
    """Build new objects after shared Trainer seeding; no registration or side files."""

    def __init__(self, inputs: MotionTrainerInputs) -> None:
        self.inputs = inputs

    def __call__(self, context: ProtocolSetupContext) -> ProtocolTrainerBinding:
        inputs, a = self.inputs, context.args
        proof = inputs.check_permission_documents()
        source_root = Path(inputs.source_root).resolve(strict=True)
        target_root = Path(inputs.target_root).resolve(strict=True)
        runtime_root = Path(inputs.runtime_source_root).resolve(strict=True)
        if (source_root != context.source_spec.root.resolve()
                or _sha(runtime_root / "source_manifest.json") != inputs.source_manifest_sha256
                or not Path(__file__).resolve().is_relative_to(runtime_root)):
            raise ValueError("factory must bind the actual same immutable code/source root")
        recipe = StageRecipe(**{f.name: getattr(a, f.name) for f in fields(StageRecipe)})
        if type(inputs.target_batch) is not int or inputs.target_batch < 1:
            raise ValueError("target batch must be an explicit positive engineering batch")
        spec = InputSpec(**{f.name: getattr(ViewConfig.from_cfg(a), f.name)
                            for f in fields(ViewConfig)})
        # Width and all scientific windows remain explicit; CPU engineering
        # batching is separate from unqualified common-B128 CUDA admission.
        model = MotionTransformerAdaptation(spec, hidden=inputs.hidden, layers=inputs.layers).to(context.device)
        model.model_cfg = {"name": "motiontransformer_phase_A_unregistered", "arch": "unregistered_motion_phase_A",
                           "input": {f.name: getattr(spec, f.name) for f in fields(ViewConfig)},
                           "args": {"hidden": inputs.hidden, "layers": inputs.layers},
                           "loss": model.default_loss, "recipes": {},
                           "adaptation": "world-average-velocity XY polar + learned Z; not original displacement"}
        source_train = tuple(read_split(source_root / "splits/train.txt"))
        if not set(inputs.source_ids) <= set(source_train):
            raise ValueError("source labels must be canonical train only")
        source_val = set(context.source_spec.sequence_ids("val"))
        if not set(inputs.source_val_ids) <= source_val:
            raise ValueError("source evaluation IDs must be actual canonical val")
        view = ViewConfig.from_cfg(a)
        source = InertialDataset([source_root / "sequences" / f"{sid}.h5" for sid in inputs.source_ids],
                                 view, stride=int(a.stride), augment=a.augment, training=True,
                                 cache=bool(a.cache), seed=int(a.seed))
        val = InertialDataset([source_root / "sequences" / f"{sid}.h5" for sid in inputs.source_val_ids],
                              view, stride=int(a.eval_stride), training=False,
                              cache=bool(a.cache), seed=int(a.seed))
        target_certificate = json.loads(Path(inputs.target_admission).read_text())
        contracts = {}
        for sid in inputs.target_ids:
            entry = target_certificate["selected_inputs"][sid]
            path = Path(entry["reference_coverage_receipt"])
            receipt = json.loads(path.read_text())
            actual = proof["target_proof"][sid]
            if (receipt["orientation_sha256"] != actual["orientation_sha256"]
                    or receipt["timestamp_sha256"] != actual["timestamp_sha256"]
                    or receipt["intervals"] != actual["coverage_intervals"]
                    or receipt["sample_count"] != actual["num_samples"]):
                raise ValueError("attitude certificate differs from actual raw-input reconstruction")
            contracts[sid] = ReferenceAttitudeContract(sid, receipt["sample_count"],
                receipt["orientation_source"], tuple(tuple(v) for v in receipt["intervals"]), _sha(path))
        target = UnlabelledInertialDataset(target_root,
            TargetInputConfig(spec.window, spec.rate, "reference_attitude_assisted", spec.remove_gravity),
            source_guard=inputs.source_guard, group_kinds=inputs.target_group_kinds,
            only=inputs.target_ids, stride=int(a.stride), attitude=contracts)
        target_manifest = json.loads((target_root / "dataset.json").read_text())
        source_provenance = DomainProvenance(context.source_spec.fingerprint(),
            _sha(source_root / "splits/train.txt"), ordered_ids_sha256(source.sequence_ids),
            inputs.source_admission_sha256, len(source))
        target_provenance = DomainProvenance(target_manifest["fingerprint"],
            _sha(target_root / "splits/train.txt"), ordered_ids_sha256(target.sequence_ids),
            inputs.target_admission_sha256, len(target))
        identity = ProtocolIdentity(protocol=MotionPhaseAProtocol.protocol_name,
            source=source_provenance, target=target_provenance,
            source_manifest_sha256=inputs.source_manifest_sha256,
            input_contract_sha256=hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode()).hexdigest(),
            experiment_seed=int(a.seed), input_privilege="reference_attitude_assisted")
        protocol = MotionPhaseAProtocol(model, source, target, identity,
            source_guard=inputs.source_guard, source_canonical_train_ids=source_train,
            source_native_root=source_root, source_admission_path=inputs.source_admission,
            target_admission_path=inputs.target_admission, runtime_source_root=runtime_root,
            source_batch=int(a.batch), target_batch=inputs.target_batch, recipe=recipe)

        def revalidate() -> None:
            inputs.check_permission_documents()
            if _sha(runtime_root / "source_manifest.json") != inputs.source_manifest_sha256:
                raise ValueError("immutable protocol source manifest changed")
            _source_binding(source, source_root, inputs.source_admission, identity.source,
                            inputs.source_guard, source_train)
            _target_binding(target, inputs.target_admission, identity.target)
            for sid in inputs.source_ids:
                row = context.source_spec.manifest()["sequences"][sid]
                if row["sha256"] != proof["source_proof"][sid]["file_sha256"]:
                    raise ValueError("source labels differ from the reviewed input proof")
            for sid in inputs.source_val_ids:
                row = context.source_spec.manifest()["sequences"][sid]
                if _sha(source_root / "sequences" / f"{sid}.h5") != row["sha256"]:
                    raise ValueError("source validation bytes changed")

        return ProtocolTrainerBinding(protocol, source_spec=context.source_spec, val_set=val,
            evaluation_model=SourcePolarEvaluation(model), revalidate=revalidate,
            permission_identity={name: getattr(inputs, name + "_sha256")
                                 for name in ("source_admission", "target_admission", "permission_proof", "permission_review")})
