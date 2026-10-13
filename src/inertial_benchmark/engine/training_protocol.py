"""Typed opt-in training protocol contracts; not wired into Trainer yet.

A protocol performs one complete update transaction, never its own epoch loop.
The future Trainer hook retains ownership of validation, callbacks and logging.
Training checkpoint payloads and inference-only payloads are distinct types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Mapping, Optional

import numpy as np
import torch


TRAINING_SCHEMA = "ipb-complete-pair-training-v1"
INFERENCE_SCHEMA = "ipb-protocol-inference-v1"


def immutable_copy(value: Any) -> Any:
    """Snapshot CPU tensors/basic types without retaining live optimizer aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: immutable_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(immutable_copy(v) for v in value)
    if isinstance(value, list):
        return [immutable_copy(v) for v in value]
    return copy.deepcopy(value)


def capture_rng() -> dict:
    """Do not initialize CUDA in a CPU protocol; NumPy state is weights-only safe."""
    name, keys, cursor, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "name": name, "keys": torch.tensor(keys.astype(np.int64)),
        "cursor": cursor, "has_gauss": has_gauss, "cached": cached},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [v.clone() for v in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else None}


def restore_rng(state: Mapping[str, Any]) -> None:
    cuda = state["torch_cuda"]
    if cuda is not None and not torch.cuda.is_initialized():
        raise ValueError("restore CUDA protocol on its initialized training device first")
    numpy = state["numpy"]
    np.random.set_state((numpy["name"], numpy["keys"].cpu().numpy().astype(np.uint32),
                         numpy["cursor"], numpy["has_gauss"], numpy["cached"]))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


@contextmanager
def preserve_rng_and_modes(model: torch.nn.Module):
    """Scoped diagnostics/export cannot shift latent draws or module mode flags."""
    state = capture_rng()
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(state)


@dataclass(frozen=True)
class DomainProvenance:
    """Admitted train-only provenance, without raw values or filesystem paths."""

    dataset_fingerprint: str
    canonical_train_split_sha256: str
    selected_train_ids_sha256: str
    input_admission_sha256: str
    window_count: int

    def __post_init__(self) -> None:
        for key in ("dataset_fingerprint", "canonical_train_split_sha256",
                    "selected_train_ids_sha256", "input_admission_sha256"):
            if not isinstance(getattr(self, key), str) or not re.fullmatch("[0-9a-f]{64}", getattr(self, key)):
                raise ValueError(f"{key} requires an admitted SHA256")
        if isinstance(self.window_count, bool) or not isinstance(self.window_count, int) or self.window_count < 1:
            raise ValueError("domain needs a positive admitted train-window count")


@dataclass(frozen=True)
class ProtocolIdentity:
    protocol: str
    source: DomainProvenance
    target: DomainProvenance
    source_manifest_sha256: str
    input_contract_sha256: str
    experiment_seed: int
    input_privilege: str

    def __post_init__(self) -> None:
        if not self.protocol or self.source.dataset_fingerprint == self.target.dataset_fingerprint:
            raise ValueError("two admitted distinct training-domain identities are required")
        for key in ("source_manifest_sha256", "input_contract_sha256"):
            if not re.fullmatch("[0-9a-f]{64}", getattr(self, key)):
                raise ValueError(f"{key} requires SHA256")
        if isinstance(self.experiment_seed, bool) or not isinstance(self.experiment_seed, int) or self.experiment_seed < 0:
            raise ValueError("protocol seed must be a persisted nonnegative integer")
        if self.input_privilege not in ("body_imu", "reference_attitude_assisted"):
            raise ValueError("declare input privilege explicitly")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class StageRecipe:
    """One common recipe for every optimizer stage; AMP is not qualified here."""

    optimizer: str = "adamw"
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    scheduler: str = "cosine"
    epochs: int = 30
    warmup_epochs: int = 0
    lr_final: float = 0.01
    gamma: float = 0.1
    step_size: int = 10
    plateau_patience: int = 5
    grad_clip: float = 0.0
    amp: bool = False
    workers: int = 0

    def __post_init__(self) -> None:
        if type(self.amp) is not bool or self.amp or type(self.workers) is not int or self.workers != 0:
            raise ValueError("complete-pair protocol is FP32/workers=0 until separately qualified")
        if self.optimizer not in ("adam", "adamw", "sgd") or self.scheduler not in ("cosine", "step", "constant", "plateau"):
            raise ValueError("unsupported common optimizer/scheduler")
        for key, minimum in (("epochs", 1), ("warmup_epochs", 0), ("step_size", 1), ("plateau_patience", 0)):
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup must finish before the common epoch budget")
        for key in ("lr", "momentum", "weight_decay", "grad_clip", "lr_final", "gamma"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite real number")
        if (self.lr <= 0 or not 0 <= self.momentum < 1 or self.weight_decay < 0 or self.grad_clip < 0
                or not 0 <= self.lr_final <= 1 or not 0 < self.gamma <= 1):
            raise ValueError("common stage recipe is outside its valid range")



@dataclass(frozen=True)
class PairResult:
    pair: int
    source_windows: int
    target_windows: int
    generator_steps: int
    discriminator_steps: int
    objectives: Mapping[str, float]


@dataclass(frozen=True)
class NativeProtocolInference:
    """Research inference payload; not yet a registered NIO native-export route."""

    identity: ProtocolIdentity
    model_state: Mapping[str, torch.Tensor]
    input_spec: Mapping[str, Any]
    architecture: Mapping[str, Any]
    inference_domain: int
    training_evidence: Mapping[str, int]

    def payload(self) -> dict:
        if self.inference_domain not in (0, 1):
            raise ValueError("inference domain must be source=0 or target=1")
        if any(not isinstance(v, torch.Tensor) for v in self.model_state.values()):
            raise ValueError("native numeric state must contain tensors only")
        if any(not isinstance(v, (str, bool, int, float)) for v in self.architecture.values()):
            raise ValueError("architecture metadata is scalar-only, never train data/state")
        allowed_spec = {"window", "frame", "orientation", "remove_gravity", "target", "dims",
                        "rate", "output_steps", "overlap", "history", "history_stride", "extra_inputs", "channels"}
        if set(self.input_spec) - allowed_spec:
            raise ValueError("native input spec cannot contain data paths or training state")
        evidence_keys = {"completed_pairs", "generator_steps", "discriminator_steps"}
        if (set(self.training_evidence) != evidence_keys
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                       for v in self.training_evidence.values())
                or len(set(self.training_evidence.values())) != 1):
            raise ValueError("native evidence requires equal completed-pair G/D counts")
        return immutable_copy({"schema": INFERENCE_SCHEMA, "inference_export": True,
            "protocol_identity": asdict(self.identity), "identity_sha256": self.identity.sha256,
            "model": dict(self.model_state), "input_spec": dict(self.input_spec),
            "architecture": dict(self.architecture), "inference_domain": self.inference_domain,
            "training_evidence": dict(self.training_evidence)})


def atomic_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete training or inference artifact with one rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(immutable_copy(dict(payload)), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class TrainingProtocol(ABC):
    """Future single-Trainer hook. No orchestration/validation/epoch loop lives here."""

    identity: ProtocolIdentity

    @abstractmethod
    def step(self) -> Optional[PairResult]: ...

    @abstractmethod
    def complete_epoch(self, validation_metric: Optional[float] = None) -> None: ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def inference(self, domain: int = 1) -> NativeProtocolInference: ...
