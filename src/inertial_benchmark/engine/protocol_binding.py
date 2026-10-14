"""Opt-in complete-pair binding for the existing Trainer.

The binding has no training loop. It verifies a protocol's admitted domains and
recipe, and exposes the source-only evaluation adapter to the normal evaluator.
Unregistered research protocols are supplied as an explicit Python factory;
there is no automatic import, model registration, or benchmark admission.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
from typing import Any, Callable
from collections.abc import Mapping
import random

import numpy as np
import torch

from ..data.manifest import DatasetSpec
from ..data.views import ViewConfig
from .training_protocol import (
    NativeProtocolInference, StageRecipe, TrainingProtocol, immutable_copy,
    preserve_rng_and_modes,
)

CHECKPOINT_SCHEMA = "ipb-Trainer-complete-pair-v1"


@dataclass(frozen=True)
class ProtocolSetupContext:
    args: Any
    device: torch.device
    source_spec: DatasetSpec


class ProtocolTrainerBinding:
    """A reviewed protocol plus its source-only native evaluation inputs.

    ``revalidate`` must bind actual certificates/files/readers, not just caller
    supplied fingerprints. It is invoked before checkpoint state is restored.
    The evaluation model shares scientific parameters with ``protocol.model``
    but implements a separately named source diagnostic, never the GAN loss.
    """

    def __init__(self, protocol: TrainingProtocol, *, source_spec: DatasetSpec,
                 val_set: Any, evaluation_model: torch.nn.Module,
                 revalidate: Callable[[], None], permission_identity: dict) -> None:
        if not isinstance(protocol, TrainingProtocol) or not callable(revalidate):
            raise TypeError("a typed protocol and actual input revalidation are required")
        for name in ("model", "source_dataset", "target_dataset", "recipe", "stream",
                     "aggregates", "optimizers", "schedulers", "start_epoch", "_healthy", "_epoch_complete"):
            if not hasattr(protocol, name):
                raise TypeError(f"complete-pair protocol is missing {name}")
        if not isinstance(protocol.recipe, StageRecipe):
            raise TypeError("protocol recipe must be typed")
        self.protocol = protocol
        self.model = protocol.model
        self.train_set = protocol.source_dataset
        self.val_set = val_set
        self.source_spec = source_spec
        self.evaluation_model = evaluation_model
        self.revalidate = revalidate
        self.permission_identity = immutable_copy(permission_identity)
        self.execution_identity = None
        self.evaluation_only = False
        self.best_evidence = None

    def validate_setup(self, context: ProtocolSetupContext) -> None:
        a, p = context.args, self.protocol
        if (context.source_spec.root.resolve() != self.source_spec.root.resolve()
                or self.source_spec.fingerprint() != p.identity.source.dataset_fingerprint
                or p.identity.experiment_seed != int(a.seed)
                or p.stream.source.batch_size != int(a.batch)
                or len(self.train_set) != p.identity.source.window_count):
            raise ValueError("Trainer source/seed/batch differs from admitted protocol identity")
        for key, expected in asdict(p.recipe).items():
            if getattr(a, key) != expected:
                raise ValueError(f"Trainer {key} differs from the two-stage common recipe")
        if int(a.train_windows_budget or 0):
            raise ValueError("resolve an explicit common epoch recipe before creating both schedulers")
        if p.recipe.scheduler == "plateau" and int(a.val_interval) != 1:
            raise ValueError("both plateau schedulers need source validation every epoch")
        if self.evaluation_model.input_spec != self.model.input_spec:
            raise ValueError("source evaluation must preserve the scientific InputSpec")
        requested = ViewConfig.from_cfg(a)
        if any(getattr(requested, key) != getattr(self.model.input_spec, key)
               for key in ViewConfig.__dataclass_fields__):
            raise ValueError("Trainer scientific view differs from the actual model InputSpec")
        owned = {id(v) for v in self.model.parameters()}
        if {id(v) for v in self.evaluation_model.parameters()} != owned:
            raise ValueError("source evaluation adapter must share the exact model parameters")
        canonical_val = set(self.source_spec.sequence_ids("val"))
        if (getattr(self.val_set, "training", True) is not False
                or not self.val_set.sequence_ids
                or not set(self.val_set.sequence_ids) <= canonical_val):
            raise ValueError("fitness requires a nonempty source-only canonical val subset")
        if any(v.device != context.device for v in self.model.parameters()):
            raise ValueError("construct both stage optimizers on the requested device")
        self.revalidate()
        self.execution_identity = self._execution_contract(a)

    @staticmethod
    def _index_digest(dataset: Any) -> str:
        index = dataset.index
        return hashlib.sha256(repr(index.shape).encode() + index.astype("<i8").tobytes()).hexdigest()

    def _execution_contract(self, args: Any) -> dict:
        keys = ("stride", "eval_stride", "augment", "metric_dims", "rte_delta", "t_rte", "d_rte",
                "min_speed", "fitness", "fitness_stat", "val_interval", "patience", "min_epochs")
        return immutable_copy({"sampling_evaluation": {key: getattr(args, key) for key in keys},
            "source_index_sha256": self._index_digest(self.train_set),
            "target_index_sha256": self._index_digest(self.protocol.target_dataset),
            "source_val_index_sha256": self._index_digest(self.val_set),
            "source_val_ids": list(self.val_set.sequence_ids)})

    @property
    def epoch(self) -> int:
        return int(self.protocol.stream.source.epoch)

    @property
    def epoch_complete(self) -> bool:
        return bool(self.protocol._epoch_complete)

    def ensure_trainable(self) -> None:
        if self.evaluation_only:
            raise RuntimeError("best inference weights cannot continue with last optimizer state; reload last")
        self.protocol._healthy()

    def begin_epoch(self, epoch: int) -> None:
        self.ensure_trainable()
        if self.epoch_complete:
            if epoch != self.epoch + 1:
                raise ValueError("advance a completed source epoch exactly once")
            self.protocol.start_epoch()
        elif epoch != self.epoch:
            raise ValueError("an incomplete source epoch must resume its committed cursor")

    def learning_rates(self) -> dict:
        return {stage: float(opt.param_groups[0]["lr"])
                for stage, opt in self.protocol.optimizers.items()}

    def evidence(self) -> dict:
        stream = self.protocol.stream
        return {"completed_pairs": int(stream.pairs),
                "generator_steps": int(stream.generator_steps),
                "discriminator_steps": int(stream.discriminator_steps),
                "source_epoch": self.epoch, "source_epoch_complete": self.epoch_complete,
                "source_windows": int(self.protocol.aggregates["source_windows"]),
                "target_windows": int(self.protocol.aggregates["target_windows"])}

    def row(self) -> dict:
        aggregate = self.protocol.aggregates
        n = max(int(aggregate["source_windows"]), 1)
        row = {f"train/{key}": value / n
               for key, value in aggregate["weighted_objective_sums"].items()}
        row["train/loss"] = row.get("train/total", 0.)
        row.update({f"train/{key}": aggregate[key]
                    for key in ("pairs", "source_windows", "target_windows")})
        row["train/source_epoch_complete"] = self.epoch_complete
        return row

    @contextmanager
    def source_evaluation(self):
        self.protocol._healthy()
        domain = self.model.inference_domain
        with preserve_rng_and_modes(self.model):
            try:
                self.model.inference_domain = 0
                yield self.evaluation_model
            finally:
                self.model.inference_domain = domain

    def validate_checkpoint(self, checkpoint: dict, *, training: bool) -> None:
        """Check identity and committed lifecycle before live state mutation.

        This is not rollback for arbitrary corrupted Torch optimizer state or
        hardware errors. A backend failure during load leaves the protocol
        poisoned; restore the last valid atomic checkpoint before continuing.
        """
        self.revalidate()
        p = self.protocol
        for key, dataset in (("source_index_sha256", self.train_set),
                             ("target_index_sha256", p.target_dataset),
                             ("source_val_index_sha256", self.val_set)):
            if self._index_digest(dataset) != self.execution_identity[key]:
                raise ValueError("actual scientific window index changed")
        if (checkpoint.get("protocol_checkpoint_schema") != CHECKPOINT_SCHEMA
                or checkpoint.get("protocol_identity") != asdict(p.identity)
                or checkpoint.get("protocol_identity_sha256") != p.identity.sha256
                or checkpoint.get("protocol_recipe") != asdict(p.recipe)
                or checkpoint.get("protocol_permission_identity") != self.permission_identity
                or checkpoint.get("protocol_execution_identity") != self.execution_identity
                or checkpoint.get("input_spec") != self.model.input_spec.to_dict()):
            raise ValueError("checkpoint code/data/input/recipe identity differs")
        actual = self.model.state_dict()
        saved = checkpoint.get("model", {})
        if (set(saved) != set(actual)
                or any(not torch.is_tensor(saved[k]) or saved[k].shape != v.shape
                       or saved[k].dtype != v.dtype for k, v in actual.items())):
            raise ValueError("checkpoint scientific parameter layout differs")
        evidence = checkpoint.get("protocol_training_evidence", {})
        keys = {"completed_pairs", "generator_steps", "discriminator_steps"}
        if (not keys <= evidence.keys()
                or any(type(evidence[k]) is not int or evidence[k] < 0 for k in keys)
                or len({evidence[k] for k in keys}) != 1):
            raise ValueError("checkpoint must prove complete D/G pairs")
        if (type(evidence.get("source_epoch")) is not int or evidence["source_epoch"] < 0
                or type(evidence.get("source_epoch_complete")) is not bool
                or checkpoint.get("epoch") != evidence["source_epoch"]):
            raise ValueError("checkpoint epoch must match its committed source epoch")
        if training:
            state = checkpoint.get("protocol_training")
            if not isinstance(state, dict):
                raise ValueError("inference-only best payload cannot resume either optimizer")
            if type(state.get("epoch_complete")) is not bool:
                raise ValueError("source epoch completion must be an explicit boolean")
            inner = state.get("model", {})
            if (set(inner) != set(saved)
                    or any(not torch.is_tensor(inner[k]) or inner[k].shape != saved[k].shape
                           or inner[k].dtype != saved[k].dtype for k in saved)):
                raise ValueError("inner model parameter layout differs from outer checkpoint")
            self._validate_stage_state(state)
            self._validate_rng_state(state.get("rng"))
            # Reuse the frozen sampler/stream validation on an independent small
            # state copy. No model/optimizer is copied, and its generator is
            # independent of global RNG and the live protocol's cursor.
            validated_stream = copy.deepcopy(p.stream)
            validated_stream.load_state_dict(state.get("stream"))
            if state["epoch_complete"] and not validated_stream.source.exhausted:
                raise ValueError("completed source epoch still has unconsumed windows")
            if (state["identity"] != asdict(p.identity) or state["recipe"] != asdict(p.recipe)
                    or evidence["source_epoch"] != state["stream"]["source"]["epoch"]
                    or evidence["source_epoch_complete"] != state["epoch_complete"]
                    or evidence["completed_pairs"] != state["stream"]["pairs"]
                    or any(not torch.equal(saved[k].cpu(), state["model"][k].cpu()) for k in saved)):
                raise ValueError("outer weights/evidence differ from the committed protocol state")

    def _validate_stage_state(self, state: dict) -> None:
        """Structural stage ownership checks; not arbitrary optimizer-slot repair."""
        optimizers, schedulers = state.get("optimizers"), state.get("schedulers")
        if (not isinstance(optimizers, Mapping) or not isinstance(schedulers, Mapping)
                or set(optimizers) != set(self.protocol.optimizers)
                or set(schedulers) != set(self.protocol.schedulers)):
            raise ValueError("checkpoint must retain both optimizer and scheduler stages")
        for stage, optimizer in self.protocol.optimizers.items():
            value = optimizers[stage]
            groups = value.get("param_groups") if isinstance(value, Mapping) else None
            if (not isinstance(groups, (list, tuple)) or len(groups) != len(optimizer.param_groups)
                    or not isinstance(value.get("state"), Mapping)):
                raise ValueError("optimizer parameter group structure differs")
            for saved, current in zip(groups, optimizer.param_groups):
                if (not isinstance(saved, Mapping) or not isinstance(saved.get("params"), (list, tuple))
                        or len(saved["params"]) != len(current["params"])):
                    raise ValueError("optimizer stage parameter ownership differs")
            scheduler = schedulers[stage]
            required = self.protocol.schedulers[stage].state_dict().keys()
            if not isinstance(scheduler, Mapping) or not set(required) <= scheduler.keys():
                raise ValueError("scheduler checkpoint structure differs")

    @staticmethod
    def _validate_rng_state(state: Any) -> None:
        """Use independent RNG objects; never change global evaluation/training draws."""
        if not isinstance(state, Mapping) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
            raise ValueError("checkpoint RNG fields differ")
        try:
            random.Random().setstate(state["python"])
            numpy = state["numpy"]
            keys = numpy["keys"]
            if (not isinstance(keys, torch.Tensor) or keys.device.type != "cpu"
                    or keys.dtype != torch.int64):
                raise ValueError("NumPy RNG keys must be a CPU integer tensor")
            np.random.RandomState().set_state((numpy["name"], keys.numpy().astype(np.uint32),
                numpy["cursor"], numpy["has_gauss"], numpy["cached"]))
            cpu = state["torch_cpu"]
            if not torch.is_tensor(cpu) or cpu.device.type != "cpu" or cpu.dtype != torch.uint8:
                raise ValueError("CPU Torch RNG must be a CPU byte state")
            torch.Generator(device="cpu").set_state(cpu)
            cuda = state["torch_cuda"]
            if cuda is not None:
                if not torch.cuda.is_initialized():
                    raise ValueError("a CPU protocol cannot restore saved CUDA RNG")
                if not isinstance(cuda, (list, tuple)) or len(cuda) != torch.cuda.device_count():
                    raise ValueError("saved CUDA RNG device count differs")
                for index, rng in enumerate(cuda):
                    if not torch.is_tensor(rng) or rng.device.type != "cpu" or rng.dtype != torch.uint8:
                        raise ValueError("CUDA RNG must use a saved CPU byte state")
                    torch.Generator(device=f"cuda:{index}").set_state(rng)
        except (TypeError, KeyError, IndexError, RuntimeError, ValueError) as error:
            raise ValueError("checkpoint RNG state is not restorable on the current protocol device") from error

    def load_training(self, checkpoint: dict) -> int:
        self.validate_checkpoint(checkpoint, training=True)
        self.protocol.load_state_dict(checkpoint["protocol_training"])
        self.evaluation_only = False
        self.best_evidence = None
        return self.epoch + int(self.epoch_complete)

    def load_best_for_inference(self, checkpoint: dict) -> None:
        self.validate_checkpoint(checkpoint, training=False)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.best_evidence = immutable_copy(checkpoint["protocol_training_evidence"])
        evidence = checkpoint.get("training_evidence", {})
        self.model._ipb_trained_epoch = evidence.get("completed_epoch")
        self.model._ipb_epoch_optimizer_updates = evidence.get("epoch_optimizer_updates", 0)
        self.model._ipb_optimizer_updates = evidence.get("total_optimizer_updates", 0)
        self.model._ipb_protocol_pair_evidence = immutable_copy(self.best_evidence)
        self.evaluation_only = True

    def inference(self, domain: int = 1) -> NativeProtocolInference:
        base = self.protocol.inference(domain)
        if not self.evaluation_only:
            return base
        evidence = {key: self.best_evidence[key]
                    for key in ("completed_pairs", "generator_steps", "discriminator_steps")}
        return replace(base, model_state=immutable_copy(self.model.state_dict()),
                       training_evidence=evidence)
