"""Shared Trainer protocol tests; generated native fixtures are engineering only.

Default-width D/G updates and interrupted third-pair replay use hidden=128.
Other focused rejection cases use explicitly smaller generated math fixtures.
Physical-domain admission is a separate, root-reviewed finite server receipt.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import h5py
import numpy as np
import pytest
import torch

from inertial_benchmark.cfg import ConfigError, get_cfg
from inertial_benchmark.engine.trainer import Trainer
from inertial_benchmark.engine.training_protocol import capture_rng, restore_rng
from inertial_benchmark.utils.torch_utils import load_checkpoint, save_checkpoint
from research.staging.algorithm_ports.motiontransformer.phase_a_protocol import IncompletePairError
from research.staging.algorithm_ports.motiontransformer.test_phase_a_protocol import (
    file_sha, make_source, make_target, same,
)
from research.staging.algorithm_ports.motiontransformer.trainer_binding import (
    MotionTrainerFactory, MotionTrainerInputs,
)


def prepare(tmp_path, *, hidden=8, source_count=6, scheduler="cosine", epochs=2):
    source, guard, sm, sa = make_source(tmp_path / "source", source_count, 3, 1043)
    target, tm, ta = make_target(tmp_path / "target", 7, guard)
    coverage = json.loads((tmp_path / "target/reference-input-coverage.json").read_text())
    proof_path, review_path = tmp_path / "synthetic-input-proof.json", tmp_path / "synthetic-review.json"
    proof = {"synthetic_engineering_fixture_not_formal_science": True,
             "source_proof": {sid: {"file_sha256": sm["sequences"][sid]["sha256"]}
                              for sid in source.sequence_ids},
             "target_proof": {"target-train": {"orientation_sha256": coverage["orientation_sha256"],
                 "timestamp_sha256": coverage["timestamp_sha256"],
                 "coverage_intervals": coverage["intervals"], "num_samples": coverage["sample_count"]}}}
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    review_path.write_text(json.dumps({"schema": "ipb-synthetic-engineering-permission-v1",
        "synthetic_engineering_fixture_not_formal_science": True,
        "source_labelled_ids": source.sequence_ids, "target_id": "target-train"}, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    inputs = MotionTrainerInputs(tmp_path / "source", tmp_path / "target", root,
        file_sha(root / "source_manifest.json"), sa, file_sha(sa), ta, file_sha(ta),
        proof_path, file_sha(proof_path), review_path, file_sha(review_path), guard,
        tuple(source.sequence_ids), tuple(target.sequence_ids), ("source-val",),
        {"target-train": "recording"}, hidden=hidden)
    model_cfg = {"name": "motion_phase_A_unregistered", "arch": "unregistered_motion_phase_A",
                 "input": {"window": 400, "rate": 200, "dims": 3}, "recipes": {}}
    args = {"model": model_cfg, "data": str(tmp_path / "source"), "device": "cpu", "seed": 1043,
            "batch": 2, "val_batch": 2, "stride": 200, "eval_stride": 200, "cache": True,
            "workers": 0, "augment": [], "amp": False, "epochs": epochs, "min_epochs": 0,
            "optimizer": "adamw", "lr": .0001, "weight_decay": .0001, "momentum": .9,
            "scheduler": scheduler, "warmup_epochs": 0, "val_interval": 1, "patience": 10,
            "metric_dims": 3, "gap_windows": 2, "grad_clip": 0., "plots": False,
            "save_predictions": False, "efficiency": False}
    return inputs, args


def ready(tmp_path, *, hidden=8, source_count=6, **kwargs):
    inputs, args = prepare(tmp_path, hidden=hidden, source_count=source_count, **kwargs)
    trainer = Trainer(overrides=args, save_dir=tmp_path / "run",
                      protocol_factory=MotionTrainerFactory(inputs))
    trainer.setup()
    return trainer, inputs, args


def shutdown(*trainers):
    for trainer in trainers:
        trainer._close_log()


def draws():
    return {"torch": torch.rand(7), "numpy": np.random.random(7).tolist(),
            "python": [random.random() for _ in range(7)]}


def test_default128_shared_Trainer_pair2_checkpoint_third_exact_resume(tmp_path):
    trainer, inputs, args = ready(tmp_path, hidden=128)
    resumed = None
    try:
        binding, protocol = trainer.protocol_binding, trainer.protocol_binding.protocol
        assert trainer.train_loader is trainer.optimizer is trainer.scheduler is trainer.scaler is None
        initial = copy.deepcopy(trainer.model.state_dict())
        trainer.train_protocol_pairs(0, max_pairs=2)
        assert protocol.model.core.hidden == 128 and protocol.model.input_spec.window == 400
        assert protocol.model.input_spec.rate == 200 and protocol.stream.pairs == 2
        assert protocol.stream.generator_steps == protocol.stream.discriminator_steps == 2
        assert set(protocol.aggregates["weighted_objective_sums"]) == {
            "discriminator", "total", "gan", "reconstruction", "prediction", "cycle", "perceptual"}
        assert all(np.isfinite(v) for v in protocol.aggregates["weighted_objective_sums"].values())
        for stage, names in protocol.parameter_names.items():
            assert any(not torch.equal(initial[n], trainer.model.state_dict()[n]) for n in names)
            assert protocol.optimizers[stage].state
        trainer.save_model(0, False)
        checkpoint = torch.load(trainer.last, weights_only=True)
        assert checkpoint["training_evidence"]["completed_epoch"] is None
        assert checkpoint["protocol_training_evidence"]["completed_pairs"] == 2
        trainer.train_protocol_pairs(0, max_pairs=1)
        expected = protocol.state_dict()
        expected_draws = draws()
        resumed = Trainer(overrides={**args, "resume": str(trainer.last)},
                          protocol_factory=MotionTrainerFactory(inputs))
        resumed.setup()
        assert resumed.start_epoch == 0
        assert resumed.protocol_binding.protocol.stream.source.cursor == 4
        resumed.train_protocol_pairs(0, max_pairs=1)
        same(expected, resumed.protocol_binding.protocol.state_dict())
        same(expected_draws, draws())
        assert resumed.protocol_binding.protocol.stream.pairs == 3
        audit = resumed.protocol_binding.protocol.target_dataset.audit.events
        assert not any(e.get("field") in {"pose/position", "pose/velocity", "valid/pose"} for e in audit)
    finally:
        shutdown(trainer, *([resumed] if resumed else []))


def test_protocol_setup_builds_no_legacy_loader_or_all_parameter_optimizer(tmp_path, monkeypatch):
    inputs, args = prepare(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError("unused legacy construction")
    monkeypatch.setattr("inertial_benchmark.engine.trainer.build_dataloader", forbidden)
    monkeypatch.setattr("inertial_benchmark.engine.trainer.build_optimizer", forbidden)
    trainer = Trainer(overrides=args, save_dir=tmp_path / "run", protocol_factory=MotionTrainerFactory(inputs))
    try:
        trainer.setup()
        partitions = [{id(p) for g in opt.param_groups for p in g["params"]}
                      for opt in trainer.protocol_binding.protocol.optimizers.values()]
        assert not partitions[0] & partitions[1]
        assert partitions[0] | partitions[1] == {id(p) for p in trainer.model.parameters()}
    finally:
        shutdown(trainer)


def test_source_validator_XYZ_same_prediction_XY_and_rng_modes_preserved(tmp_path):
    trainer, _, _ = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=2)
        trainer.model.inference_domain = 1
        trainer.model.core.discriminators[0].eval()
        modes = {n: m.training for n, m in trainer.model.named_modules()}
        rng = capture_rng()
        metrics = trainer.validate(0)
        same(rng, capture_rng())
        assert modes == {n: m.training for n, m in trainer.model.named_modules()}
        assert trainer.model.inference_domain == 1
        assert trainer.val_result.cfg["validation_domain"] == "source"
        assert trainer.val_result.cfg["loss_semantics"] == "source_polar_velocity_diagnostic"
        assert metrics["loss"] >= 0 and np.isfinite(metrics["loss"])
        result = trainer.val_result.sequences[0]
        assert result.pos_pred.shape[1] == 3 and result.vel_pred.shape[1] == 3
        assert np.any(result.vel_pred[:, 2] != 0)
        assert any(k.startswith("xy_") for k in metrics)
        assert not any(e.get("field") in {"pose/position", "pose/velocity", "valid/pose"}
                       for e in trainer.protocol_binding.protocol.target_dataset.audit.events)
    finally:
        shutdown(trainer)


def test_single_Trainer_epoch_advances_both_schedulers_once_and_strips_best(tmp_path):
    inputs, args = prepare(tmp_path, source_count=4, epochs=1)
    trainer = Trainer(overrides=args, save_dir=tmp_path / "run", protocol_factory=MotionTrainerFactory(inputs))
    metrics = trainer.train()
    assert np.isfinite(metrics["loss"])
    last, best = load_checkpoint(trainer.last), load_checkpoint(trainer.best)
    state = last["protocol_training"]
    assert state["epoch_complete"] is True
    assert state["stream"]["generator_steps"] == state["stream"]["discriminator_steps"] == 2
    assert state["schedulers"]["generator"]["last_epoch"] == 1
    assert state["schedulers"]["discriminator"]["last_epoch"] == 1
    assert best["protocol_training"] is None
    assert best["protocol_training_evidence"] == last["protocol_training_evidence"]
    assert best["training_evidence"]["completed_epoch"] == 0
    assert trainer.protocol_binding.evaluation_only is True
    payload = trainer.protocol_binding.inference(1).payload()
    assert payload["training_evidence"]["completed_pairs"] == 2
    assert "optimizers" not in payload and "stream" not in payload
    with pytest.raises(RuntimeError, match="last optimizer"):
        trainer.train_protocol_pairs(1, max_pairs=1)


@pytest.mark.parametrize("change", ["source", "recipe", "input", "outer_weights", "sampling", "permission"])
def test_resume_identity_rejection_precedes_parameter_mutation(tmp_path, change):
    trainer, _, _ = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=1)
        checkpoint = trainer.checkpoint(0)
        initial = copy.deepcopy(trainer.model.state_dict())
        if change == "source":
            checkpoint["protocol_identity"]["source_manifest_sha256"] = "a" * 64
        elif change == "recipe":
            checkpoint["protocol_recipe"]["lr"] *= 2
        elif change == "input":
            checkpoint["input_spec"]["window"] = 200
        elif change == "outer_weights":
            checkpoint["model"] = {k: v.clone() for k, v in checkpoint["model"].items()}
            key = next(k for k, v in checkpoint["model"].items() if v.is_floating_point())
            checkpoint["model"][key].add_(1)
        elif change == "sampling":
            checkpoint["protocol_execution_identity"]["sampling_evaluation"]["stride"] += 1
        else:
            checkpoint["protocol_permission_identity"]["permission_review"] = "a" * 64
        with pytest.raises(ValueError):
            trainer.protocol_binding.load_training(checkpoint)
        same(initial, trainer.model.state_dict())
    finally:
        shutdown(trainer)


def test_half_pair_poison_blocks_Trainer_checkpoint_val_and_inference(tmp_path, monkeypatch):
    trainer, _, _ = ready(tmp_path)
    try:
        protocol = trainer.protocol_binding.protocol
        def fail(*a, **kw):
            raise RuntimeError("injected after D optimizer step")
        monkeypatch.setattr(protocol.model.core, "generator_objective", fail)
        with pytest.raises(IncompletePairError):
            trainer.train_protocol_pairs(0, max_pairs=1)
        assert protocol.stream.pairs == protocol.stream.generator_steps == protocol.stream.discriminator_steps == 0
        for operation in (lambda: trainer.checkpoint(0), lambda: trainer.validate(0),
                          lambda: trainer.protocol_binding.inference()):
            with pytest.raises(IncompletePairError):
                operation()
        assert not trainer.last.exists()
    finally:
        shutdown(trainer)


def test_best_evidence_is_not_last_optimizer_evidence(tmp_path):
    trainer, _, _ = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=1)
        best = trainer.checkpoint(0)
        trainer.train_protocol_pairs(0, max_pairs=1)
        assert trainer.protocol_binding.protocol.stream.pairs == 2
        trainer.protocol_binding.load_best_for_inference(best)
        payload = trainer.protocol_binding.inference(0).payload()
        assert payload["training_evidence"]["completed_pairs"] == 1
        same(best["model"], payload["model"])
        assert trainer.model._ipb_optimizer_updates == 2
    finally:
        shutdown(trainer)


def test_plateau_needs_source_validation_every_epoch(tmp_path):
    inputs, args = prepare(tmp_path, scheduler="plateau")
    args["val_interval"] = 2
    trainer = Trainer(overrides=args, save_dir=tmp_path / "run", protocol_factory=MotionTrainerFactory(inputs))
    with pytest.raises(ValueError, match="source validation every epoch"):
        trainer.train()


def test_changed_permission_bytes_rejected_before_protocol_setup(tmp_path):
    inputs, args = prepare(tmp_path)
    inputs.permission_review.write_text("{}\n")
    trainer = Trainer(overrides=args, save_dir=tmp_path / "run", protocol_factory=MotionTrainerFactory(inputs))
    with pytest.raises(ValueError, match="permission_review bytes changed"):
        trainer.train()
    assert trainer.model is None and trainer.protocol_binding is None


def test_protocol_checkpoint_requires_explicit_factory_and_best_cannot_resume(tmp_path):
    trainer, inputs, args = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=1)
        trainer.save_model(0, False)
        with pytest.raises(ConfigError, match="explicit typed protocol factory"):
            Trainer(overrides={**args, "resume": str(trainer.last)})
        best = trainer.checkpoint(0)
        best["protocol_training"] = None
        save_checkpoint(trainer.best, best)
        resumed = Trainer(overrides={**args, "resume": str(trainer.best)},
                          protocol_factory=MotionTrainerFactory(inputs))
        with pytest.raises(ValueError, match="inference-only best"):
            resumed.train()
    finally:
        shutdown(trainer)


def test_observation_callbacks_preserve_paired_training_rng_and_modes(tmp_path):
    inputs, args = prepare(tmp_path, source_count=4, epochs=1)
    baseline = Trainer(overrides=args, save_dir=tmp_path / "baseline",
                       protocol_factory=MotionTrainerFactory(inputs))
    baseline.train()
    expected = load_checkpoint(baseline.last)["protocol_training"]
    observed = Trainer(overrides=args, save_dir=tmp_path / "observed",
                       protocol_factory=MotionTrainerFactory(inputs))
    def observation(trainer):
        draws()
        trainer.model.eval()
    for event in ("on_pretrain_routine_end", "on_train_start", "on_train_epoch_start",
                  "on_train_batch_start", "on_train_batch_end", "on_train_epoch_end",
                  "on_model_save", "on_fit_epoch_end"):
        observed.add_callback(event, observation)
    observed.train()
    same(expected, load_checkpoint(observed.last)["protocol_training"])


def test_false_completed_epoch_rejected_without_live_model_optimizer_sampler_rng_mutation(tmp_path):
    trainer, _, _ = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=2)
        forged = trainer.checkpoint(0)
        assert forged["protocol_training"]["stream"]["source"]["cursor"] == 4
        assert trainer.protocol_binding.protocol.stream.source.limit == 6
        forged["protocol_training"]["epoch_complete"] = True
        forged["protocol_training_evidence"]["source_epoch_complete"] = True
        trainer.train_protocol_pairs(0, max_pairs=1)
        before = trainer.protocol_binding.protocol.state_dict()
        with pytest.raises(ValueError, match="unconsumed windows"):
            trainer.protocol_binding.load_training(forged)
        same(before, trainer.protocol_binding.protocol.state_dict())
        assert trainer.protocol_binding.protocol._poisoned is False
    finally:
        shutdown(trainer)


@pytest.mark.parametrize("change", ["inner_model", "optimizer_groups", "scheduler_mapping", "cpu_saved_cuda",
                                     "python_rng", "numpy_rng", "torch_rng"])
def test_late_semantic_state_rejection_preserves_live_pair_and_global_rng(tmp_path, change):
    trainer, _, _ = ready(tmp_path)
    try:
        trainer.train_protocol_pairs(0, max_pairs=2)
        forged = trainer.checkpoint(0)
        inner = forged["protocol_training"]
        if change == "inner_model":
            inner["model"] = dict(inner["model"])
            inner["model"]["unexpected_scientific_parameter"] = torch.zeros(1)
        elif change == "optimizer_groups":
            inner["optimizers"]["generator"]["param_groups"] = []
        elif change == "scheduler_mapping":
            inner["schedulers"]["discriminator"] = []
        elif change == "cpu_saved_cuda":
            inner["rng"]["torch_cuda"] = [torch.zeros(1, dtype=torch.uint8)]
        elif change == "python_rng":
            inner["rng"]["python"] = (999, (), None)
        elif change == "numpy_rng":
            inner["rng"]["numpy"]["name"] = "invalid-generator"
        else:
            inner["rng"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
        trainer.train_protocol_pairs(0, max_pairs=1)
        before = trainer.protocol_binding.protocol.state_dict()
        with pytest.raises(ValueError):
            trainer.protocol_binding.load_training(forged)
        same(before, trainer.protocol_binding.protocol.state_dict())
        assert trainer.protocol_binding.protocol._poisoned is False
        assert torch.cuda.is_initialized() is False
    finally:
        shutdown(trainer)
