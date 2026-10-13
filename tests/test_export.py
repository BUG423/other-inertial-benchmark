"""Inference-artifact contracts; execute framework tests on the benchmark server."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
from synthetic import make_sequence

torch = pytest.importorskip("torch")

from inertial_benchmark import NIO  # noqa: E402
from inertial_benchmark.cfg import ConfigError, get_cfg  # noqa: E402
from inertial_benchmark.cfg.export import (  # noqa: E402
    ExportOptions,
    resolve_export_model,
    verify_export_bundle,
)
from inertial_benchmark.cli import entrypoint  # noqa: E402
from inertial_benchmark.data.views import SequenceView, ViewConfig  # noqa: E402
from inertial_benchmark.engine.exporter import ExportUnsupported, export_capabilities  # noqa: E402
from inertial_benchmark.engine.predictor import spec_to_view, uncertainty_contract  # noqa: E402
from inertial_benchmark.engine.trainer import Trainer  # noqa: E402
from inertial_benchmark.utils import to_builtin  # noqa: E402
from inertial_benchmark.utils.torch_utils import load_checkpoint, save_checkpoint  # noqa: E402


def tiny_nio(dims=3):
    return NIO("ronin_resnet18", dims=dims, metric_dims=dims, device="cpu",
               model_args={"group_sizes": [1, 1], "base_plane": 8,
                           "fc_dim": 16, "trans_planes": 4})


def predictions(nio, sequence):
    nio.predict(sequence, device="cpu", eval_stride=10, val_batch=2, efficiency=False, plots=False)
    result = nio.results[0]
    assert not result.skipped
    return {key: getattr(result, key).copy()
            for key in ("vel_pred", "vel_target", "pos_pred", "pos_gt", "pos_oracle")}


def checkpoint(nio, path, epoch=12, evidence=None):
    save_checkpoint(path, {"epoch": epoch, "model": nio.model.state_dict(),
        "input_spec": nio.model.input_spec.to_dict(), "model_cfg": nio.model.model_cfg,
        "training_evidence": evidence or {}, "cfg": get_cfg(nio.overrides).to_dict(),
        "dataset_fingerprint": "test-source-fingerprint"})


def test_native_same_xyz_trajectory_and_rng_and_no_mutation(tmp_path):
    nio = tiny_nio()
    seq = make_sequence(duration=2, seed=19)
    reference = predictions(nio, seq)
    training_flag = nio.model.training
    state = {key: value.clone() for key, value in nio.model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    artifact = nio.export(path=tmp_path / "native")
    assert torch.equal(rng, torch.get_rng_state())
    assert python_rng == random.getstate()
    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
    assert numpy_rng[0] == np.random.get_state()[0] and numpy_rng[2:] == np.random.get_state()[2:]
    assert nio.model.training == training_flag
    for key, value in state.items():
        assert torch.equal(value, nio.model.state_dict()[key])
    manifest = verify_export_bundle(artifact)
    assert manifest["scientific_contract"]["dims"] == 3
    assert manifest["scientific_contract"]["channels"] == ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
    assert manifest["validation"]["native"]["state_input_spec_model_cfg_exact"]
    assert manifest["training_phase"]["training_evidence"] == {}
    restored = predictions(NIO(artifact), seq)
    for key, value in reference.items():
        np.testing.assert_array_equal(value, restored[key])
    assert get_cfg({"model": str(artifact)}).dims == 3
    # XY always comes from this same XYZ run, not a second planar prediction.
    reloaded = NIO(artifact)
    reloaded.predict(seq, device="cpu", val_batch=2, eval_stride=10)
    reloaded.results[0].compute_metrics(3)
    assert "xy_ate" in reloaded.results[0].metrics


@pytest.mark.parametrize("profile", ["pdr", "mean_speed_heading", "ridi", "ridi_3d"])
def test_numeric_classical_calibration_and_physics_roundtrip(profile, tmp_path):
    args = {"body_axes": {"synthetic_device": [[0, 1, 0], [0, 0, -1]]}}
    if profile.startswith("ridi"):
        args = {"max_fit_windows": 8, "cache_mb": 8}
    elif profile == "pdr":
        args["segment_s"] = 2.0
    nio = NIO(profile, device="cpu", orientation="reference", model_args=args)
    train = make_sequence(duration=4, seed=4, sequence_id="train_calibration")
    test = make_sequence(duration=2, seed=9, sequence_id="different_test")
    view = SequenceView(train, ViewConfig.from_cfg(get_cfg(nio.overrides)))
    stats = nio.model.calibrate([view], split="train")
    nio.model.model_cfg["calibration"] = stats
    reference = predictions(nio, test)
    artifact = nio.export(path=tmp_path / profile)
    manifest = verify_export_bundle(artifact)
    assert manifest["scientific_contract"]["calibration_in_numeric_state"]
    saved = load_checkpoint(artifact / "model.pt")
    assert saved["model_cfg"]["calibration"] == to_builtin(stats)
    assert all(torch.is_tensor(t) for t in saved["model"].values())
    assert saved["optimizer"] is None
    reloaded = NIO(artifact)
    for key, value in nio.model.state_dict().items():
        assert torch.equal(value, reloaded.model.state_dict()[key])
    actual = predictions(reloaded, test)
    for key, value in reference.items():
        np.testing.assert_array_equal(value, actual[key])
    assert not export_capabilities(nio.model)["torchscript"]["supported"]


def test_native_exact_phase_evidence_and_source_hashes(tmp_path):
    fresh = NIO("tlio", device="cpu")
    source = tmp_path / "source.pt"
    evidence = {"completed_epoch": 12, "epoch_optimizer_updates": 2, "total_optimizer_updates": 24}
    checkpoint(fresh, source, evidence=evidence)
    loaded = NIO(source, device="cpu")
    artifact = loaded.export(path=tmp_path / "copy")
    saved = load_checkpoint(artifact / "model.pt")
    assert saved["epoch"] == 12 and saved["training_evidence"] == evidence
    assert saved["model_cfg"]["loss"] == fresh.model.loss_name
    phase = verify_export_bundle(artifact)["training_phase"]
    assert phase["numeric_state_matches_parent_checkpoint"]
    assert phase["parent_loss_matches_current_loss"]
    provenance = json.loads((artifact / "provenance.json").read_text())
    assert len(provenance["parent_checkpoint_sha256"]) == 64
    assert provenance["installed_python_source_sha256"]["engine/exporter.py"]


@pytest.mark.parametrize("changed", ["loss", "weights"])
def test_export_does_not_recertify_unmatched_trained_head(changed, tmp_path):
    fresh = NIO("tlio", device="cpu")
    source = tmp_path / "source.pt"
    checkpoint(fresh, source, evidence={"completed_epoch": 12,
        "epoch_optimizer_updates": 2, "total_optimizer_updates": 24})
    loaded = NIO(source, device="cpu", **({"loss": "mse"} if changed == "loss" else {}))
    if changed == "weights":
        with torch.no_grad():
            next(loaded.model.parameters()).add_(.1)
    artifact = loaded.export(path=tmp_path / "changed")
    saved = load_checkpoint(artifact / "model.pt")
    assert not saved["training_evidence"].get("total_optimizer_updates", 0)
    assert saved["model_cfg"]["loss"] == loaded.model.loss_name
    assert saved["epoch"] == 12  # evaluation phase survives; head training is separate evidence
    restored = NIO(artifact, device="cpu")
    contract = uncertainty_contract(restored.model, spec_to_view(restored.model.input_spec), 12)
    assert not contract["head_trained"]
    expected = "unsupported_or_nonprobabilistic_scientific_loss" if changed == "loss" else \
        "missing_completed_optimizer_training_evidence"
    assert contract["reason"] == expected


@pytest.mark.parametrize("epoch,evidence,reason", [
    (0, {"completed_epoch": 0, "epoch_optimizer_updates": 2, "total_optimizer_updates": 2},
     "uncertainty_head_before_training_phase"),
    (12, {}, "missing_completed_optimizer_training_evidence"),
])
def test_export_keeps_before_phase_and_untrained_sigma_unavailable(epoch, evidence, reason, tmp_path):
    source = tmp_path / "source.pt"
    checkpoint(NIO("tlio", device="cpu"), source, epoch=epoch, evidence=evidence)
    bundle = NIO(source, device="cpu").export(path=tmp_path / "phase")
    loaded = NIO(bundle, device="cpu")
    contract = uncertainty_contract(loaded.model, spec_to_view(loaded.model.input_spec), epoch)
    assert not contract["head_trained"] and contract["reason"] == reason


def test_loss_kwargs_override_cannot_inherit_sigma_training_proof(tmp_path):
    source = tmp_path / "source.pt"
    checkpoint(NIO("tlio", device="cpu"), source, evidence={"completed_epoch": 12,
        "epoch_optimizer_updates": 2, "total_optimizer_updates": 24})
    # Same loss name, different scientific phase: it is still a different loss contract.
    changed = NIO(source, device="cpu", loss="nll_detach_then_nll", loss_switch_epoch=10)
    bundle = changed.export(path=tmp_path / "different-switch")
    loaded = NIO(bundle, device="cpu")
    contract = uncertainty_contract(loaded.model, spec_to_view(loaded.model.input_spec), 12)
    assert not contract["head_trained"]
    assert contract["reason"] == "missing_completed_optimizer_training_evidence"
    assert not verify_export_bundle(bundle)["training_phase"]["parent_loss_matches_current_loss"]


def test_legacy_checkpoint_no_invented_training_evidence(tmp_path):
    nio = tiny_nio()
    path = tmp_path / "legacy.pt"
    checkpoint(nio, path, epoch=17)
    exported = NIO(path).export(path=tmp_path / "legacy-copy")
    assert load_checkpoint(exported / "model.pt")["training_evidence"] == {}


def test_unrelated_historical_manifest_does_not_claim_an_export_bundle(tmp_path):
    path = tmp_path / "history.pt"
    checkpoint(tiny_nio(), path, epoch=17)
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "another.training-run.v1"}))
    assert resolve_export_model(path) == path
    assert NIO(path).model.input_spec.dims == 3


@pytest.mark.parametrize("entry", ["trainer-path", "trainer-auto", "nio-auto", "cli-path", "cli-auto"])
def test_inference_export_cannot_resume_before_training_directory_creation(entry, tmp_path):
    artifact = tiny_nio().export(path=tmp_path / "inference")
    project = tmp_path / "never-created"
    common = {"data": "ridi", "project": str(project), "name": "blocked", "device": "cpu"}
    if entry == "trainer-path":
        with pytest.raises(ConfigError, match="inference export cannot resume"):
            Trainer(overrides={**common, "resume": str(artifact / "model.pt")})
    elif entry == "trainer-auto":
        with pytest.raises(ConfigError, match="inference export cannot resume"):
            Trainer(overrides={**common, "model": str(artifact), "resume": True})
    elif entry == "nio-auto":
        with pytest.raises(ConfigError, match="inference export cannot resume"):
            NIO(artifact, device="cpu").train(**common, resume=True)
    else:
        resume = str(artifact / "model.pt") if entry == "cli-path" else "true"
        assert entrypoint(["train", f"model={artifact}", "data=ridi", f"project={project}",
                           "name=blocked", f"resume={resume}", "device=cpu"]) == 2
    assert not project.exists()


def test_inference_weights_can_initialize_a_new_training_run(tmp_path):
    artifact = tiny_nio().export(path=tmp_path / "inference")
    project = tmp_path / "new-run"
    trainer = Trainer(overrides={"model": str(artifact), "pretrained": str(artifact / "model.pt"),
        "data": "ridi", "device": "cpu", "resume": False, "project": str(project)})
    assert trainer.resume_ckpt is None and trainer.args.pretrained == str(artifact / "model.pt")
    # Constructor acceptance is not a claim that this unit case trained anything.
    assert not project.exists()


def test_live_completed_epoch_supersedes_older_parent_phase(tmp_path):
    source = tmp_path / "old.pt"
    checkpoint(tiny_nio(), source, epoch=3)
    nio = NIO(source, device="cpu")
    # Construct the engine's documented live metadata contract for this unit case.
    # Genuine Trainer proof is separately checked on accepted server checkpoints.
    nio.model._ipb_trained_epoch = 8
    nio.model._ipb_epoch_optimizer_updates = 2
    nio.model._ipb_optimizer_updates = 18
    with torch.no_grad():
        next(nio.model.parameters()).add_(.1)
    artifact = nio.export(path=tmp_path / "live")
    state = load_checkpoint(artifact / "model.pt")
    assert state["epoch"] == 8 and state["training_evidence"]["completed_epoch"] == 8
    assert not verify_export_bundle(artifact)["training_phase"]["numeric_state_matches_parent_checkpoint"]


def test_native_preserves_privileged_extra_input_contract(tmp_path):
    nio = NIO("nio_lie_events_ronin_3d", device="cpu")
    bundle = nio.export(path=tmp_path / "events")
    restored = NIO(bundle)
    assert restored.model.input_spec.to_dict() == nio.model.input_spec.to_dict()
    assert {"orientation", "init_velocity"}.issubset(restored.model.input_spec.extra_inputs)
    with pytest.raises(ExportUnsupported, match="no_reviewed_tensor_adapter"):
        nio.export(format="torchscript", path=tmp_path / "not-traced")
    assert not (tmp_path / "not-traced").exists()


@pytest.mark.parametrize("member", ["model.pt", "input-spec.json", "training-phase.json"])
def test_bundle_corruption_fails_before_reload(member, tmp_path):
    artifact = tiny_nio().export(path=tmp_path / "corrupt")
    with (artifact / member).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ConfigError, match="integrity failed"):
        NIO(artifact)
    with pytest.raises(ConfigError, match="integrity failed"):
        NIO(artifact / "model.pt")


def test_manifest_member_cannot_escape_bundle(tmp_path):
    artifact = tiny_nio().export(path=tmp_path / "escape")
    manifest = json.loads((artifact / "manifest.json").read_text())
    manifest["files"]["../outside.pt"] = {"bytes": 1, "sha256": "0" * 64}
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ConfigError, match="unsafe or missing"):
        verify_export_bundle(artifact)


def test_existing_export_directory_is_preserved(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ConfigError, match="already exists"):
        tiny_nio().export(path=output)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("field,value", [("primary_artifact", []), ("files", []),
                                        ("inference_checkpoint", {"name": "model.pt"})])
def test_malformed_manifest_types_raise_config_error(field, value, tmp_path):
    artifact = tiny_nio().export(path=tmp_path / "malformed")
    document = json.loads((artifact / "manifest.json").read_text())
    document[field] = value
    (artifact / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ConfigError):
        NIO(artifact / "model.pt")


def test_native_rejects_unreloadable_dtype_without_publishing(tmp_path):
    nio = tiny_nio()
    nio.model.double()
    output = tmp_path / "unsupported-dtype"
    with pytest.raises(ExportUnsupported, match="cannot preserve state dtype"):
        nio.export(path=output)
    assert not output.exists()


@pytest.mark.parametrize("values", [{"format": "traced"}, {"batch": True}, {"batch": 0},
    {"opset": 100}, {"rtol": float("nan")}, {"atol": -1}, {"dynamic": True}])
def test_typed_export_options_reject_invalid_requests(values):
    with pytest.raises(ConfigError):
        ExportOptions.from_kwargs(values)


@pytest.mark.parametrize("profile", ["ronin_resnet18_3d", "imunet_3d"])
def test_reviewed_torchscript_actual_runtime_dynamic_batch_and_input_guard(profile, tmp_path):
    nio = tiny_nio() if profile.startswith("ronin") else NIO(profile, device="cpu")
    artifact = nio.export(format="torchscript", path=tmp_path / profile)
    manifest = verify_export_bundle(artifact.parent)
    assert not manifest["validation"]["backend"]["trace_used"]
    assert manifest["validation"]["backend"]["backend_loaded_and_executed"]
    runtime = torch.jit.load(str(artifact), map_location="cpu").eval()
    nio.model.eval()
    generator = torch.Generator().manual_seed(899)
    with torch.no_grad():
        for batch in (1, 2, 5):
            x = torch.randn(batch, *nio.model.input_spec.input_shape, generator=generator)
            torch.testing.assert_close(runtime(x), nio.model(x)["vel"], rtol=1e-5, atol=1e-6)
    with pytest.raises(Exception, match="InputSpec"):
        runtime(torch.zeros(1, 6, nio.model.input_spec.window + 1))
    assert resolve_export_model(artifact) == str(artifact.parent / "model.pt")
    assert NIO(artifact).model.input_spec.dims == 3


@pytest.mark.parametrize("profile", ["ronin_resnet18_3d", "imunet_3d"])
def test_onnx_verified_runtime_dynamic_batch_and_native_reload(profile, tmp_path):
    nio = tiny_nio() if profile.startswith("ronin") else NIO(profile, device="cpu")
    output = tmp_path / profile
    try:
        import onnx  # noqa: F401
        import onnxruntime
    except ImportError:
        with pytest.raises(ExportUnsupported, match="verification requires"):
            nio.export(format="onnx", path=output)
        assert not output.exists()
        return
    artifact = nio.export(format="onnx", path=output)
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    runtime = onnxruntime.InferenceSession(str(artifact), sess_options=session_options,
                                         providers=["CPUExecutionProvider"])
    nio.model.eval()
    for batch in (1, 2, 5):
        sample = np.linspace(-2, 2, batch * np.prod(nio.model.input_spec.input_shape),
                             dtype=np.float32).reshape(batch, *nio.model.input_spec.input_shape)
        with torch.no_grad():
            expected = nio.model(torch.from_numpy(sample))["vel"].numpy()
        np.testing.assert_allclose(runtime.run(["vel"], {"imu": sample})[0], expected,
                                   rtol=1e-5, atol=1e-6)
    assert verify_export_bundle(output)["validation"]["backend"]["backend_loaded_and_executed"]
    assert runtime.get_session_options().intra_op_num_threads == 2
    assert runtime.get_session_options().inter_op_num_threads == 1
    assert NIO(artifact).model.input_spec.dims == 3


def test_cli_export_native_and_typed_errors(tmp_path, capsys):
    source = tmp_path / "input.pt"
    checkpoint(tiny_nio(), source, epoch=0)
    out = tmp_path / "cli"
    assert entrypoint(["export", f"model={source}", f"path={out}", "format=native"]) == 0
    assert str(out) in capsys.readouterr().out
    assert NIO(out).model.input_spec.dims == 3
    assert entrypoint(["export", f"model={source}", "format=not-a-format"]) == 2
    assert entrypoint(["export", f"model={source}", "batch=false"]) == 2
    assert entrypoint(["export", f"model={source}", "dynamic=true"]) == 2
    assert entrypoint(["export", "format=native"]) == 2
