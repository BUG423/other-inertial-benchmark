"""Portable inference bundles and explicitly reviewed tensor export adapters.

Native packages preserve all model state and use the installed IPB Predictor.
Tensor exports expose raw task outputs, not windowing/world conversion/integration.
Unreviewed data-dependent or stochastic models never silently fall back to trace.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import random
import tempfile
from typing import Any, Optional

import torch
import numpy as np
from torch import nn

from .. import __version__
from ..cfg import ConfigError, cfg_to_dict
from ..cfg.export import ExportOptions, SCHEMA, file_sha256, verify_export_bundle
from ..nn.base import SequenceModel
from ..nn.registry import build_model
from ..utils import to_builtin
from ..utils.torch_utils import load_checkpoint, save_checkpoint


class ExportUnsupported(ConfigError):
    """The requested format has no scientifically reviewed executable contract."""


class RoNINVelocityAdapter(nn.Module):
    """Same pure tensor operations and weights as RoNINResNet.forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone, self.transition, self.fc = model.backbone, model.transition, model.fc
        self.linear = model.head.linear
        self.channels, self.window = model.input_spec.num_channels, model.input_spec.window

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.dim() != 3 or imu.size(1) != self.channels or imu.size(2) != self.window:
            raise RuntimeError("IMU shape conflicts with the exported scientific InputSpec")
        return self.linear(self.fc(self.transition(self.backbone(imu))))


class IMUNetVelocityAdapter(nn.Module):
    """Retains the learned noise-layer pairing; it is deterministic at inference."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.stem = model.input_block
        names = ("conv_1_1", "conv_1_2", "conv_3_1", "conv_3_2", "conv_4_1", "conv_4_2",
                 "conv_5_1", "conv_5_2", "conv_6_1", "conv_6_2", "conv_7_1", "conv_7_2")
        self.blocks = nn.Sequential(*(getattr(model, name) for name in names))
        self.output, self.noise, self.act = model.output_block, model.noise, model.act
        self.linear = model.head.linear
        self.channels, self.window = model.input_spec.num_channels, model.input_spec.window

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        if imu.dim() != 3 or imu.size(1) != self.channels or imu.size(2) != self.window:
            raise RuntimeError("IMU shape conflicts with the exported scientific InputSpec")
        features = self.output(self.blocks(self.stem(imu))).flatten(1)
        return self.linear(self.act(self.noise(features, imu.flatten(1))))


TENSOR_ADAPTERS = {
    "inertial_benchmark.models.ronin.model.RoNINResNet": RoNINVelocityAdapter,
    "inertial_benchmark.models.imunet.model.IMUNet": IMUNetVelocityAdapter,
}


def export_capabilities(model: Any) -> dict:
    """Capability is a supported code path, not a claim all profiles passed tests."""
    cls = f"{type(model).__module__}.{type(model).__name__}"
    spec = model.input_spec
    reason = None
    if isinstance(model, SequenceModel):
        reason = "sequence_physics_or_calibration_requires_native_IPB_pipeline"
    elif cls not in TENSOR_ADAPTERS:
        reason = "no_reviewed_tensor_adapter; data-dependent/stochastic operators are not traced"
    elif spec.extra_inputs or spec.sub_windows or spec.output_layout != "window":
        reason = "tensor_adapter_requires_one_window_without_extra_inputs"
    elif any(t.is_floating_point() and t.dtype != torch.float32
             for t in model.state_dict().values()):
        reason = "tensor_backend_currently_requires_float32; native_preserves_actual_dtypes"
    return {"native": {"supported": True, "runtime": "installed_compatible_IPB"},
            "torchscript": {"supported": reason is None, "reason": reason},
            "onnx": {"supported": reason is None, "reason": reason,
                     "dependencies": ["onnx", "onnxruntime"]}}


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(to_builtin(data), indent=2, allow_nan=False) + "\n")


@contextmanager
def _preserve_rng():
    """Preserve caller RNG without initializing CUDA in CPU-only processes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _provenance(model: Any, checkpoint: Optional[str]) -> dict:
    package_root = Path(__file__).resolve().parents[1]
    source_root = package_root.parent.parent
    binding = source_root / "source_manifest.json"
    dependencies = {}
    for name in ("torch", "numpy", "scipy", "h5py", "pyyaml"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"IPB_version": __version__, "dependencies": dependencies,
            "model_sources": {key: model.model_cfg.get(key)
                              for key in ("paper", "code", "commit", "license", "fidelity")},
            "source_manifest_sha256": file_sha256(binding) if binding.is_file() else None,
            "installed_python_source_sha256": {
                str(path.relative_to(package_root)): file_sha256(path)
                for path in sorted(package_root.rglob("*.py"))},
            "parent_checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None}


def _native_checkpoint(model: Any, cfg: Any, checkpoint: Optional[str]) -> tuple:
    """Current numeric state wins; a mismatched loss cannot inherit head evidence."""
    origin = load_checkpoint(checkpoint) if checkpoint else {}
    model_cfg = copy.deepcopy(model.model_cfg)
    model_cfg.update(loss=model.loss_name, loss_kwargs=dict(model.loss_kwargs))
    origin_loss = {"loss": (origin.get("model_cfg") or {}).get("loss"),
                   "loss_kwargs": dict((origin.get("model_cfg") or {}).get("loss_kwargs") or {})}
    live_loss = {"loss": model.loss_name, "loss_kwargs": dict(model.loss_kwargs)}
    loss_matches = not origin or origin_loss == live_loss
    trained = getattr(model, "_ipb_trained_epoch", None)
    live_evidence = {"completed_epoch": trained,
                     "epoch_optimizer_updates": getattr(model, "_ipb_epoch_optimizer_updates", 0),
                     "total_optimizer_updates": getattr(model, "_ipb_optimizer_updates", 0)}
    raw_state = model.state_dict()
    if not all(torch.is_tensor(tensor) for tensor in raw_state.values()):
        raise ExportUnsupported("native export supports numeric tensor state; pickle objects are forbidden")
    state = {name: tensor.detach().cpu().clone() for name, tensor in raw_state.items()}
    source_state = origin.get("model") or {}
    state_matches = bool(origin) and set(source_state) == set(state) and all(
        torch.is_tensor(source_state[key]) and source_state[key].dtype == value.dtype
        and torch.equal(source_state[key].cpu(), value) for key, value in state.items())
    updates, total = live_evidence["epoch_optimizer_updates"], live_evidence["total_optimizer_updates"]
    live_proof = (type(trained) is int and trained >= 0 and type(updates) is int
                  and updates > 0 and type(total) is int and total >= updates)
    evidence = live_evidence if live_proof else dict(origin.get("training_evidence") or {}) \
        if state_matches else {}
    if not loss_matches:
        evidence = {"completed_epoch": None, "epoch_optimizer_updates": 0,
                    "total_optimizer_updates": 0}
    epoch = trained if live_proof else origin.get("epoch", -1)
    if type(epoch) is not int:
        epoch = -1
    inference_cfg = copy.deepcopy(origin.get("cfg") or cfg_to_dict(cfg))
    inference_cfg.update(model=None, model_args={}, loss=None, pretrained=None, resume=False)
    phase = {"evaluation_epoch": epoch, "loss": live_loss,
             "parent_checkpoint_loss": origin_loss if origin else None,
             "parent_loss_matches_current_loss": loss_matches,
             "numeric_state_matches_parent_checkpoint": state_matches,
             "training_evidence": evidence,
             "evidence_policy": "no inferred optimizer/head training from epoch alone"}
    artifact = {"epoch": epoch, "model": state, "model_cfg": to_builtin(model_cfg),
                "input_spec": model.input_spec.to_dict(), "cfg": to_builtin(inference_cfg),
                "training_evidence": to_builtin(evidence),
                "dataset_fingerprint": origin.get("dataset_fingerprint"),
                "optimizer": None, "scheduler": None, "scaler": None,
                "version": __version__, "date": datetime.now(timezone.utc).isoformat(),
                "inference_export": {"schema": SCHEMA, "resume_training_supported": False,
                                     "phase": phase}}
    return artifact, phase


def _native_state_check(path: Path, cfg: Any, expected: dict) -> dict:
    recovered = load_checkpoint(path)
    if recovered["input_spec"] != expected["input_spec"] or recovered["model_cfg"] != expected["model_cfg"]:
        raise ConfigError("native export changed the scientific input/model configuration")
    # Building consumes initialization randomness; the caller preserves RNG.
    from ..nn.base import InputSpec

    replica = build_model(cfg, InputSpec.from_dict(recovered["input_spec"]), recovered["model_cfg"])
    try:
        replica.load_state_dict(recovered["model"], strict=True)
    except RuntimeError as exc:
        raise ExportUnsupported("installed model cannot reload the exported numeric state") from exc
    restored = replica.state_dict()
    if set(restored) != set(expected["model"]):
        raise ConfigError("native export changed numeric state membership")
    for key, tensor in expected["model"].items():
        if restored[key].dtype != tensor.dtype:
            raise ExportUnsupported(f"native reload cannot preserve state dtype for {key}; "
                                    "this artifact is not published")
        if not torch.equal(restored[key].cpu(), tensor):
            raise ConfigError(f"native reload changed numeric state: {key}")
    return {"numeric_state_tensor_count": len(restored), "state_input_spec_model_cfg_exact": True,
            "trajectory_parity": "requires_recording_acceptance; not_inferred_from_state_check"}


def _probes(spec: Any, batch: int) -> list:
    probes = []
    generator = torch.Generator(device="cpu").manual_seed(7723)
    for size, scale in ((batch, 0.0), (max(batch, 2), 1.0), (max(batch, 3), -2.0)):
        probes.append(torch.randn((size, *spec.input_shape), generator=generator) * scale)
    return probes


def _tensor_export(model: Any, directory: Path, options: ExportOptions) -> tuple:
    capability = export_capabilities(model)[options.format]
    if not capability["supported"]:
        raise ExportUnsupported(f"{options.format} unsupported: {capability['reason']}")
    if options.format == "onnx":
        try:
            import onnx
            import onnxruntime
        except ImportError as exc:
            raise ExportUnsupported("ONNX export verification requires pip install "
                                    "'other-inertial-benchmark[export]'") from exc
    reference = copy.deepcopy(model).cpu().eval()
    cls = f"{type(reference).__module__}.{type(reference).__name__}"
    adapter = TENSOR_ADAPTERS[cls](reference).eval()
    inputs = _probes(reference.input_spec, options.batch)
    # Scripting preserves control flow; tracing and eager fallback are forbidden.
    try:
        scripted = torch.jit.script(adapter)
    except Exception as exc:
        raise ExportUnsupported(f"reviewed adapter could not be scripted: {type(exc).__name__}: {exc}") from exc
    with torch.no_grad():
        expected = []
        for sample in inputs:
            original = reference(sample)
            if set(original) != {"vel"}:
                raise ExportUnsupported("tensor adapter would omit scientific auxiliary outputs")
            truth = original["vel"]
            if not torch.isfinite(truth).all():
                raise ExportUnsupported("nonfinite reference prediction cannot be verified")
            torch.testing.assert_close(adapter(sample), truth, rtol=0, atol=0)
            expected.append(truth)
    if options.format == "torchscript":
        name = "model.torchscript"
        scripted.save(str(directory / name))
        runtime = torch.jit.load(str(directory / name), map_location="cpu").eval()
        with torch.no_grad():
            actual = [runtime(sample) for sample in inputs]
    else:
        name = "model.onnx"
        kwargs = {"opset_version": options.opset, "input_names": ["imu"],
                  "output_names": ["vel"], "dynamic_axes": {"imu": {0: "batch"}, "vel": {0: "batch"}}}
        # A scripted, explicitly reviewed fixed-span graph supports the legacy
        # exporter on torch>=2.0; no eager trace is accepted for unknown models.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            kwargs["dynamo"] = False
        try:
            torch.onnx.export(scripted, inputs[0], str(directory / name), **kwargs)
            onnx.checker.check_model(onnx.load(str(directory / name)))
            session_options = onnxruntime.SessionOptions()
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 1
            runtime = onnxruntime.InferenceSession(str(directory / name), sess_options=session_options,
                                                  providers=["CPUExecutionProvider"])
            actual = [torch.from_numpy(runtime.run(["vel"], {"imu": sample.numpy()})[0]) for sample in inputs]
        except Exception as exc:
            raise ExportUnsupported(f"ONNX backend could not export and execute the reviewed adapter: "
                                    f"{type(exc).__name__}: {exc}") from exc
    errors = []
    for sample, truth, result in zip(inputs, expected, actual):
        torch.testing.assert_close(result, truth, rtol=options.rtol, atol=options.atol)
        errors.append({"batch": len(sample), "max_absolute_raw_output_error": float((result - truth).abs().max())})
    return name, {"backend_loaded_and_executed": True, "cpu_float32": True,
                  "torch_num_threads": torch.get_num_threads(),
                  "onnx_cpu_threads": {"intra_op": 2, "inter_op": 1} if options.format == "onnx" else None,
                  "fixed_scientific_input_shape": list(model.input_spec.input_shape),
                  "batch_dynamic": True, "raw_output_keys": ["vel"], "trace_used": False,
                  "rtol": options.rtol, "atol": options.atol, "actual_comparisons": errors}


def export_model(model: Any, cfg: Any, options: ExportOptions,
                 checkpoint: Optional[str] = None) -> Path:
    """Publish a verified bundle atomically and return its primary artifact.

    Native returns the directory for ``NIO(directory)``. Tensor formats return
    their backend file; ``NIO(file)`` reloads that bundle's verified native state.
    The artifact omits optimizer/trajectory data and preserves current weights.
    """
    name = str(model.model_cfg.get("name", "model"))
    if Path(name).name != name or name in ("", ".", ".."):
        raise ConfigError("model name must be a simple export directory name")
    destination = options.path or Path("runs/export") / name / options.format
    if destination.exists():
        raise ConfigError(f"export destination already exists: {destination}; choose a new path")
    capability = export_capabilities(model)[options.format]
    if not capability["supported"]:
        raise ExportUnsupported(f"{options.format} unsupported: {capability['reason']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with _preserve_rng():
            artifact, phase = _native_checkpoint(model, cfg, checkpoint)
            save_checkpoint(staging / "model.pt", artifact)
            native_validation = _native_state_check(staging / "model.pt", cfg, artifact)
            _json(staging / "input-spec.json", model.input_spec.to_dict())
            _json(staging / "model-config.json", artifact["model_cfg"])
            _json(staging / "training-phase.json", phase)
            _json(staging / "provenance.json", _provenance(model, checkpoint))
            primary, backend = ("model.pt", None)
            if options.format != "native":
                primary, backend = _tensor_export(model, staging, options)
        files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                 for path in sorted(staging.iterdir()) if path.is_file()}
        spec = model.input_spec
        manifest = {"schema": SCHEMA, "status": "verified", "format": options.format,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "primary_artifact": primary, "inference_checkpoint": "model.pt", "files": files,
                    "capabilities": export_capabilities(model), "validation": {"native": native_validation,
                                                                                "backend": backend},
                    "scientific_contract": {"frame": spec.frame, "orientation": spec.orientation,
                        "target": spec.target, "dims": spec.dims, "layout": spec.output_layout,
                        "unit": "m" if spec.target in ("displacement", "multi_displacement") else "m/s",
                        "channels": list(spec.channels), "output_scales_to_velocity": list(spec.output_scales),
                        "extra_inputs": list(spec.extra_inputs), "calibration_in_numeric_state": isinstance(model, SequenceModel),
                        "tensor_backend_scope": "raw_task_output_only; use_IPB_for_views_and_trajectory",
                        "XY": "slice_same_XYZ_prediction_for_XYZ_profiles"},
                    "training_phase": phase, "resume_training_supported": False,
                    "models_and_profiles_tested": "this_artifact_only; no_full_matrix_export_claim"}
        _json(staging / "manifest.json", manifest)
        verify_export_bundle(staging)
        if destination.exists():
            raise ConfigError("export destination appeared during publication")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination if options.format == "native" else destination / primary
