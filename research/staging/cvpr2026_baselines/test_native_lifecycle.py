"""UNEXECUTED server-only lifecycle drafts for ten CVPR2026 core adapters.

The local conftest rejects collection without the audited original-task tail
handoff.  The engineering fixture is not evidence of formal accuracy. Public
registry and original coverage/source/config files are never changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from inertial_benchmark import NIO
from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView
from inertial_benchmark.metrics import sequence_metrics
from inertial_benchmark.nn.registry import MODELS, import_models
from inertial_benchmark.utils.torch_utils import load_checkpoint


HERE = Path(__file__).resolve().parent
FIXTURE_MANIFEST_SHA = "3d8a676f45ec7b93a0365eadddee45ec1bad7025deb212a6dcecddc922bd8c53"
# Names must match the static drafts; independent review precedes execution.
CLASSES = {
    "vitt3": "ViT3Velocity",
    "edge_recvit": "EdgeRecViTVelocity",
    "pfgnet": "PFGNetVelocity",
    "binary_attention": "BinaryAttentionVelocity",
    "ucan": "UCANVelocity",
    "mstemba": "MSTembaVelocity",
    "specscalpel": "SpectralScalpelVelocity",
    "tessera_v1": "TesseraV1DualStreamVelocity",
    "csan_vit": "CSaNVelocity",
    "mixercseg": "TransMixerVelocity",
}


def _parameter_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.named_parameters()}


def _bounded_bytes(path, limit):
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    assert len(value) <= limit, "engineering fixture exceeds explicit small-file budget"
    return value


@pytest.mark.parametrize("method", list(CLASSES))
def test_native_xyz_train_val_predict_checkpoint_reload(method, tmp_path):
    value = os.environ.get("IPB_CROSS_DOMAIN_FIXTURE")
    if not value:
        pytest.fail("set IPB_CROSS_DOMAIN_FIXTURE to the admitted AutoDL engineering excerpt")
    fixture = Path(value).resolve(strict=True)
    private_fixture_root = os.environ.get(
        "IPB_V1_PRIVATE_EVIDENCE_ROOT", "<agent-project-root>/.local"
    )
    assert str(fixture).startswith(private_fixture_root.rstrip("/") + "/")
    manifest_path = (fixture / "dataset.json").resolve(strict=True)
    assert manifest_path.is_relative_to(fixture) and manifest_path.is_file()
    manifest_bytes = _bounded_bytes(manifest_path, 1024 * 1024)
    assert hashlib.sha256(manifest_bytes).hexdigest() == FIXTURE_MANIFEST_SHA
    manifest = json.loads(manifest_bytes)
    assert manifest["engineering_only"] is True and len(manifest["sequences"]) == 3
    assert all(len(manifest["splits"][split]) == 1 for split in ("train", "val", "test"))
    ids = [manifest["splits"][split][0] for split in ("train", "val", "test")]
    assert len(set(ids)) == 3
    fixture_paths = {}
    used = 0
    for sid, row in manifest["sequences"].items():
        relative = Path(row["file"])
        assert not relative.is_absolute() and ".." not in relative.parts
        path = (fixture / relative).resolve(strict=True)
        assert path.is_relative_to(fixture) and path.is_file()
        value = _bounded_bytes(path, min(32 * 1024 * 1024, 96 * 1024 * 1024 - used))
        used += len(value)
        assert hashlib.sha256(value).hexdigest() == row["sha256"]
        fixture_paths[sid] = path
    assert len(set(fixture_paths.values())) == 3
    spec = importlib.util.spec_from_file_location("cvpr2026_lifecycle_" + method,
                                                HERE / method / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = yaml.safe_load((HERE / method / "model_3d.yaml").read_text())
    assert profile["registered"] is False and profile["input"]["dims"] == 3
    assert profile["input"]["window"] == 200 and profile["loss"] == "mse"
    import_models()
    registry_before = dict(MODELS)
    arch = profile["arch"]
    assert arch not in MODELS
    MODELS[arch] = getattr(module, CLASSES[method])
    try:
        options = {
            "data": str(fixture), "device": "cpu", "metric_dims": 3, "epochs": 1,
            "batch": 2, "val_batch": 2, "workers": 0, "cache": False, "amp": False,
            "seed": 1043, "augment": [], "stride": 10, "eval_stride": 10,
            "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.0001,
            "efficiency": False, "plots": False, "save_predictions": True,
            "project": str(tmp_path / "runs"), "name": method + "_xyz_lifecycle",
            "gap_windows": 2,
        }
        nio = NIO(profile, device="cpu", metric_dims=3)
        state = {"updates": 0}

        def setup(trainer):
            state["initial"] = _parameter_hashes(trainer.model)
            assert len(trainer.train_set) == 4 and len(trainer.val_set) == 2
            sequence = load_sequence(fixture_paths[ids[0]])
            view = SequenceView(sequence, trainer.model.input_spec)
            targets = view.targets(view.starts(10))
            assert targets.shape[-1] == 3 and np.isfinite(targets).all()
            assert np.any(np.abs(targets[:, 2]) > 0), "real vertical engineering labels required"

        def step(trainer):
            state["updates"] += 1
            for name, parameter in trainer.model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, f"missing gradient: {name}"
                    assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"

        nio.add_callback("on_pretrain_routine_end", setup)
        nio.add_callback("on_train_batch_end", step)
        nio.train(**options)
        assert state["updates"] == 2
        after = _parameter_hashes(nio.model)
        assert any(digest != after[name] for name, digest in state["initial"].items())
        checkpoint = load_checkpoint(nio.trainer.last)
        assert checkpoint["training_evidence"]["total_optimizer_updates"] == state["updates"]
        assert checkpoint["model_cfg"]["loss"] == "mse" and checkpoint["input_spec"]["dims"] == 3
        nio.callbacks.clear()
        validation = nio.val(**{**options, "split": "val", "name": "before_reload_val"})
        assert len(validation.sequences) == 1 and not validation.sequences[0].skipped
        assert np.isfinite(validation.metrics["ate"])
        test_path = fixture_paths[ids[2]]
        nio.predict(test_path, **{**options, "name": "before_reload_predict"}, save=True)
        before = nio.results[0]
        assert not before.skipped and before.vel_pred.shape[-1] == 3
        arrays = {key: getattr(before, key).copy()
                  for key in ("vel_pred", "vel_target", "pos_pred", "pos_gt")}
        assert all(np.isfinite(array).all() for array in arrays.values())
        before.compute_metrics(3)
        xy = sequence_metrics(
            arrays["pos_pred"][:, :2], arrays["pos_gt"][:, :2], before.valid_pose,
            arrays["vel_pred"][:, :2], arrays["vel_target"][:, :2], before.window_valid,
            rate=before.rate, dims=2,
            pos_oracle=before.pos_oracle[:, :2] if before.pos_oracle is not None else None,
        )
        for key, expected in xy.items():
            np.testing.assert_allclose(before.metrics[f"xy_{key}"], expected,
                                       rtol=0, atol=0, equal_nan=True)
        loaded = NIO(nio.ckpt_path, device="cpu")
        assert loaded.model.input_spec == nio.model.input_spec
        assert loaded.model.model_cfg == nio.model.model_cfg
        revalidation = loaded.val(**{**options, "split": "val", "name": "after_reload_val"})
        for key in arrays:
            np.testing.assert_array_equal(getattr(validation.sequences[0], key),
                                          getattr(revalidation.sequences[0], key))
        loaded.predict(test_path, **{**options, "name": "after_reload_predict"}, save=True)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(expected, getattr(loaded.results[0], key))
        assert not torch.cuda.is_initialized()
    finally:
        MODELS.clear()
        MODELS.update(registry_before)
        assert MODELS == registry_before
