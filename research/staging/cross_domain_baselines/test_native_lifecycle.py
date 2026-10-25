"""Server-only native lifecycle for isolated cross-domain architecture adapters.

The caller supplies a previously admitted three-sequence RIDI excerpt and holds
the supplemental CPU lease. Registration is temporary and process-local; the
public registry and frozen experiment files are never modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import yaml

from inertial_benchmark import NIO
from inertial_benchmark.data.format import load_sequence
from inertial_benchmark.data.views import SequenceView
from inertial_benchmark.nn.registry import MODELS, import_models
from inertial_benchmark.metrics import sequence_metrics
from inertial_benchmark.utils.torch_utils import load_checkpoint


HERE = Path(__file__).resolve().parent
CLASSES = {
    "moderntcn": "ModernTCNVelocity",
    "patchtst": "PatchTSTVelocity",
    "s4d": "S4DVelocity",
}


def _parameter_hashes(model):
    return {
        name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in model.named_parameters()
    }


@pytest.mark.parametrize("method", list(CLASSES))
@pytest.mark.parametrize("dims", [2, 3])
def test_admitted_native_train_validate_predict_reload(method, dims, tmp_path):
    started = time.monotonic()
    fixture_value = os.environ.get("IPB_CROSS_DOMAIN_FIXTURE")
    if not fixture_value:
        pytest.fail("set IPB_CROSS_DOMAIN_FIXTURE to the accepted server engineering excerpt")
    fixture = Path(fixture_value).resolve(strict=True)
    private_fixture_root = os.environ.get(
        "IPB_V1_PRIVATE_EVIDENCE_ROOT", "<agent-project-root>/.local"
    )
    assert str(fixture).startswith(private_fixture_root.rstrip("/") + "/")
    manifest_bytes = (fixture / "dataset.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["engineering_only"] is True
    assert len(manifest["sequences"]) == 3
    assert all(len(manifest["splits"][split]) == 1 for split in ("train", "val", "test"))
    ids = [manifest["splits"][split][0] for split in ("train", "val", "test")]
    assert len(set(ids)) == 3
    fixture_hashes = {}
    for sid, row in manifest["sequences"].items():
        path = fixture / row["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == row["sha256"], sid
        fixture_hashes[path] = actual

    model_file = HERE / method / "model.py"
    module_spec = importlib.util.spec_from_file_location("cross_domain_lifecycle_" + method, model_file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    profile = yaml.safe_load((HERE / method / f"model_{dims}d.yaml").read_text())
    assert profile["input"]["window"] == 200 and profile["input"]["dims"] == dims
    assert profile["loss"] == "mse"
    import_models()
    registry_before = dict(MODELS)
    arch = profile["arch"]
    assert arch not in MODELS
    MODELS[arch] = getattr(module, CLASSES[method])
    try:
        options = {
            "data": str(fixture), "device": "cpu", "metric_dims": dims,
            "epochs": 1, "batch": 2, "val_batch": 2, "workers": 0,
            "cache": False, "amp": False, "seed": 1043, "augment": [],
            "stride": 10, "eval_stride": 10, "optimizer": "adamw", "lr": 0.001,
            "weight_decay": 0.0001, "efficiency": False, "plots": False,
            "save_predictions": True, "project": str(tmp_path / "runs"),
            "name": method + "_native_lifecycle", "gap_windows": 2,
        }
        nio = NIO(profile, device="cpu", metric_dims=dims)
        state = {"updates": 0}

        def record_setup(trainer):
            state["initial"] = _parameter_hashes(trainer.model)
            assert len(trainer.train_set) == 4 and len(trainer.val_set) == 2
            train_seq = load_sequence(fixture / manifest["sequences"][ids[0]]["file"])
            view = SequenceView(train_seq, trainer.model.input_spec)
            targets = view.targets(view.starts(10))
            assert targets.shape == (4, dims) and np.isfinite(targets).all()
            if dims == 3:
                assert np.any(np.abs(targets[:, 2]) > 0), "real vertical targets required"

        def record_step(trainer):
            state["updates"] += 1
            for name, parameter in trainer.model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, f"missing gradient: {name}"
                    assert torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}"

        nio.add_callback("on_pretrain_routine_end", record_setup)
        nio.add_callback("on_train_batch_end", record_step)
        nio.train(**options)
        assert state["updates"] == 2
        current = _parameter_hashes(nio.model)
        assert any(value != current[name] for name, value in state["initial"].items())
        last = load_checkpoint(nio.trainer.last)
        assert last["training_evidence"]["total_optimizer_updates"] == 2
        assert last["model_cfg"]["loss"] == "mse"
        assert last["input_spec"]["dims"] == dims
        assert all(torch.isfinite(value).all() for value in last["model"].values())

        nio.callbacks.clear()
        validation = nio.val(**{**options, "split": "val", "name": "before_reload_val"})
        assert len(validation.sequences) == 1 and not validation.sequences[0].skipped
        assert np.isfinite(validation.metrics["ate"])
        test_path = fixture / manifest["sequences"][ids[2]]["file"]
        nio.predict(test_path, **{**options, "name": "before_reload_predict"}, save=True)
        before = nio.results[0]
        assert not before.skipped and before.vel_pred.shape[-1] == dims
        arrays = {key: getattr(before, key).copy()
                  for key in ("vel_pred", "vel_target", "pos_pred", "pos_gt")}
        assert all(np.isfinite(value).all() for value in arrays.values())
        if dims == 3:
            before.compute_metrics(dims)
            sliced_metrics = sequence_metrics(
                arrays["pos_pred"][:, :2], arrays["pos_gt"][:, :2], before.valid_pose,
                arrays["vel_pred"][:, :2], arrays["vel_target"][:, :2], before.window_valid,
                rate=before.rate, dims=2,
                pos_oracle=before.pos_oracle[:, :2] if before.pos_oracle is not None else None,
            )
            assert np.isfinite(sliced_metrics["ate"])
            for key, expected in sliced_metrics.items():
                np.testing.assert_allclose(before.metrics[f"xy_{key}"], expected,
                                           rtol=0, atol=0, equal_nan=True, err_msg=key)
            for key, expected in arrays.items():
                np.testing.assert_array_equal(getattr(before, key), expected)

        loaded = NIO(nio.ckpt_path, device="cpu")
        assert loaded.model.input_spec == nio.model.input_spec
        assert loaded.model.model_cfg == nio.model.model_cfg
        after_validation = loaded.val(**{**options, "split": "val", "name": "after_reload_val"})
        for key in arrays:
            np.testing.assert_array_equal(getattr(validation.sequences[0], key),
                                          getattr(after_validation.sequences[0], key))
        loaded.predict(test_path, **{**options, "name": "after_reload_predict"}, save=True)
        for key, expected in arrays.items():
            np.testing.assert_array_equal(expected, getattr(loaded.results[0], key))
        assert not torch.cuda.is_initialized()
        receipt = {
            "schema": "ib.cross-domain-baseline.native-lifecycle.v1",
            "method": method, "dims": dims, "status": "passed",
            "input_spec": nio.model.input_spec.to_dict(),
            "model_cfg": nio.model.model_cfg, "parameters": nio.model.num_params,
            "actual_optimizer_updates": state["updates"],
            "all_trainable_parameters_have_finite_gradients_each_step": True,
            "changed_parameter_tensors": sum(value != current[name]
                                             for name, value in state["initial"].items()),
            "train_windows": 4, "val_windows": 2, "distinct_fixture_sequences": ids,
            "fixture_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "model_source_sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(Path(nio.ckpt_path).read_bytes()).hexdigest(),
            "validation_predictor_reload_arrays_exact": True,
            "prediction_shape": list(arrays["vel_pred"].shape),
            "xyz_xy_same_prediction": dims == 3, "cuda_initialized": False,
            "xy_slice_metrics_verified": sorted(sliced_metrics) if dims == 3 else [],
            "engineering_only": True, "accuracy_experiment": False,
            "elapsed_s": time.monotonic() - started,
        }
        (tmp_path / "lifecycle-receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    finally:
        MODELS.pop(arch, None)
    assert MODELS == registry_before
    assert (fixture / "dataset.json").read_bytes() == manifest_bytes
    for path, expected in fixture_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
