"""Server-only research transport tests; no NIO/raw-clock/CUDA qualification."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from .full_sequence_prediction import (
    CartesianUncertaintyUnavailable, FullSequenceContract, FullSequencePrediction,
    from_filter_trajectory, scientific_units,
)


def synthetic_payload(length=9):
    # Deliberately synthetic nonuniform nanosecond clock, never a raw recording.
    clock = np.arange(length, dtype=np.int64) * 10_000_000 + 1_600_000_000_000_000_000
    clock[1::2] += 17
    n = len(clock)
    t = np.arange(n, dtype=np.float64)
    p = np.stack((1 + t * .3, 2 + t ** 2, 3 + t * .7), -1)
    # Independently specified velocities: codec must preserve corrected p, not
    # integrate this deliberately unrelated velocity into a different path.
    v = np.tile(np.array([30., -20., 10.]), (n, 1))
    A = np.eye(21); A[6, 1] = .7; A[8, 0] = -.4
    P = np.tile((A @ A.T)[None], (n, 1, 1))
    return {"timestamp_ns": clock, "p_world": p, "v_world": v,
            "R_world_from_IMU": np.tile(np.eye(3)[None], (n, 1, 1)),
            "gyro_bias": np.tile(np.array([.01, -.02, .03]), (n, 1)),
            "accelerometer_bias": np.tile(np.array([.04, .05, -.06]), (n, 1)),
            "R_IMU_from_car": np.tile(np.eye(3)[None], (n, 1, 1)),
            "lever_arm": np.tile(np.array([.8, -.3, .2]), (n, 1)),
            "error_covariance": P, "measurement_variances": np.tile(np.array([.5, 2.]), (n, 1)),
            "valid_input": np.ones(n, dtype=bool)}


def contract():
    return FullSequenceContract("synthetic-transport", "a" * 64, "synthetic_engineering")


def make_prediction():
    return FullSequencePrediction(contract(), **synthetic_payload())


def test_direct_corrected_p_and_same_XYZ_XY_without_integration_or_frame_rotation():
    values = synthetic_payload(); pred = FullSequencePrediction(contract(), **values)
    xyz, xy = pred.metric_arrays(3), pred.metric_arrays(2)
    np.testing.assert_array_equal(xyz["pos_pred"], values["p_world"])
    np.testing.assert_array_equal(xyz["vel_pred"], values["v_world"])
    np.testing.assert_array_equal(xy["pos_pred"], xyz["pos_pred"][:, :2])
    assert np.shares_memory(xy["pos_pred"], pred.p_world)
    assert np.shares_memory(xyz["pos_pred"], pred.p_world)
    assert not np.allclose(np.diff(pred.p_world, axis=0), pred.v_world[:-1] * np.diff(pred.relative_seconds)[:, None])
    assert np.any(pred.p_world[:, 2] != 0)


def test_original_ns_jitter_survives_large_absolute_offset():
    pred = make_prediction(); times = pred.relative_seconds
    np.testing.assert_allclose(times[:3], [0., .010000017, .02], rtol=0, atol=1e-17)
    shifted = synthetic_payload(); shifted["timestamp_ns"] -= np.int64(1_600_000_000_000_000_000)
    np.testing.assert_array_equal(times, FullSequencePrediction(contract(), **shifted).relative_seconds)


def test_gaps_are_returned_as_masks_not_interpolated_or_filled():
    values = synthetic_payload(); values["valid_input"][3:5] = False
    pred = FullSequencePrediction(contract(), **values)
    arrays = pred.metric_arrays(3)
    np.testing.assert_array_equal(arrays["valid_input"], values["valid_input"])
    np.testing.assert_array_equal(arrays["pos_pred"][3:5], values["p_world"][3:5])
    assert len(arrays["t"]) == len(values["timestamp_ns"])


def test_arrays_are_owned_readonly_and_caller_mutation_cannot_change_transport():
    values = synthetic_payload(); pred = FullSequencePrediction(contract(), **values)
    saved = pred.p_world.copy(); values["p_world"][:] = 0
    np.testing.assert_array_equal(pred.p_world, saved)
    with pytest.raises(ValueError):
        pred.p_world[0, 0] = 7
    with pytest.raises(ValueError):
        pred.metric_arrays(2)["pos_pred"][0, 0] = 7


@pytest.mark.parametrize("field,kind", [("timestamp_ns", "float"), ("p_world", "float32"),
    ("v_world", "twoaxes"), ("R_world_from_IMU", "improper"),
    ("R_IMU_from_car", "reflection"), ("error_covariance", "negative"),
    ("error_covariance", "asymmetric"), ("measurement_variances", "threeaxes"),
    ("measurement_variances", "zero"), ("gyro_bias", "nonfinite"),
    ("valid_input", "integer"), ("valid_input", "empty")])
def test_invalid_state_contract_rejected_without_projection_or_covariance_repair(field, kind):
    values = synthetic_payload()
    if kind == "float": values[field] = values[field].astype(np.float64)
    elif kind == "float32": values[field] = values[field].astype(np.float32)
    elif kind == "twoaxes": values[field] = values[field][:, :2]
    elif kind == "improper": values[field][0, 0, 0] = 2
    elif kind == "reflection": values[field][0, 0, 0] = -1
    elif kind == "negative": values[field][0, 0, 0] = -3
    elif kind == "asymmetric": values[field][0, 0, 1] += .5
    elif kind == "threeaxes": values[field] = np.ones((9, 3))
    elif kind == "zero": values[field][0, 0] = 0
    elif kind == "nonfinite": values[field][0, 0] = np.nan
    elif kind == "integer": values[field] = values[field].astype(np.uint8)
    elif kind == "empty": values[field][:] = False
    with pytest.raises(ValueError):
        FullSequencePrediction(contract(), **values)


@pytest.mark.parametrize("clock", [[2, 1], [1, 1], [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
                                    [np.iinfo(np.int64).min, np.iinfo(np.int64).max]])
def test_original_clock_descending_duplicate_and_int64_overflow_rejected(clock):
    values = synthetic_payload(2); values["timestamp_ns"] = np.array(clock, dtype=np.int64)
    with pytest.raises(ValueError):
        FullSequencePrediction(contract(), **values)


@pytest.mark.parametrize("dims", [1, 4, True, 2.])
def test_metric_slice_dimension_guard(dims):
    with pytest.raises(ValueError):
        make_prediction().metric_arrays(dims)


def test_covariance_and_N_are_preserved_but_Cartesian_uncertainty_is_NA():
    values = synthetic_payload(); pred = FullSequencePrediction(contract(), **values)
    np.testing.assert_array_equal(pred.error_covariance, values["error_covariance"])
    np.testing.assert_array_equal(pred.measurement_variances, values["measurement_variances"])
    support = pred.uncertainty_support()
    assert not support["supported"] and not support["measurement_variances_are_prediction_sigma"]
    with pytest.raises(CartesianUncertaintyUnavailable):
        pred.position_covariance_world()
    # Independent diagnostic counterexample only. No mapped covariance is
    # returned or promoted to qualified uncertainty by the transport module.
    p = pred.p_world[0]; hat = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0.]])
    J = np.zeros((3, 21)); J[:, :3] = -hat; J[:, 6:9] = np.eye(3)
    assert not np.allclose(J @ pred.error_covariance[0] @ J.T, pred.error_covariance[0, 6:9, 6:9])


def test_caller_receipt_metadata_cannot_be_inferred_from_dtype_and_clock():
    pred = make_prediction()
    assert pred.contract.clock_provenance == "synthetic_engineering"
    assert pred.contract.raw_source_sha256 is None
    with pytest.raises(ValueError, match="receipts"):
        replace(pred.contract, clock_provenance="caller_acquisition_receipts")
    with pytest.raises(ValueError):
        replace(pred.contract, discretization="exact_discrete_with_new_reset")


def test_npz_roundtrip_exact_full_state_metadata_and_no_overwrite(tmp_path):
    pred = make_prediction(); path = tmp_path / "research-transport.npz"
    pred.save_npz(path); restored = FullSequencePrediction.load_npz(path)
    assert asdict(pred.contract) == asdict(restored.contract)
    for field in synthetic_payload():
        np.testing.assert_array_equal(getattr(pred, field), getattr(restored, field))
    with pytest.raises(FileExistsError):
        pred.save_npz(path)


@pytest.mark.parametrize("change", ["schema", "covariance_claim", "extra_sigma_field"])
def test_npz_transport_refuses_faked_export_uncertainty_capabilities(tmp_path, change):
    pred = make_prediction(); initial = tmp_path / "initial.npz"; bad = tmp_path / "bad.npz"
    pred.save_npz(initial)
    with np.load(initial, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    metadata = json.loads(values["metadata"].item())
    if change == "schema": metadata["schema"] = "public_NIO_export"
    elif change == "covariance_claim": metadata["Cartesian_uncertainty_supported"] = True
    else: values["world_sigma"] = np.ones((9, 3))
    values["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(bad, **values)
    with pytest.raises(ValueError):
        FullSequencePrediction.load_npz(bad)


def test_SI_contract_fields_are_explicit_and_invented_units_are_rejected():
    c = contract()
    assert (c.unit_system, c.timestamp_unit, c.position_unit, c.velocity_unit) == ("SI", "ns", "m", "m_s")
    assert (c.rotation_unit, c.gyro_bias_unit, c.accelerometer_bias_unit, c.lever_arm_unit) == (
        "dimensionless", "rad_s", "m_s2", "m")
    assert c.measurement_variance_unit == "m2_s2"
    assert c.error_covariance_units == "mixed_error_coordinate_outer_products"
    for name, value in [("position_unit", "cm"), ("velocity_unit", "km_h"),
                        ("rotation_unit", "deg"), ("gyro_bias_unit", "deg_s"),
                        ("accelerometer_bias_unit", "g"), ("lever_arm_unit", "cm"),
                        ("measurement_variance_unit", "m_s"), ("error_covariance_units", "m2")]:
        with pytest.raises(ValueError):
            replace(c, **{name: value})
    # Returned nested metadata must not be a mutable global contract.
    tampered = scientific_units(); tampered["error_covariance"]["coordinate_block_units"][0] = "deg"
    assert scientific_units()["error_covariance"]["coordinate_block_units"][0] == "rad"


@pytest.mark.parametrize("change", ["missing", "position_scale", "gyro_degrees", "acceleration_g",
                                    "N_standard_deviation", "P_single_unit", "P_coordinate_order"])
def test_npz_rejects_missing_or_tampered_SI_unit_declarations(tmp_path, change):
    initial = tmp_path / "initial.npz"; bad = tmp_path / "bad.npz"
    make_prediction().save_npz(initial)
    with np.load(initial, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    metadata = json.loads(values["metadata"].item())
    if change == "missing": del metadata["units"]
    elif change == "position_scale": metadata["units"]["p_world"] = "cm"
    elif change == "gyro_degrees": metadata["units"]["gyro_bias"] = "deg_s"
    elif change == "acceleration_g": metadata["units"]["accelerometer_bias"] = "g"
    elif change == "N_standard_deviation": metadata["units"]["measurement_variances"]["unit"] = "m_s"
    elif change == "P_single_unit": metadata["units"]["error_covariance"] = "m2"
    else: metadata["units"]["error_covariance"]["coordinate_order"][0:2] = ["xiv", "theta"]
    values["metadata"] = np.array(json.dumps(metadata)); np.savez_compressed(bad, **values)
    with pytest.raises(ValueError, match="SI unit"):
        FullSequencePrediction.load_npz(bad)


def test_from_actual_filter_tensor_output_preserves_all_original_values_RNG_and_batch_selection():
    # Reuses the accepted component at default width/float64; synthetic inputs.
    # No retraining, native reader, data download or CUDA invocation.
    from .test_scientific_contract import make_model, sample_input, sample_state
    model = make_model().eval(); clock, imu = sample_input(2, 13)
    with torch.no_grad():
        original = model(clock, imu, sample_state(2))
    rng = torch.get_rng_state().clone()
    pred = from_filter_trajectory(clock, original, contract(), batch_index=1,
                                  valid_input=np.ones(13, dtype=bool))
    mapping = {"p_world": "p", "v_world": "v", "R_world_from_IMU": "R",
               "gyro_bias": "bg", "accelerometer_bias": "ba", "R_IMU_from_car": "Rci",
               "lever_arm": "tci", "error_covariance": "P", "measurement_variances": "measurement_variances"}
    for dest, source in mapping.items():
        np.testing.assert_array_equal(getattr(pred, dest), getattr(original, source)[1].numpy())
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(pred.timestamp_ns, clock[1].numpy())
    assert pred.contract.unit_system == "SI" and pred.contract.position_unit == "m"
    assert pred.contract.velocity_unit == "m_s" and pred.contract.gyro_bias_unit == "rad_s"
    assert pred.contract.accelerometer_bias_unit == "m_s2" and pred.contract.lever_arm_unit == "m"
    units = scientific_units()
    assert units["measurement_variances"] == {"axes": ["car_lateral", "car_up"], "unit": "m2_s2"}
    assert units["error_covariance"]["coordinate_block_units"] == ["rad", "m_s", "m", "rad_s", "m_s2", "rad", "m"]
    with pytest.raises(ValueError):
        from_filter_trajectory(clock, original, contract(), batch_index=2, valid_input=np.ones(13, dtype=bool))
