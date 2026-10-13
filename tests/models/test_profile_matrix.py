"""All registered methods retain explicit scientific reference/adaptation tracks.

Runtime model/Trainer coverage is executed on AutoDL.  These checks prevent a
new XYZ head/profile from silently replacing the original planar protocol or
discarding its algorithm-specific target layout and loss.
"""

import pytest

from inertial_benchmark.cfg import get_cfg, load_model_cfg, model_yaml_names
from inertial_benchmark.nn.registry import MODELS, import_models


ADAPTED = (
    "ctin", "deepils", "eqnio_ronin", "imot", "imunet",
    "imunet_efficientnetb0", "imunet_mnasnet", "imunet_mobilenet",
    "imunet_mobilenetv2", "ionet", "nio_lie_events_ronin",
    "rio", "ronin_lstm", "ronin_resnet18", "ronin_resnet50",
    "ronin_resnet101", "ronin_tcn",
)


def test_every_registered_learning_architecture_has_an_xyz_profile():
    import_models()
    configs = [load_model_cfg(name) for name in model_yaml_names()]
    references = {"pdr", "mean_speed_heading"}
    xyz = {c["arch"] for c in configs if c["input"]["dims"] == 3}
    builtin = {name for name, cls in MODELS.items()
               if cls.__module__.startswith("inertial_benchmark.models.")}
    assert xyz | references == builtin
    assert all(c["arch"] in MODELS for c in configs)
    # Classical reference methods remain part of the matrix, with their
    # original calibrated planar definition and without fabricated Z.
    assert all(load_model_cfg(name)["input"]["dims"] == 2 for name in references)


@pytest.mark.parametrize("name", ADAPTED)
def test_xyz_adaptation_preserves_original_task_and_specific_loss(name):
    original = load_model_cfg(name)
    adapted = load_model_cfg(name + "_3d")
    assert original["input"]["dims"] == 2 and adapted["input"]["dims"] == 3
    assert original["arch"] == adapted["arch"]
    assert adapted["fidelity"].startswith("3d_adapted")
    assert "official" not in adapted["recipes"]
    assert original.get("loss") == adapted.get("loss")
    for key in ("window", "frame", "orientation", "remove_gravity", "target",
                "output_steps", "overlap", "history", "history_stride", "extra_inputs"):
        assert original["input"].get(key) == adapted["input"].get(key)
    before = get_cfg({"model": name})
    after = get_cfg({"model": name + "_3d"})
    assert before.augment == after.augment


def test_dive_keeps_original_xyz_and_explicit_planar_reference():
    assert load_model_cfg("dive")["input"]["dims"] == 3
    reference = load_model_cfg("dive_2d")
    assert reference["arch"] == "dive" and reference["input"]["dims"] == 2
    assert reference["fidelity"].startswith("2d_reference")


def test_author_excluded_method_cannot_be_reactivated_by_import():
    from inertial_benchmark.models.ionext.model import IONext

    assert IONext.benchmark_excluded is True
    import_models()
    assert "ionext" not in MODELS
    assert "ionext" not in model_yaml_names()
    assert "ionext_3d" not in model_yaml_names()
