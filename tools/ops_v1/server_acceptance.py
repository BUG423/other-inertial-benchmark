#!/usr/bin/env python3
"""Run v1 CPU acceptance using public raw files and the existing NIO API.

Invoke with the shared CPU flock and PYTHONPATH=src. Source data are read only;
all conversions, excerpts, checkpoints and receipts live under this v1 root.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ridi_source_view(root: Path, production: Path) -> tuple:
    """Add author lists in a private source view without modifying raw files."""
    raw = production / "datasets/ridi/raw_public_processed/data_publish_v2"
    lists = production / "datasets/ridi/audit/public_processed/source_splits"
    view = root / ".local/raw-views/ridi"
    view.mkdir(parents=True, exist_ok=True)
    for directory in sorted(raw.iterdir()):
        if directory.is_dir() and not (view / directory.name).exists():
            (view / directory.name).symlink_to(directory, target_is_directory=True)
    available = {}
    for split in ("train", "test"):
        name = f"list_{split}_publish_v2.txt"
        source = lists / name
        target = view / name
        if not target.exists():
            target.symlink_to(source)
        available[split] = [line.strip().split(",")[0] for line in source.read_text().splitlines()
                            if line.strip() and not line.lstrip().startswith("#")]
        available[split] = [name for name in available[split]
                            if (raw / name / "processed/data.csv").is_file()]
    training = []
    subjects = set()
    for name in available["train"]:
        subject = name.split("_")[0]
        if subject not in subjects:
            training.append(name)
            subjects.add(subject)
        if len(training) == 2:
            break
    test = next(name for name in available["test"] if name.split("_")[0] not in subjects)
    return view, training + [test]


def make_excerpts(root: Path, dataset: str, converted: Path, manifest: dict) -> tuple:
    """Copy 800 valid samples per split; retain disjoint recording identities."""
    import numpy as np

    from inertial_benchmark.data.format import load_sequence, save_sequence
    from inertial_benchmark.data.manifest import fingerprint
    from inertial_benchmark.data.splits import read_split, write_split

    output = root / "datasets/ipb-v1" / f"{dataset}_cpu_excerpt"
    sequences, splits, excerpt_records = {}, {}, []
    for split in ("train", "val", "test"):
        ids = read_split(converted / "splits" / f"{split}.txt")
        if not ids:
            raise RuntimeError(f"{dataset}: no independent {split} recording after conversion")
        sid = ids[0]
        source = converted / "sequences" / f"{sid}.h5"
        seq = load_sequence(source)
        valid = np.asarray(seq.valid_device, dtype=np.int64)
        prefix = np.r_[0, np.cumsum(valid)]
        candidates = np.flatnonzero(prefix[800:] - prefix[:-800] == 800)
        if not len(candidates):
            raise RuntimeError(f"{dataset}/{sid}: no 800-sample valid device/pose segment")
        start = int(candidates[0])
        excerpt = seq.slice(start, start + 800)
        excerpt.timestamp = np.arange(800, dtype=np.float64) / excerpt.sample_rate
        excerpt.attrs.update(cpu_acceptance_only=True, source_sequence_sha256=sha256(source),
                             source_slice_start=start, source_slice_stop=start + 800)
        target = output / "sequences" / f"{sid}.h5"
        save_sequence(target, excerpt)
        sequences[sid] = {"file": f"sequences/{sid}.h5", "sha256": sha256(target),
                          "num_samples": 800, "group_id": excerpt.group_id}
        splits[split] = [sid]
        write_split(output / "splits" / f"{split}.txt", [sid])
        excerpt_records.append({"split": split, "sequence_id": sid, "samples": 800,
                                "source_slice": [start, start + 800],
                                "source_sha256": sha256(source), "excerpt_sha256": sha256(target)})
        del seq, excerpt
        gc.collect()
    if len({sid for ids in splits.values() for sid in ids}) != 3:
        raise RuntimeError("CPU train/val/test excerpts must use distinct recordings")
    derived = {"schema_version": "1.0", "dataset": f"{dataset}_cpu_excerpt",
               "cpu_acceptance_only": True,
               "source_conversion_fingerprint": manifest["fingerprint"],
               "sequences": sequences, "splits": splits,
               "fingerprint": fingerprint({sid: item["sha256"] for sid, item in sequences.items()},
                                          splits)}
    write_json(output / "dataset.json", derived)
    return output, excerpt_records


def real_data_acceptance(root: Path, production: Path, stage: str) -> dict:
    import numpy as np
    import torch

    from inertial_benchmark import NIO
    from inertial_benchmark.data.convert import convert_dataset
    from inertial_benchmark.utils import yaml_load

    torch.set_num_threads(1)
    profile = yaml_load(root / "tools/ops_v1/profiles/autodl-cpu.yaml")
    receipts = []
    ridi_view, ridi_ids = ridi_source_view(root, production)
    sources = {"ridi": (ridi_view, ridi_ids),
               "ronin": (production / "datasets/ronin/raw_public_partial",
                         ["a000_1", "a000_8", "a000_7"])}
    for dataset, (raw, selected) in sources.items():
        converted = root / "datasets/ipb-v1" / f"{dataset}_smoke"
        manifest = convert_dataset(dataset, raw, output=converted, only=selected, workers=1,
                                   overwrite=False, val_fraction=0.5, seed=42, strata=[])
        if len(manifest["sequences"]) != 3:
            raise RuntimeError(f"{dataset}: expected three accepted public recordings")
        excerpt, records = make_excerpts(root, dataset, converted, manifest)
        project = root / "runs/acceptance" / stage / dataset
        model = NIO("ronin_resnet18")
        model.train(**profile, data=str(excerpt), project=str(project), name="cpu-real")
        trainer = model.trainer
        checkpoint = Path(model.ckpt_path)
        windows = len(trainer.train_set)
        updates = len(trainer.train_loader)
        if windows != 4 or updates != 2:
            raise RuntimeError(
                f"unexpected execution budget: {windows} windows / {updates} updates")
        reloaded = NIO(checkpoint)
        val = reloaded.val(**profile, data=str(excerpt), split="val",
                           project=str(project), name="reload-val")
        prediction_source = excerpt / "sequences" / f"{records[2]['sequence_id']}.h5"
        common = {**profile, "project": str(project)}
        model.predict(prediction_source, **common, name="before-reload", save=True)
        reloaded.predict(prediction_source, **common, name="after-reload", save=True)
        before, after = model.results[0], reloaded.results[0]
        if before.skipped or after.skipped:
            raise RuntimeError(f"prediction skipped: {before.skipped}/{after.skipped}")
        np.testing.assert_array_equal(before.pos_pred, after.pos_pred)
        artifacts = {str(path.relative_to(project)): sha256(path) for path in project.rglob("*")
                     if path.is_file() and path.suffix in (".pt", ".npz", ".png", ".json", ".yaml")}
        receipts.append({"dataset": dataset, "source_raw": str(raw), "raw_selection": selected,
                         "converted_recordings": 3,
                         "conversion_fingerprint": manifest["fingerprint"],
                         "conversion_manifest_sha256": sha256(converted / "dataset.json"),
                         "excerpts": records, "train_windows": windows,
                         "optimizer_updates": updates,
                         "epochs": 1, "model": "ronin_resnet18",
                         "model_parameters": model.model.num_params,
                         "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                         "nio_train_val_predict_save_load": True,
                         "checkpoint_prediction_max_abs_error": 0.0,
                         "val_sequences": len(val.sequences), "artifacts": artifacts,
                         "export_completed": False, "publication_eligible": False})
        del trainer, model, reloaded
        gc.collect()
    return {"schema": "ib.dev-v1.cpu-real-acceptance.v1", "stage": stage,
            "created_utc": datetime.now(timezone.utc).isoformat(), "datasets": receipts,
            "source_manifest_sha256": sha256(root / "source_manifest.json"),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "h5py", "scipy", "pandas", "pytest", "ruff")},
            "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "cuda_available": torch.cuda.is_available(), "torch_threads": torch.get_num_threads(),
            "scope": "real public recordings, short disjoint excerpts, execution acceptance only"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--production", type=Path,
                        default=Path("/root/autodl-tmp/inertial-benchmark"))
    parser.add_argument("--stage", default="baseline-faee13c")
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1")
    receipt = real_data_acceptance(root, args.production.resolve(), args.stage)
    target = root / ".local/reports" / f"{args.stage}-cpu-real.json"
    write_json(target, receipt)
    print(json.dumps({"receipt": str(target), "datasets": len(receipt["datasets"]),
                      "converted_recordings": 6, "optimizer_updates": 4,
                      "checkpoint_prediction_max_abs_error": 0.0, "publication_eligible": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
