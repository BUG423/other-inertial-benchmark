"""Typed export options and integrity checks; no torch import in this module."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

SCHEMA = "ib.inference-artifact.v1"
FORMATS = ("native", "torchscript", "onnx")


def file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportOptions:
    """Scientific input dimensions stay in InputSpec; only batch is dynamic.

    ``path`` is the destination bundle directory, including for tensor formats.
    Existing destinations are never overwritten. Every backend is verified.
    """

    format: str = "native"
    path: Optional[Path] = None
    batch: int = 1
    opset: int = 17
    rtol: float = 1e-5
    atol: float = 1e-6

    @classmethod
    def from_kwargs(cls, values: Mapping[str, Any]) -> "ExportOptions":
        from . import ConfigError

        allowed = {"format", "path", "batch", "opset", "rtol", "atol"}
        unknown = set(values) - allowed
        if unknown:
            raise ConfigError(f"unknown export option(s): {sorted(unknown)}")
        merged = {**cls().__dict__, **dict(values)}
        if merged["format"] not in FORMATS:
            raise ConfigError(f"export format must be one of {FORMATS}")
        for name, lo, hi in (("batch", 1, 64), ("opset", 13, 20)):
            value = merged[name]
            if type(value) is not int or not lo <= value <= hi:
                raise ConfigError(f"export {name} must be an integer in [{lo}, {hi}]")
        for name in ("rtol", "atol"):
            value = merged[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or not 0 <= value <= 1e-3:
                raise ConfigError(f"export {name} must be finite and between 0 and 1e-3")
            merged[name] = float(value)
        if merged["path"] is not None:
            if not isinstance(merged["path"], (str, Path)) or not str(merged["path"]).strip():
                raise ConfigError("export path must be a nonempty directory path")
            merged["path"] = Path(merged["path"]).expanduser()
        return cls(**merged)


def verify_export_bundle(directory: Any) -> dict:
    """Verify manifest member bytes before any checkpoint deserialization.

    A checksum detects corruption and binds package members, not publisher identity.
    Users still need a trusted source for the package and its manifest.
    """
    from . import ConfigError

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"invalid inference artifact manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA \
            or manifest.get("status") != "verified":
        raise ConfigError("unsupported or incomplete inference artifact")
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.pt" not in files or not files:
        raise ConfigError("inference artifact has no complete file inventory")
    if manifest.get("inference_checkpoint") != "model.pt":
        raise ConfigError("inference artifact must declare its numeric model.pt state")
    primary = manifest.get("primary_artifact")
    if not isinstance(primary, str) or primary not in files:
        raise ConfigError("primary inference artifact is not a verified member")
    root = directory.resolve()
    for name, expected in files.items():
        if not isinstance(name, str):
            raise ConfigError("inference artifact member name must be a string")
        member = directory / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or member.is_symlink() or not member.is_file()
                or not member.resolve().is_relative_to(root)):
            raise ConfigError(f"unsafe or missing inference artifact member: {name}")
        try:
            valid = (isinstance(expected, dict) and member.stat().st_size == expected.get("bytes")
                     and file_sha256(member) == expected.get("sha256"))
        except OSError as exc:
            raise ConfigError(f"cannot read inference artifact member: {name}") from exc
        if not valid:
            raise ConfigError(f"inference artifact integrity failed: {name}")
    return manifest


def resolve_export_model(model: Any) -> Any:
    """Resolve a bundle or its tensor artifact to the verified IPB checkpoint.

    Ordinary .pt checkpoints without a sibling manifest retain old semantics.
    Tensor-artifact arguments reload the native state contained in their bundle;
    NIO's scientific view/trajectory pipeline does not execute an ONNX session.
    """
    if not isinstance(model, (str, Path)):
        return model
    path = Path(model).expanduser()
    if path.is_dir() and (path / "manifest.json").is_file():
        verify_export_bundle(path)
        return str(path / "model.pt")
    if path.suffix in (".pt", ".pth") and path.is_file() \
            and _has_export_manifest(path.parent):
        manifest = verify_export_bundle(path.parent)
        if path.name != manifest["inference_checkpoint"]:
            from . import ConfigError

            raise ConfigError("checkpoint is not the bundle's declared native state")
        return str(path)
    if path.suffix in (".onnx", ".torchscript") and path.is_file():
        manifest = verify_export_bundle(path.parent)
        if path.name != manifest["primary_artifact"]:
            from . import ConfigError

            raise ConfigError("tensor artifact is not the bundle's declared primary output")
        return str(path.parent / "model.pt")
    return model


def _has_export_manifest(directory: Path) -> bool:
    """An unrelated historical manifest does not turn a .pt into our bundle."""
    from . import ConfigError

    path = directory / "manifest.json"
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ConfigError(f"invalid sibling artifact manifest: {path}") from exc
    return isinstance(document, dict) and document.get("schema") == SCHEMA
