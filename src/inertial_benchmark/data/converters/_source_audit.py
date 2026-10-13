"""Catalogue unknown-frame sources without making up a valid IPB pose.

Archive discovery reads only the directory. Nothing is extracted, resampled,
or marked as supervision. Rejection goes through the normal conversion report.
"""

import zipfile
from pathlib import Path

from ._external_utils import sha256_file


def discover(source, leaf):
    source = Path(source)
    found = {}
    if source.is_dir():
        for path in sorted(source.glob("**/" + leaf)):
            relative = path.relative_to(source).as_posix()
            name = relative[:-len(leaf)].rstrip("/").replace("/", "__") or source.name
            found[name] = {"source_files": relative, "source_path": str(path)}
    archives = ([source] if source.is_file() and source.suffix == ".zip"
                else sorted(source.glob("*.zip")))
    for archive in archives:
        with zipfile.ZipFile(archive) as stream:
            members = [p for p in stream.namelist() if p.endswith("/" + leaf)]
        digest = sha256_file(archive)
        for member in members:
            parent = member[:-len(leaf)].rstrip("/")
            name = parent.replace("/", "__")
            found[name] = {"source_files": f"{archive.name}:{member}",
                           "source_archive_sha256": digest,
                           "source_archive_bytes": archive.stat().st_size}
    return found

