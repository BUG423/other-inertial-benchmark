"""Repository-wide documentation consistency checks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "tel:")
IGNORED_DIRECTORIES = {".git", ".local", ".venv", "venv", "__pycache__", "node_modules"}
GENERATED_ROOT_DIRECTORIES = {"datasets", "external", "runs", "reports", "build", "dist"}


def _documents():
    """Check maintained Markdown, excluding generated/data/vendor environments."""
    for parent, directories, files in os.walk(ROOT):
        excluded = IGNORED_DIRECTORIES | (GENERATED_ROOT_DIRECTORIES
                                         if Path(parent) == ROOT else set())
        directories[:] = sorted(d for d in directories if d not in excluded)
        for name in sorted(files):
            if name.endswith(".md"):
                yield Path(parent) / name


def _local_target(raw_target: str) -> str | None:
    """Return a local link target, excluding URLs and page fragments."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        # Markdown permits an optional title after the destination.
        target = target.split(maxsplit=1)[0]

    if not target or target.startswith("#") or target.startswith(EXTERNAL_SCHEMES):
        return None
    return unquote(target.split("#", 1)[0].split("?", 1)[0])


def test_relative_markdown_links_resolve() -> None:
    missing: list[str] = []

    for document in _documents():
        for match in MARKDOWN_LINK.finditer(document.read_text(encoding="utf-8")):
            target = _local_target(match.group(1))
            if target is None:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")

    assert not missing, "Broken relative Markdown links:\n" + "\n".join(missing)


def test_public_markdown_uses_tables_or_flows_instead_of_directory_trees() -> None:
    violations = [str(document.relative_to(ROOT)) for document in _documents()
                  if any(char in document.read_text(encoding="utf-8")
                         for char in ("\u251c", "\u2514"))]
    assert not violations, "Replace directory trees with tables/flows: " + ", ".join(violations)
