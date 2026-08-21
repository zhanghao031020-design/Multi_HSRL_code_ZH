from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


def stable_config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted-or-unavailable"


def source_digest(project_root: Path | None = None) -> str:
    """Hash the executable Python sources and project metadata."""

    root = project_root or Path(__file__).resolve().parents[2]
    files = [root / "pyproject.toml"]
    files.extend(sorted((root / "src").rglob("*.py")))
    files.extend(sorted((root / "scripts").rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
