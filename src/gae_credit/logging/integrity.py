"""Checksummed bundles: the success marker commits a validated inventory."""

import hashlib
import json
from pathlib import Path

from gae_credit.storage.base import safe_key


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_checksums(path: Path) -> dict:
    entries = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.name not in {"checksums.json", "_SUCCESS"}:
            if file.is_symlink():
                raise ValueError("Artifact bundles cannot contain symlinks")
            data = file.read_bytes()
            entries[file.relative_to(path).as_posix()] = {"sha256": sha256(data), "size": len(data)}
    return {"version": 1, "files": entries}


def verify_checksums(path: Path):
    raw = (path / "checksums.json").read_bytes()
    marker = json.loads((path / "_SUCCESS").read_text())
    if marker != {"checksums_sha256": sha256(raw)}:
        raise ValueError("Success marker does not match checksum inventory")
    inventory = json.loads(raw)
    if inventory.get("version") != 1 or not inventory.get("files"):
        raise ValueError("Invalid checksum inventory")
    required = {
        "manifest.json",
        "resolved_config.yaml",
        "updates.parquet",
        "episodes.parquet",
        "evaluations.parquet",
        "summary.json",
        "validation_report.json",
    }
    if not required <= inventory["files"].keys():
        raise ValueError("Checksum inventory omits required artifacts")
    for key, entry in inventory["files"].items():
        file = path / safe_key(key)
        if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
            raise ValueError("Artifact escapes run directory")
        data = file.read_bytes()
        if {"sha256": sha256(data), "size": len(data)} != entry:
            raise ValueError(f"Artifact checksum mismatch: {key}")
