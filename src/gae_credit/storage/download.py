"""Fetch only checksummed artifacts from a committed run; validate before returning."""

import json
import shutil
import tempfile
from pathlib import Path

from gae_credit.logging.integrity import sha256
from gae_credit.logging.run_logger import validate_run
from gae_credit.storage.base import safe_key


def download_run(store, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    marker = store.read_bytes("_SUCCESS")
    checksum_bytes = store.read_bytes("checksums.json")
    if json.loads(marker) != {"checksums_sha256": sha256(checksum_bytes)}:
        raise ValueError("Remote success marker checksum mismatch")
    inventory = json.loads(checksum_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".download-") as temporary:
        path = Path(temporary) / destination.name
        path.mkdir()
        for key, entry in inventory["files"].items():
            key = safe_key(key)
            data = store.read_bytes(key)
            if {"sha256": sha256(data), "size": len(data)} != entry:
                raise ValueError(f"Download checksum mismatch: {key}")
            target = path / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (path / "checksums.json").write_bytes(checksum_bytes)
        (path / "_SUCCESS").write_bytes(marker)
        (path / "diagnostics").mkdir(exist_ok=True)
        (path / "checkpoints").mkdir(exist_ok=True)
        result = validate_run(path)
        shutil.move(str(path), destination)
    return result
