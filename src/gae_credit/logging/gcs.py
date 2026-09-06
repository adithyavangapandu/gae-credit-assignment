"""Publish authoritative artifacts with exclusive writer leases and a final marker."""

import json
import uuid
from pathlib import Path

from gae_credit.logging.integrity import sha256


class GCSArtifactLogger:
    def __init__(self, store, identity: dict):
        self.store = store
        self.lease = {"attempt": uuid.uuid4().hex, **identity}
        if store.exists("_SUCCESS"):
            raise FileExistsError("Cloud run is already complete")
        store.write_json("_LEASE.json", self.lease)
        try:
            if store.exists("_IDENTITY.json"):
                if json.loads(store.read_bytes("_IDENTITY.json")) != identity:
                    raise ValueError("Cloud prefix belongs to a different config/code/image")
            else:
                store.write_json("_IDENTITY.json", identity)
        except BaseException:
            self.release()
            raise

    def release(self):
        if (
            self.store.exists("_LEASE.json")
            and json.loads(self.store.read_bytes("_LEASE.json")) == self.lease
        ):
            self.store.delete("_LEASE.json")

    def checkpoint(self, path, update_index):
        data = Path(path).read_bytes()
        digest = sha256(data)
        key = f"recovery/update-{update_index:06d}-{digest[:12]}.pt"
        if not self.store.exists(key):
            self.store.upload_checkpoint(path, key)
        self.store.write_json(
            "_LATEST_CHECKPOINT.json", {"key": key, "sha256": digest}, overwrite=True
        )

    def download_checkpoint(self, destination):
        metadata = json.loads(self.store.read_bytes("_LATEST_CHECKPOINT.json"))
        if not metadata["key"].startswith("recovery/"):
            raise ValueError("Invalid recovery checkpoint key")
        data = self.store.read_bytes(metadata["key"])
        if sha256(data) != metadata["sha256"]:
            raise ValueError("Recovery checkpoint checksum mismatch")
        Path(destination).write_bytes(data)
        return Path(destination)

    def publish(self, path):
        inventory = json.loads((path / "checksums.json").read_text())
        for key in [*inventory["files"], "checksums.json"]:
            data = (path / key).read_bytes()
            # Resuming after an interrupted upload can safely reuse identical objects.
            if self.store.exists(key) and self.store.read_bytes(key) == data:
                continue
            self.store.write_bytes(key, data, overwrite=True)
            if self.store.read_bytes(key) != data:
                raise ValueError(f"Uploaded artifact failed read-back validation: {key}")
        self.store.mark_success(sha256((path / "checksums.json").read_bytes()))
