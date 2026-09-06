from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq


def safe_key(key: str) -> str:
    if (
        not key
        or "\\" in key
        or PurePosixPath(key).is_absolute()
        or any(part in ("", ".", "..") for part in key.split("/"))
    ):
        raise ValueError(f"Unsafe artifact key: {key!r}")
    return key


class ArtifactStore(ABC):
    @abstractmethod
    def write_bytes(self, key: str, data: bytes, *, overwrite: bool = False): ...

    @abstractmethod
    def read_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str): ...

    def write_json(self, key, value, *, overwrite=False):
        self.write_bytes(
            key,
            (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
            overwrite=overwrite,
        )

    def write_parquet(self, key, table: pa.Table, *, overwrite=False):
        stream = pa.BufferOutputStream()
        pq.write_table(table, stream, compression="zstd")
        self.write_bytes(key, stream.getvalue().to_pybytes(), overwrite=overwrite)

    def upload_checkpoint(self, path, key, *, overwrite=False):
        self.write_bytes(key, Path(path).read_bytes(), overwrite=overwrite)

    def mark_success(self, checksum_sha256: str):
        self.write_json("_SUCCESS", {"checksums_sha256": checksum_sha256})
