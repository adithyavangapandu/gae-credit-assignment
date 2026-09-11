"""Checksummed, analysis-only mirrors of completed cloud runs."""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from gae_credit.cloud.config import image_digest
from gae_credit.logging.integrity import sha256
from gae_credit.logging.schemas import TABLE_SCHEMAS
from gae_credit.logging.trajectories import TRAJECTORY_SCHEMA
from gae_credit.storage.base import safe_key

ROOT_FILES = {
    "config.json",
    "manifest.json",
    "metadata.json",
    "resolved_config.yaml",
    "summary.json",
    "updates.parquet",
    "episodes.parquet",
    "evaluations.parquet",
    "validation_report.json",
}


def analysis_keys(inventory):
    return sorted(
        key
        for key in inventory["files"]
        if key in ROOT_FILES
        or key.startswith("diagnostics/")
        or key.startswith("diagnostic_trajectories/")
    )


def validate_analysis_bundle(path, row):
    path = Path(path)
    bundle = json.loads((path / "_ANALYSIS_BUNDLE.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {
        "run_id": row["run_id"],
        "config_hash": row["config_hash"],
        "git_commit": row["git_sha"],
        "image_digest": image_digest(row["image_digest"]),
        "phase": "confirmatory",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Analysis bundle identity mismatch")
    for key, entry in bundle["selected_files"].items():
        data = (path / safe_key(key)).read_bytes()
        if {"sha256": sha256(data), "size": len(data)} != entry:
            raise ValueError(f"Analysis bundle checksum mismatch: {key}")
    expected_rows = {"updates": 1000, "evaluations": 101, "episodes": 7020}
    for name, schema in TABLE_SCHEMAS.items():
        table = pq.read_table(path / f"{name}.parquet")
        if (
            not table.schema.equals(schema, check_metadata=True)
            or table.num_rows != expected_rows[name]
        ):
            raise ValueError(f"Analysis bundle {name} table mismatch")
    diagnostics = list((path / "diagnostics").glob("update-*.npz"))
    trajectories = list((path / "diagnostic_trajectories").glob("checkpoint-*.parquet"))
    if len(diagnostics) != 1000 or len(trajectories) != 20:
        raise ValueError("Analysis bundle diagnostic file count mismatch")
    for trajectory in trajectories:
        table = pq.read_table(trajectory)
        if (
            not table.schema.equals(TRAJECTORY_SCHEMA, check_metadata=True)
            or table.num_rows != 4000
        ):
            raise ValueError(f"Analysis trajectory mismatch: {trajectory.name}")
    return {
        "run_id": row["run_id"],
        "status": "valid",
        "selected_files": len(bundle["selected_files"]),
    }


def download_analysis_bundle(store, destination, row, *, workers=24):
    destination = Path(destination)
    marker = store.read_bytes("_SUCCESS")
    checksums = store.read_bytes("checksums.json")
    if json.loads(marker) != {"checksums_sha256": sha256(checksums)}:
        raise ValueError("Remote success marker checksum mismatch")
    inventory = json.loads(checksums)
    keys = analysis_keys(inventory)
    if not ROOT_FILES <= set(keys):
        raise ValueError("Remote inventory omits analysis tables or metadata")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".analysis-download-") as temp:
        path = Path(temp) / destination.name
        path.mkdir()

        def fetch(key):
            data = store.read_bytes(key)
            entry = inventory["files"][key]
            if {"sha256": sha256(data), "size": len(data)} != entry:
                raise ValueError(f"Remote analysis checksum mismatch: {key}")
            target = path / safe_key(key)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(fetch, keys))
        bundle = {
            "version": 1,
            "source_checksums_sha256": sha256(checksums),
            "selected_files": {key: inventory["files"][key] for key in keys},
            "excluded_prefixes": ["checkpoints/", "recovery/"],
        }
        (path / "_ANALYSIS_BUNDLE.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n"
        )
        validate_analysis_bundle(path, row)
        shutil.move(str(path), destination)
    return validate_analysis_bundle(destination, row)
