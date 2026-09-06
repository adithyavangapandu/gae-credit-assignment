"""Local-only run logging with an explicit artifact contract."""

from __future__ import annotations

import json
import math
import platform
import re
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gae_credit.logging.schemas import SCHEMA_VERSION, TABLE_SCHEMAS

if TYPE_CHECKING:
    from gae_credit.config import StudyConfig


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _git_metadata() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "numpy", "gymnasium", "PyYAML", "pyarrow", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _finite_json(value: Any) -> None:
    """Reject nonfinite summary/manifest numbers before writing any final status."""
    json.dumps(value, allow_nan=False)


class LocalArtifactLogger:
    """Write schema-checked tables; call ``complete`` or ``fail`` to close them.

    A run directory cannot be reused implicitly. Explicit overwrite only removes
    a real directory whose manifest identifies this exact run and config hash.
    """

    def __init__(
        self,
        config: StudyConfig,
        overwrite: bool = False,
        seeds: dict[str, int] | None = None,
        context: dict | None = None,
        resume_snapshot: dict | None = None,
    ) -> None:
        self.config = config
        self.context = context or {}
        self.run_id = self.context.get("run_id", config.run_id)
        self._rows = {name: [] for name in TABLE_SCHEMAS}
        self.path = Path(config.logging.output_dir).expanduser() / self.run_id
        self._closed = False
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._counts = dict.fromkeys(TABLE_SCHEMAS, 0)
        if self.path.exists() or self.path.is_symlink():
            if not overwrite and resume_snapshot is None:
                raise FileExistsError(f"Run already exists: {self.path}. Use explicit overwrite.")
            if self.path.is_symlink() or not self.path.is_dir():
                raise ValueError(f"Refusing to overwrite a symlink or non-directory: {self.path}")
            try:
                previous = json.loads((self.path / "manifest.json").read_text())
            except (OSError, ValueError) as error:
                raise ValueError(
                    "Refusing to overwrite a directory without a valid manifest"
                ) from error
            if (
                previous.get("run_id") != self.run_id
                or previous.get("config_hash") != config.config_hash()
            ):
                raise ValueError("Refusing to overwrite a directory belonging to another run")
            if resume_snapshot is not None and previous.get("status") == "completed":
                raise ValueError("Cannot resume a completed local run")
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        (self.path / "diagnostics").mkdir()
        (self.path / "checkpoints").mkdir()
        (self.path / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=True)
        )
        commit, dirty = _git_metadata()
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_hash": config.config_hash(),
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "git_commit": commit,
            "git_dirty": dirty,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "seed": config.seed,
            "seeds": {name: int(seed) for name, seed in (seeds or {}).items()},
            "actor_horizon": config.estimator.actor_horizon,
            "critic_horizon": config.estimator.critic_horizon,
        }
        self.manifest.update(self.context)
        _write_json(self.path / "manifest.json", self.manifest)
        for name, schema in TABLE_SCHEMAS.items():
            self._writers[name] = pq.ParquetWriter(
                self.path / f"{name}.parquet", schema, compression="zstd"
            )

        if resume_snapshot is not None:
            for name, rows in resume_snapshot["rows"].items():
                for row in rows:
                    self._log(name, row)
            for name, data in resume_snapshot["diagnostics"].items():
                if not re.fullmatch(r"[A-Za-z0-9_-]+\.npz", name):
                    raise ValueError("Invalid checkpoint diagnostic filename")
                (self.path / "diagnostics" / name).write_bytes(data)

    def snapshot(self) -> dict:
        from copy import deepcopy

        return {
            "rows": deepcopy(self._rows),
            "diagnostics": {
                p.name: p.read_bytes() for p in sorted((self.path / "diagnostics").glob("*.npz"))
            },
        }

    def _log(self, name: str, row: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot log to a closed run")
        enriched = {"run_id": self.run_id, **row}
        if enriched["run_id"] != self.run_id:
            raise ValueError("Row run_id does not match this run")
        schema = TABLE_SCHEMAS[name]
        if set(enriched) != set(schema.names):
            missing = sorted(set(schema.names) - set(enriched))
            extra = sorted(set(enriched) - set(schema.names))
            raise ValueError(f"Invalid {name} fields: missing={missing}, extra={extra}")
        for field in schema:
            value = enriched[field.name]
            if value is None:
                if not field.nullable:
                    raise ValueError(f"{name}.{field.name} cannot be null")
            elif pa.types.is_floating(field.type) and not math.isfinite(float(value)):
                raise ValueError(f"{name}.{field.name} must be finite")
        if name == "episodes" and enriched["split"] not in {"train", "evaluation", "diagnostic"}:
            raise ValueError("Episode split must be train, evaluation, or diagnostic")
        table = pa.Table.from_pylist([enriched], schema=schema)
        self._writers[name].write_table(table)
        self._counts[name] += 1
        self._rows[name].append(enriched)

    def log_update(self, row: dict[str, Any]) -> None:
        self._log("updates", row)

    def log_episode(self, row: dict[str, Any]) -> None:
        self._log("episodes", row)

    def log_evaluation(self, row: dict[str, Any]) -> None:
        self._log("evaluations", row)

    def write_diagnostics(self, name: str, arrays: dict[str, np.ndarray]) -> Path:
        if self._closed:
            raise RuntimeError("Cannot log to a closed run")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Diagnostic name must be a simple filename stem")
        values = {key: np.asarray(value) for key, value in arrays.items()}
        for key, value in values.items():
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Diagnostic {key} must be a finite numeric array")
        path = self.path / "diagnostics" / f"{name}.npz"
        np.savez_compressed(path, **values)
        return path

    def _close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._closed = True

    def complete(self, summary: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Run has already been closed")
        result = {"run_id": self.run_id, "config_hash": self.config.config_hash(), **summary}
        if result["run_id"] != self.run_id or result["config_hash"] != self.config.config_hash():
            raise ValueError("Summary identity does not match this run")
        _finite_json(result)
        self._close()
        _write_json(self.path / "summary.json", result)
        self.manifest.update(status="completed", finished_at=_now(), row_counts=self._counts)
        _write_json(self.path / "manifest.json", self.manifest)

    def fail(self, error: BaseException | str) -> None:
        if not self._closed:
            self._close()
        self.manifest.update(status="failed", finished_at=_now(), error=str(error))
        _write_json(self.path / "manifest.json", self.manifest)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_table(path: Path, name: str, run_id: str) -> pa.Table:
    table = pq.read_table(path / f"{name}.parquet")
    schema = TABLE_SCHEMAS[name]
    _require(table.schema.equals(schema, check_metadata=True), f"{name}: schema mismatch")
    for field in schema:
        column = table[field.name]
        if not field.nullable:
            _require(column.null_count == 0, f"{name}.{field.name}: unexpected nulls")
        if pa.types.is_floating(field.type):
            _require(
                bool(np.isfinite(column.to_numpy()).all()), f"{name}.{field.name}: nonfinite values"
            )
    _require(
        all(item == run_id for item in table["run_id"].to_pylist()), f"{name}: run_id mismatch"
    )
    return table


def validate_run(run_dir: str | Path, *, require_success: bool = True) -> dict[str, Any]:
    """Validate a completed run and return a compact, JSON-serializable report.

    Raises ``ValueError`` on contract violations; malformed or missing input
    files may additionally raise the underlying file/parsing exception.
    """
    from gae_credit.config import load_config

    path = Path(run_dir)
    for name in (
        "manifest.json",
        "resolved_config.yaml",
        "updates.parquet",
        "episodes.parquet",
        "evaluations.parquet",
        "summary.json",
    ):
        _require((path / name).is_file(), f"Missing required file: {name}")
    for name in ("diagnostics", "checkpoints"):
        _require((path / name).is_dir(), f"Missing required directory: {name}")
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    _finite_json(manifest)
    _finite_json(summary)
    config = load_config(path / "resolved_config.yaml")
    from gae_credit.cloud.config import run_id_for

    expected_run_id = run_id_for(config, manifest.get("phase", "local"))
    _require(manifest.get("status") == "completed", "Run is not completed")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unknown schema version")
    _require(manifest.get("run_id") == expected_run_id, "Manifest run_id does not match config")
    _require(path.name == expected_run_id, "Run directory name does not match config")
    _require(manifest.get("config_hash") == config.config_hash(), "Manifest config hash mismatch")
    _require(summary.get("run_id") == expected_run_id, "Summary run_id mismatch")
    _require(summary.get("config_hash") == config.config_hash(), "Summary config hash mismatch")
    for field in ("actor_horizon", "critic_horizon"):
        _require(manifest.get(field) == getattr(config.estimator, field), f"Incorrect {field}")
    _require(manifest.get("seed") == config.seed, "Manifest seed mismatch")
    for field in (
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "dependency_versions",
        "seeds",
    ):
        _require(field in manifest, f"Missing manifest field: {field}")
    tables = {name: _validate_table(path, name, expected_run_id) for name in TABLE_SCHEMAS}
    updates = tables["updates"].to_pylist()
    episodes = tables["episodes"].to_pylist()
    evaluations = tables["evaluations"].to_pylist()
    rollout_size = config.training.episodes_per_rollout * config.environment.max_steps
    total_steps = config.training.total_env_steps
    expected_updates = total_steps // rollout_size
    _require(len(updates) == expected_updates and bool(updates), "Incorrect update row count")
    _require(
        [row["update_index"] for row in updates] == list(range(1, expected_updates + 1)),
        "Update indices must be unique and consecutive starting at one",
    )
    _require(
        [row["env_steps"] for row in updates]
        == list(range(rollout_size, total_steps + 1, rollout_size)),
        "Update environment steps do not match rollout increments or final step count",
    )
    _require(
        all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(updates, updates[1:])),
        "Update elapsed time decreases",
    )
    _require(
        [row["episode_id"] for row in episodes] == list(range(len(episodes))),
        "Episode IDs must be globally unique and consecutive starting at zero",
    )
    _require(
        all(row["split"] in {"train", "evaluation", "diagnostic"} for row in episodes),
        "Unknown episode split",
    )
    _require(
        all(row["length"] == config.environment.max_steps for row in episodes),
        "Episode length does not match finite task horizon",
    )
    for row in episodes:
        _require(0 <= row["upright_fraction"] <= 1, "Upright fraction is outside [0, 1]")
        _require(0 <= row["longest_upright_streak"] <= row["length"], "Invalid upright streak")
        first = row["first_upright_timestep"]
        _require(first is None or 1 <= first <= row["length"], "Invalid first upright timestep")
        _require(0 <= row["env_steps"] <= total_steps, "Invalid episode environment step count")
    train_episodes = [row for row in episodes if row["split"] == "train"]
    _require(
        len(train_episodes) == total_steps // config.environment.max_steps,
        "Incorrect training episode count",
    )
    _require(
        [row["env_steps"] for row in train_episodes]
        == list(range(config.environment.max_steps, total_steps + 1, config.environment.max_steps)),
        "Training episode environment steps are not consecutive",
    )
    for update in updates:
        group = [row for row in train_episodes if row["update_index"] == update["update_index"]]
        _require(
            len(group) == config.training.episodes_per_rollout, "Incorrect episodes per update"
        )
        for metric in ("train_return", "base_dense_return"):
            _require(
                bool(np.isclose(update[f"mean_{metric}"], np.mean([row[metric] for row in group]))),
                f"Update mean_{metric} does not match episode rows",
            )
    interval = config.evaluation.interval_env_steps
    expected_eval_steps = sorted({0, total_steps, *range(interval, total_steps + 1, interval)})
    _require(
        [row["env_steps"] for row in evaluations] == expected_eval_steps,
        "Evaluation steps do not match initial/interval/final schedule",
    )
    _require(
        [row["eval_index"] for row in evaluations] == list(range(len(evaluations))),
        "Evaluation indices must be unique and consecutive starting at zero",
    )
    evaluation_episodes = [row for row in episodes if row["split"] == "evaluation"]
    _require(
        len(evaluation_episodes) == len(evaluations) * config.evaluation.episodes,
        "Incorrect evaluation episode count",
    )
    for evaluation in evaluations:
        group = [row for row in evaluation_episodes if row["env_steps"] == evaluation["env_steps"]]
        _require(
            len(group) == evaluation["episodes"] == config.evaluation.episodes,
            "Evaluation episode row count mismatch",
        )
        for aggregate, column in (
            ("mean_train_return", "train_return"),
            ("mean_base_dense_return", "base_dense_return"),
            ("stable_success_rate", "stable_success"),
            ("mean_upright_fraction", "upright_fraction"),
        ):
            _require(
                bool(np.isclose(evaluation[aggregate], np.mean([row[column] for row in group]))),
                f"Evaluation {aggregate} does not match episode rows",
            )
    for key, expected in (
        ("total_env_steps", total_steps),
        ("updates", len(updates)),
        ("episodes", len(episodes)),
        ("evaluation_count", len(evaluations)),
    ):
        _require(summary.get(key) == expected, f"Incorrect summary {key}")
    _require(summary.get("actor_parameters_changed") is True, "No actor parameter update recorded")
    _require(
        manifest.get("row_counts") == {name: len(table) for name, table in tables.items()},
        "Manifest table counts mismatch",
    )
    diagnostic_files = list((path / "diagnostics").glob("*.npz"))
    if config.logging.save_diagnostics:
        _require(bool(diagnostic_files), "Diagnostics enabled but no diagnostic arrays were saved")
    for diagnostic in diagnostic_files:
        with np.load(diagnostic, allow_pickle=False) as arrays:
            for key in arrays.files:
                _require(bool(np.isfinite(arrays[key]).all()), f"Nonfinite diagnostic: {key}")
    if config.logging.save_checkpoint:
        _require(any((path / "checkpoints").iterdir()), "Checkpoint enabled but none was saved")
    if manifest.get("phase", "local") != "local":
        _require(manifest.get("vertex_run_id") == expected_run_id, "Vertex run ID mismatch")
        _require(
            str(manifest.get("artifact_prefix", "")).endswith("/run_id=" + expected_run_id),
            "GCS prefix run ID mismatch",
        )
        _require(
            bool(re.fullmatch(r"sha256:[0-9a-f]{64}", manifest.get("image_digest", ""))),
            "Missing immutable image digest",
        )
    if manifest.get("artifact_protocol") == 2 and require_success:
        from gae_credit.logging.integrity import verify_checksums

        verify_checksums(path)
        report = json.loads((path / "validation_report.json").read_text())
        _require(
            report.get("valid") is True and report.get("run_id") == expected_run_id,
            "Invalid validation report",
        )
    return {
        "valid": True,
        "run_id": expected_run_id,
        "total_env_steps": total_steps,
        "updates": len(updates),
        "episodes": len(episodes),
        "evaluation_count": len(evaluations),
        "actor_horizon": config.estimator.actor_horizon,
        "critic_horizon": config.estimator.critic_horizon,
    }
