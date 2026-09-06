"""Composite facade; local records remain authoritative with optional cloud sinks."""

from gae_credit.cloud.config import run_id_for, source_digest
from gae_credit.logging.integrity import build_checksums, sha256
from gae_credit.logging.run_logger import LocalArtifactLogger, _write_json, validate_run
from gae_credit.storage.local import LocalArtifactStore


class RunLogger:
    def __init__(
        self,
        config,
        overwrite=False,
        seeds=None,
        *,
        context=None,
        resume_snapshot=None,
        artifact_logger=None,
        metrics_logger=None,
    ):
        self.context = {
            "artifact_protocol": 2,
            "phase": "local",
            "source_digest": source_digest(),
            "image_digest": None,
            **(context or {}),
        }
        self.context["run_id"] = run_id_for(config, self.context["phase"])
        self.local = LocalArtifactLogger(
            config,
            overwrite=overwrite,
            seeds=seeds,
            context=self.context,
            resume_snapshot=resume_snapshot,
        )
        self.path = self.local.path
        self.artifact_logger = artifact_logger
        self.metrics_logger = metrics_logger

    @property
    def manifest(self):
        return self.local.manifest

    def log_update(self, row):
        self.local.log_update(row)
        if self.metrics_logger:
            self.metrics_logger.log_update(row)

    def log_evaluation(self, row):
        self.local.log_evaluation(row)
        if self.metrics_logger:
            self.metrics_logger.log_evaluation(row)

    def log_episode(self, row):
        self.local.log_episode(row)

    def write_diagnostics(self, name, arrays):
        return self.local.write_diagnostics(name, arrays)

    def snapshot(self):
        return self.local.snapshot()

    def checkpoint(self, path, update_index):
        if self.artifact_logger:
            self.artifact_logger.checkpoint(path, update_index)

    def complete(self, summary):
        try:
            self.local.complete(summary)
            report = validate_run(self.path, require_success=False)
            _write_json(self.path / "validation_report.json", report)
            if self.metrics_logger:
                self.metrics_logger.close()
            _write_json(self.path / "checksums.json", build_checksums(self.path))
            if self.artifact_logger:
                self.artifact_logger.publish(self.path)
            LocalArtifactStore(self.path).mark_success(
                sha256((self.path / "checksums.json").read_bytes())
            )
        except BaseException as exc:
            self.fail(exc)
            raise
        finally:
            if self.artifact_logger:
                self.artifact_logger.release()

    def fail(self, error):
        self.local.fail(error)
        if self.metrics_logger:
            try:
                self.metrics_logger.close(failed=True)
            except Exception:
                pass
        if self.artifact_logger:
            self.artifact_logger.release()
