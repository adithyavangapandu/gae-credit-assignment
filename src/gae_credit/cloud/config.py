"""Non-secret deployment configuration and portable experiment identities."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from gae_credit.config import StudyConfig

PHASES = ("smoke", "pilot", "confirmatory", "replication")


def run_id_for(config: StudyConfig, phase: str = "local") -> str:
    if phase == "local":
        return config.run_id
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    reward = config.reward.kind
    if reward == "delayed":
        reward += str(config.reward.delay_block_size)
    h = config.estimator.horizon
    horizon = "full" if h == "full" else f"h{h:03d}"
    value = (
        f"{phase}-{config.algorithm}-{reward}-{horizon}-seed{config.seed}-"
        f"{config.config_hash()[:12]}"
    )
    if len(value) > 63:
        raise ValueError("Generated run ID exceeds Vertex label length limit")
    return value


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def image_digest(image_uri: str) -> str:
    if not re.fullmatch(
        r"[a-z0-9.-]+-docker\.pkg\.dev/[a-z0-9/_.-]+@sha256:[0-9a-f]{64}", image_uri
    ):
        raise ValueError(
            "image-uri must be an immutable Artifact Registry URI ending @sha256:<64 hex>"
        )
    return image_uri.rsplit("@", 1)[1]


@dataclass(frozen=True)
class GCPConfig:
    project_id: str = "your-project-id"
    region: str = "us-central1"
    artifact_repository: str = "gae-training"
    artifact_bucket: str = "gs://your-project-id-gae-credit-data"
    vertex_experiment: str = "gae-pendulum-v1"
    tensorboard: str | None = None
    machine_type: str = "n1-standard-4"
    accelerator: None = None
    prefix: str = "study_v1"
    timeout_seconds: int = 14400

    def validate(self, *, deployed: bool = False):
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", self.project_id):
            raise ValueError("Invalid GCP project ID")
        if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]", self.region):
            raise ValueError("Invalid GCP region")
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", self.artifact_bucket):
            raise ValueError("artifact_bucket must be gs://bucket with no object path")
        for name in ("artifact_repository", "vertex_experiment"):
            if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", getattr(self, name)):
                raise ValueError(f"Invalid {name}")
        if not re.fullmatch(r"[a-z0-9_-]+", self.prefix):
            raise ValueError("prefix must be one safe path component")
        if not re.fullmatch(r"[a-z][a-z0-9-]+", self.machine_type) or self.accelerator is not None:
            raise ValueError("Day 4 supports a CPU machine with accelerator: null")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive integer")
        if self.tensorboard is not None and not re.fullmatch(
            r"projects/[a-z0-9-]+/locations/" + re.escape(self.region) + r"/tensorboards/[0-9]+",
            self.tensorboard,
        ):
            raise ValueError("tensorboard must be a full resource name in the selected region")
        if deployed and (
            "your-project" in self.project_id or "your-project" in self.artifact_bucket
        ):
            raise ValueError("Replace the project/bucket placeholders before cloud execution")
        if deployed and self.tensorboard is None:
            raise ValueError("Set tensorboard to log Vertex time-series metrics")
        return self

    def to_dict(self):
        return asdict(self)

    def run_prefix(self, config: StudyConfig, phase: str) -> str:
        run_id = run_id_for(config, phase)
        reward = config.reward.kind
        if reward == "delayed":
            reward += f"-{config.reward.delay_block_size}"
        h = config.estimator.horizon
        horizon = "full" if h == "full" else f"h{h:03d}"
        return (
            f"{self.artifact_bucket}/{self.prefix}/{phase}/algorithm={config.algorithm}/reward={reward}/"
            f"horizon={horizon}/seed={config.seed}/run_id={run_id}"
        )


def gcp_from_dict(raw: dict, *, environ=None) -> GCPConfig:
    if not isinstance(raw, dict):
        raise ValueError("GCP configuration must be a mapping")
    data = dict(raw)
    env = os.environ if environ is None else environ
    for variable, field in {
        "GAE_GCP_PROJECT": "project_id",
        "GAE_GCP_REGION": "region",
        "GAE_ARTIFACT_BUCKET": "artifact_bucket",
        "GAE_VERTEX_EXPERIMENT": "vertex_experiment",
        "GAE_VERTEX_TENSORBOARD": "tensorboard",
    }.items():
        if variable in env:
            data[field] = env[variable]
    try:
        return GCPConfig(**data).validate()
    except TypeError as exc:
        raise ValueError(f"Invalid GCP configuration: {exc}") from exc


def load_gcp_config(path, *, environ=None) -> GCPConfig:
    return gcp_from_dict(yaml.safe_load(Path(path).read_text()), environ=environ)
