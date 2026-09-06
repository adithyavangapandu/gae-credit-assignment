"""Opt-in cloud adapters around the same local trainer."""

import json
import os
import re
import tempfile
from pathlib import Path

from gae_credit.cloud.config import gcp_from_dict, image_digest, run_id_for, source_digest
from gae_credit.logging.gcs import GCSArtifactLogger
from gae_credit.logging.vertex import VertexMetricsLogger
from gae_credit.storage.gcs import GCSArtifactStore


def run_cloud(config, *, resume_latest=False, stop_after_update=None):
    from gae_credit.train import run_training

    gcp = gcp_from_dict(json.loads(os.environ["GAE_GCP_CONFIG"])).validate(deployed=True)
    phase = os.environ["GAE_PHASE"]
    run_id = run_id_for(config, phase)
    prefix = gcp.run_prefix(config, phase)
    if os.environ.get("GAE_RUN_ID") != run_id or os.environ.get("GAE_GCS_PREFIX") != prefix:
        raise ValueError(
            "Deployment run ID/prefix does not match resolved experiment configuration"
        )
    digest = image_digest(os.environ["GAE_IMAGE_URI"])
    context = {
        "phase": phase,
        "run_id": run_id,
        "image_digest": digest,
        "source_digest": source_digest(),
        "vertex_run_id": run_id,
        "artifact_prefix": prefix,
        "vertex_experiment": gcp.vertex_experiment,
        "gcp_project": gcp.project_id,
        "gcp_region": gcp.region,
        "machine_type": gcp.machine_type,
    }
    if re.fullmatch(r"[0-9a-f]{40}", os.environ.get("GAE_GIT_SHA", "")):
        context["git_commit"] = os.environ["GAE_GIT_SHA"]
        context["git_dirty"] = False
    identity = {key: context[key] for key in ("run_id", "source_digest", "image_digest")}
    identity["config_hash"] = config.config_hash()
    artifacts = GCSArtifactLogger(GCSArtifactStore(prefix, project=gcp.project_id), identity)
    metrics = None
    try:
        with tempfile.TemporaryDirectory(prefix="gae-resume-") as temp:
            resume_path = (
                artifacts.download_checkpoint(Path(temp) / "checkpoint.pt")
                if resume_latest
                else None
            )
            # Resume may refer to a Vertex run not created yet if initialization failed early.
            metrics = VertexMetricsLogger(gcp, run_id, config, resume=resume_latest)
            return run_training(
                config,
                context=context,
                resume_path=resume_path,
                artifact_logger=artifacts,
                metrics_logger=metrics,
                stop_after_update=stop_after_update,
            )
    except BaseException:
        if metrics is not None:
            try:
                metrics.close(failed=True)
            except Exception:
                pass
        raise
    finally:
        artifacts.release()
