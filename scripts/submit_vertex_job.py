"""Print a job specification without credentials, or submit one CPU Custom Job."""

import argparse
import json

from gae_credit.cloud.config import PHASES, load_gcp_config
from gae_credit.cloud.submit import build_job_spec, submit_job
from gae_credit.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--image-uri", required=True, help="Immutable Artifact Registry URI with @sha256 digest"
    )
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--run-id", help="Optional assertion of the generated identity")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print spec only; no SDK clients, ADC, or resource creation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest checkpoint after confirming prior job stopped",
    )
    args = parser.parse_args()
    config, gcp = load_config(args.config), load_gcp_config(args.gcp_config)
    spec = build_job_spec(
        config,
        gcp,
        args.phase,
        args.image_uri,
        args.service_account,
        args.run_id,
        resume=args.resume,
    )
    print(json.dumps(spec if args.dry_run else submit_job(spec, gcp, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
