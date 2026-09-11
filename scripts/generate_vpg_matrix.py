"""Generate the checksum-frozen 60-run VPG replication manifest."""

import argparse

from gae_credit.cloud.config import load_gcp_config
from gae_credit.replication import generate_manifest, load_replication_spec, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", default="configs/replication/vpg_v1.yaml")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--output", default="data/manifests/vpg_replication_v1.parquet")
    args = parser.parse_args()
    spec, base = load_replication_spec(args.study_config)
    rows = generate_manifest(
        spec, base, load_gcp_config(args.gcp_config), args.image_uri, args.git_sha
    )
    digest = write_manifest(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}; sha256={digest}")


if __name__ == "__main__":
    main()
