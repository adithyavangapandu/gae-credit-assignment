"""Generate the deterministic 24-run excluded PPO pilot matrix."""

import argparse

from gae_credit.cloud.config import load_gcp_config
from gae_credit.pilot import generate_matrix, load_pilot_spec, write_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", default="configs/pilot/study_v1.yaml")
    parser.add_argument("--gcp-config", default="configs/cloud/gcp.yaml")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--output", default="data/manifests/pilot_run_matrix.parquet")
    args = parser.parse_args()
    spec, base = load_pilot_spec(args.study_config)
    rows = generate_matrix(spec, base, load_gcp_config(args.gcp_config), args.image_uri)
    write_matrix(rows, args.output)
    print(f"wrote {len(rows)} deterministic pilot rows to {args.output}")


if __name__ == "__main__":
    main()
