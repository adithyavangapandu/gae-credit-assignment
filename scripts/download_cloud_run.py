import argparse
import json

from gae_credit.storage.download import download_run
from gae_credit.storage.gcs import GCSArtifactStore


def main():
    parser = argparse.ArgumentParser(description="Download and validate a completed GCS run")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--destination", required=True, help="New local path ending in the run ID")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            download_run(GCSArtifactStore(args.prefix, project=args.project), args.destination),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
