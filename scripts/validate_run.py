"""Validate local training artifacts: python scripts/validate_run.py runs/RUN_ID."""

import argparse
import json
import sys

from gae_credit.logging import validate_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    try:
        report = validate_run(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
