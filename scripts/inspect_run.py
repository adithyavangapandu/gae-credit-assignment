"""Print run metadata, table sizes and the last update without loading a model."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    tables = {
        name: pq.read_table(args.run_dir / f"{name}.parquet")
        for name in ("updates", "episodes", "evaluations")
    }
    summary_file = args.run_dir / "summary.json"
    print(
        json.dumps(
            {
                "manifest": manifest,
                "rows": {name: len(table) for name, table in tables.items()},
                "last_update": tables["updates"]
                .slice(max(len(tables["updates"]) - 1, 0))
                .to_pylist(),
                "summary": json.loads(summary_file.read_text()) if summary_file.exists() else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
