"""Render the excluded pilot QC result as a self-contained HTML report."""

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/pilot_qc_data.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/pilot_quality_control.html"))
    args = parser.parse_args()
    qc = json.loads(args.input.read_text())
    runs = pd.DataFrame(qc["reports"])
    if not qc["complete"] or qc["valid_runs"] != 24:
        raise ValueError("Refusing to render a completion report for an incomplete pilot")
    summary = {
        "Expected runs": qc["expected_runs"],
        "Valid runs": qc["valid_runs"],
        "Missing runs": qc["missing_runs"],
        "Invalid runs": qc["invalid_runs"],
        "Median runtime (seconds)": round(qc["median_runtime_seconds"], 3),
        "Median checkpoint bytes/run": int(runs["checkpoint_bytes"].median()),
        "Total sparse rewarded episodes": int(
            runs.loc[runs["run_id"].str.contains("sparse"), "rewarded_training_episodes"].sum()
        ),
    }
    table = runs[
        [
            "run_id",
            "status",
            "runtime_seconds",
            "checkpoint_bytes",
            "upright_entries",
            "rewarded_training_episodes",
        ]
    ].to_html(index=False, border=0)
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Excluded Pilot Quality Control</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}}
.ok{{color:#176b2c;font-weight:700}}</style></head><body>
<h1>Excluded Pilot Quality Control</h1>
<p class="ok">PASS — all 24 excluded pilot runs satisfy automated quality control.</p>
<p>Pilot seeds 100–102 are excluded from confirmatory inference. This report makes no
reward or horizon performance comparison.</p>
<h2>Completeness, runtime, and feasibility</h2>{pd.Series(summary).to_frame("value").to_html(border=0)}
<h2>Run-level validation</h2>{table}
<h2>Checks applied</h2><p>{html.escape("Finite schemas and metrics; monotonic 250,000-step progress; 250 updates; expected evaluation cadence; positive action standard deviation; sparse reward discovery; delayed payout timing and discounted conservation; seed-consistent initialization; checkpoints; and checksummed GCS artifacts.")}</p>
<h2>Frozen decision</h2><p>The operationally validated global settings are frozen in
<code>configs/confirmatory/study_v1.yaml</code>. Detailed decisions are in
<code>docs/pilot_decisions.md</code>.</p></body></html>"""
    args.output.write_text(body)
    print(args.output)


if __name__ == "__main__":
    main()
