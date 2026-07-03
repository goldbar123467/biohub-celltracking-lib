from __future__ import annotations

from pathlib import Path

from biohub_ct.data.paths import discover_datasets
from biohub_ct.pipelines.baseline_classical import run_classical_baseline
from biohub_ct.submission.validator import validate_submission
from biohub_ct.submission.writer import write_submission


def run_submission_pipeline(
    *,
    data_dir: Path | str,
    output_path: Path | str,
    debug: bool = False,
):
    records = discover_datasets(data_dir, require_geff=False)
    if not records:
        raise FileNotFoundError(f"No .zarr datasets found under {data_dir}")
    graphs = {
        record.name: run_classical_baseline(record, debug=debug)
        for record in records
    }
    out = Path(output_path)
    write_submission(graphs, out)
    validate_submission(out, expected_datasets=[r.name for r in records])
    return out

