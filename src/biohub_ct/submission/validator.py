from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from biohub_ct.config import SUBMISSION_COLUMNS


class SubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    row_count: int
    datasets: tuple[str, ...]


def validate_submission(
    path: Path | str,
    *,
    expected_datasets: list[str] | tuple[str, ...] | None = None,
) -> ValidationResult:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SUBMISSION_COLUMNS:
            raise SubmissionError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
        rows = list(reader)
    ids = [_parse_int(row["id"], "id") for row in rows]
    if ids != list(range(len(rows))):
        raise SubmissionError("Submission id values must be consecutive integers starting at 0")
    datasets: set[str] = set()
    for i, row in enumerate(rows):
        _validate_row(row, i)
        datasets.add(row["dataset"])
    if expected_datasets:
        missing = set(expected_datasets) - datasets
        if missing:
            raise SubmissionError(f"Missing datasets in submission: {sorted(missing)}")
    return ValidationResult(csv_path, len(rows), tuple(sorted(datasets)))


def _validate_row(row: dict[str, str], row_index: int) -> None:
    for col in SUBMISSION_COLUMNS:
        value = row[col]
        if value is None or value == "":
            raise SubmissionError(f"Blank value in row {row_index}, column {col}")
        if value.lower() == "nan":
            raise SubmissionError(f"NaN value in row {row_index}, column {col}")
    if row["row_type"] not in {"node", "edge"}:
        raise SubmissionError(f"Invalid row_type {row['row_type']!r} in row {row_index}")
    int_fields = ["node_id", "t", "z", "y", "x", "source_id", "target_id"]
    values = {field: _parse_int(row[field], field) for field in int_fields}
    if row["row_type"] == "node":
        if values["source_id"] != -1 or values["target_id"] != -1:
            raise SubmissionError(f"Node row {row_index} must use -1 source/target placeholders")
        for field in ["node_id", "t", "z", "y", "x"]:
            if values[field] < 0:
                raise SubmissionError(f"Node row {row_index} has negative {field}")
    else:
        for field in ["node_id", "t", "z", "y", "x"]:
            if values[field] != -1:
                raise SubmissionError(f"Edge row {row_index} must use -1 for {field}")
        if values["source_id"] < 0 or values["target_id"] < 0:
            raise SubmissionError(f"Edge row {row_index} must have nonnegative endpoints")


def _parse_int(value: str, column: str) -> int:
    try:
        if math.isnan(float(value)):
            raise ValueError
        parsed = int(value)
    except ValueError as exc:
        raise SubmissionError(f"Column {column} must contain integer values") from exc
    if str(parsed) != str(value):
        raise SubmissionError(f"Column {column} must contain canonical integer values")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_csv")
    args = parser.parse_args(argv)
    result = validate_submission(args.submission_csv)
    print(f"valid submission: rows={result.row_count} datasets={len(result.datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

