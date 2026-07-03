#!/usr/bin/env python
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.pipelines.submission_pipeline import run_submission_pipeline
from biohub_ct.submission.validator import validate_submission


def _make_fake_zarr(path: Path, shape: tuple[int, int, int, int] = (2, 3, 4, 5)) -> None:
    meta = path / "0"
    meta.mkdir(parents=True)
    meta.joinpath("zarr.json").write_text(
        "{"
        f'"shape":[{",".join(str(x) for x in shape)}],'
        '"data_type":"uint16",'
        '"chunk_grid":{"configuration":{"chunk_shape":[1,3,4,5]}},'
        '"codecs":[]'
        "}",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "test"
        data_dir.mkdir()
        _make_fake_zarr(data_dir / "sample.zarr")
        out = root / "submission.csv"
        run_submission_pipeline(data_dir=data_dir, output_path=out, debug=True)
        result = validate_submission(out, expected_datasets=["sample"])
        print(f"smoke ok: {result.row_count} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

