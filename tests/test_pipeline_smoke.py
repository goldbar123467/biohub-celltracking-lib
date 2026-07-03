from __future__ import annotations

from biohub_ct.pipelines.submission_pipeline import run_submission_pipeline
from biohub_ct.submission.validator import validate_submission


def make_fake_zarr(path, shape=(2, 3, 4, 5)):
    meta = path / "0"
    meta.mkdir(parents=True)
    (meta / "zarr.json").write_text(
        "{"
        '"shape":[%s],'
        '"data_type":"uint16",'
        '"chunk_grid":{"configuration":{"chunk_shape":[1,3,4,5]}},'
        '"codecs":[]'
        "}" % ",".join(str(x) for x in shape),
        encoding="utf-8",
    )


def test_pipeline_writes_valid_submission_from_metadata_only_zarr(tmp_path):
    data_dir = tmp_path / "test"
    data_dir.mkdir()
    make_fake_zarr(data_dir / "sample.zarr")
    out = tmp_path / "submission.csv"

    run_submission_pipeline(data_dir=data_dir, output_path=out, debug=True)

    result = validate_submission(out, expected_datasets=["sample"])
    assert result.row_count >= 1

