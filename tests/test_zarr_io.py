from __future__ import annotations

from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.zarr_io import open_zarr_volume


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


def test_open_zarr_volume_reads_metadata_without_loading_chunks(tmp_path):
    zarr_path = tmp_path / "sample.zarr"
    make_fake_zarr(zarr_path)

    volume = open_zarr_volume(zarr_path)

    assert volume.shape == (2, 3, 4, 5)
    assert volume.chunks == (1, 3, 4, 5)
    assert volume.scale == (1.625, 0.40625, 0.40625)


def test_discover_datasets_pairs_train_geff_and_test_zarr(tmp_path):
    make_fake_zarr(tmp_path / "a.zarr")
    (tmp_path / "a.geff").mkdir()
    make_fake_zarr(tmp_path / "b.zarr")

    train = discover_datasets(tmp_path, require_geff=True)
    test = discover_datasets(tmp_path, require_geff=False)

    assert [d.name for d in train] == ["a"]
    assert [d.name for d in test] == ["a", "b"]

