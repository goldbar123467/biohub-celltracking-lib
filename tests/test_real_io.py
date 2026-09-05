import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.zarr_io import open_zarr_volume


def test_real_zarr_strict_loading_and_missing_chunk(tmp_path):
    path = tmp_path / "image.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.attrs["multiscales"] = [
        {
            "axes": [{"name": n} for n in ("T", "Z", "Y", "X")],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1.625, 0.40625, 0.40625]}
                    ],
                }
            ],
        }
    ]
    group.create_array("0", data=np.ones((2, 4, 5, 6), dtype=np.uint16), chunks=(1, 4, 5, 6))
    volume = open_zarr_volume(path, require_complete_chunks=True)
    assert volume.read_frame(0).shape == (4, 5, 6)
    (path / "0/c/1/0/0/0").unlink()
    with pytest.raises(FileNotFoundError):
        volume.read_frame(1)
    with pytest.raises(IndexError):
        volume.read_frame(-1)
    (path / "0/zarr.json").write_text("{broken")
    with pytest.raises(ValueError):
        open_zarr_volume(path)


def test_real_geff_preserves_sparse_estimate_and_checks_coordinates(tmp_path):
    group = zarr.open_group(str(tmp_path / "labels.geff"), mode="w")
    group.attrs["geff"] = {
        "directed": True,
        "geff_version": "1.1",
        "extra": {"estimated_number_of_nodes": 1234},
    }
    group.create_array("nodes/ids", data=np.array([5, 8], dtype=np.uint64))
    for key, values in [("t", [0, 1]), ("z", [2, 2]), ("y", [3, 3]), ("x", [4, 4])]:
        group.create_array(f"nodes/props/{key}/values", data=np.array(values, dtype=np.int64))
    group.create_array("edges/ids", data=np.array([[5, 8]], dtype=np.uint64))
    graph, metadata = read_geff_graph(tmp_path / "labels.geff")
    assert graph.num_nodes == 2 and graph.edges_set() == {(5, 8)}
    assert metadata.estimated_number_of_nodes == 1234
    group["nodes/props/x/values"][0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        read_geff_graph(tmp_path / "labels.geff")


@pytest.mark.parametrize(
    "scales,axes", [([1, -1, 1, 1], ["t", "z", "y", "x"]), ([1, 1, 1, 1], ["t", "x", "y", "z"])]
)
def test_rejects_invalid_image_scale_or_axis_order(tmp_path, scales, axes):
    group = zarr.open_group(str(tmp_path / "image.zarr"), mode="w")
    group.create_array("0", data=np.ones((2, 2, 2, 2), dtype=np.uint16))
    group.attrs["multiscales"] = [
        {
            "axes": [{"name": n} for n in axes],
            "datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scales}]}
            ],
        }
    ]
    with pytest.raises(ValueError, match="metadata"):
        open_zarr_volume(tmp_path / "image.zarr")
