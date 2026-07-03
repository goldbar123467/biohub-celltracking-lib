from __future__ import annotations

import json

from biohub_ct.data.eda import write_eda_report
from biohub_ct.data.splits import write_splits
from biohub_ct.pipelines.evaluate import evaluate_fold


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


def make_graph_json(path, *, offset: int = 0):
    path.mkdir()
    (path / "graph.json").write_text(
        json.dumps(
            {
                "estimated_number_of_nodes": 4,
                "nodes": [
                    {"node_id": 1 + offset, "t": 0, "z": 1, "y": 2, "x": 2},
                    {"node_id": 2 + offset, "t": 1, "z": 1, "y": 2, "x": 3},
                ],
                "edges": [{"source_id": 1 + offset, "target_id": 2 + offset}],
            }
        ),
        encoding="utf-8",
    )


def make_train_dir(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    for name, offset in [("embA_0001", 0), ("embB_0001", 10)]:
        make_fake_zarr(train / f"{name}.zarr")
        make_graph_json(train / f"{name}.geff", offset=offset)
    return train


def test_eda_report_contains_graph_and_displacement_stats(tmp_path):
    train = make_train_dir(tmp_path)
    out = tmp_path / "eda.md"

    write_eda_report(train, out)

    text = out.read_text(encoding="utf-8")
    assert "embA_0001" in text
    assert "gt_nodes" in text
    assert "edge_disp_um_p50" in text


def test_write_splits_keeps_embryos_in_one_side(tmp_path):
    train = make_train_dir(tmp_path)
    out = tmp_path / "splits.json"

    splits = write_splits(train, out, k=2)

    assert "fold0" in splits
    fold0 = splits["fold0"]
    train_embryos = {name.split("_", 1)[0] for name in fold0["train"]}
    val_embryos = {name.split("_", 1)[0] for name in fold0["val"]}
    assert train_embryos.isdisjoint(val_embryos)
    assert json.loads(out.read_text(encoding="utf-8"))["fold0"] == fold0


def test_evaluate_fold_writes_baseline_report(tmp_path):
    train = make_train_dir(tmp_path)
    splits_path = tmp_path / "splits.json"
    write_splits(train, splits_path, k=2)
    out = tmp_path / "baseline.md"

    summary = evaluate_fold(
        data_dir=train,
        splits_path=splits_path,
        fold="fold0",
        output_path=out,
        pipeline="classical",
    )

    text = out.read_text(encoding="utf-8")
    assert summary.dataset_count >= 1
    assert "edge_tp" in text
    assert "node_count_ratio" in text

