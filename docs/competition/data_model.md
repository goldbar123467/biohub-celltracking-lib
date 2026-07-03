# Data Model

## Layout

- `train/`: paired `{name}.zarr` images and `{name}.geff` sparse ground-truth tracking graphs.
- `test/`: `{name}.zarr` images only.
- Hidden test data are swapped in during notebook rerun.

## Images

- OME-Zarr/Zarr v3 volumes.
- Single array at path `0/`.
- Shape is `(T, Z, Y, X)`.
- Typical shape from Kaggle data page snapshot: `(100, 64, 256, 256)`.
- Typical chunks are one timepoint: `(1, 64, 256, 256)`.
- Physical voxel scale: `z=1.625`, `y=0.40625`, `x=0.40625` microns per voxel unless metadata says otherwise.

## Graphs

Train GEFF folders contain sparse lineage graph data:

- nodes have `node_id,t,z,y,x`;
- edges are directed `(source_id,target_id)`;
- annotations are sparse;
- `estimated_number_of_nodes` is important for node-count penalty calibration.

## Repo Representation

The dependency-light core uses `biohub_ct.data.schema.Graph`, `Node`, and `Edge`. Real GEFF reading is optional through the `geff` package; synthetic fallback tests use `.geff/graph.json`.

