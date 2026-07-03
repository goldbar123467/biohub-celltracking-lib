# Submission Format

The file must be named `submission.csv` and contain exactly:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

## Node Rows

- `row_type=node`
- `node_id,t,z,y,x` are nonnegative integers.
- `source_id=-1`
- `target_id=-1`

## Edge Rows

- `row_type=edge`
- `source_id,target_id` are nonnegative node IDs within the same dataset.
- `node_id=-1,t=-1,z=-1,y=-1,x=-1`

## Invariants

- `id` values are consecutive integers starting at 0.
- Dataset names match test folder stems without `.zarr`.
- Every test dataset appears.
- No blanks, NaNs, float-looking integers, or negative coordinates.
- If a detector returns no nodes, the current fallback emits one documented node so the dataset is represented.

Validator:

```bash
python -m biohub_ct.submission.validator submission.csv
```

