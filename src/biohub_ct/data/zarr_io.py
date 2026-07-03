from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biohub_ct.config import DEFAULT_SCALE


@dataclass
class LazyZarrVolume:
    path: Path
    shape: tuple[int, ...]
    chunks: tuple[int, ...] | None
    dtype: str
    scale: tuple[float, float, float] = DEFAULT_SCALE
    _array: Any | None = None

    @property
    def can_read_chunks(self) -> bool:
        return self._array is not None

    def read_frame(self, t: int) -> np.ndarray:
        if self._array is None:
            raise RuntimeError(
                "Zarr package is unavailable or chunks were not opened; only metadata is loaded"
            )
        return np.asarray(self._array[t])


def open_zarr_volume(path: Path | str) -> LazyZarrVolume:
    zarr_path = Path(path)
    try:
        import zarr  # type: ignore

        group = zarr.open_group(str(zarr_path), mode="r")
        arr = group["0"]
        return LazyZarrVolume(
            path=zarr_path,
            shape=tuple(int(x) for x in arr.shape),
            chunks=tuple(int(x) for x in arr.chunks) if arr.chunks else None,
            dtype=str(arr.dtype),
            scale=_parse_scale(dict(getattr(group, "attrs", {}))),
            _array=arr,
        )
    except ImportError:
        return _open_metadata_only(zarr_path)
    except Exception:
        return _open_metadata_only(zarr_path)


def _open_metadata_only(zarr_path: Path) -> LazyZarrVolume:
    meta_path = zarr_path / "0" / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find Zarr array metadata at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_shape = (
        meta.get("chunk_grid", {})
        .get("configuration", {})
        .get("chunk_shape")
    )
    dtype = meta.get("data_type") or meta.get("dtype") or "unknown"
    return LazyZarrVolume(
        path=zarr_path,
        shape=tuple(int(x) for x in meta["shape"]),
        chunks=tuple(int(x) for x in chunk_shape) if chunk_shape else None,
        dtype=str(dtype),
        scale=_parse_scale(_read_group_attrs(zarr_path)),
        _array=None,
    )


def _read_group_attrs(zarr_path: Path) -> dict[str, Any]:
    for rel in ("zarr.json", ".zattrs"):
        p = zarr_path / rel
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            return raw.get("attributes", raw)
    return {}


def _parse_scale(attrs: dict[str, Any]) -> tuple[float, float, float]:
    try:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform.get("type") == "scale":
            scale = transform["scale"][-3:]
            return (float(scale[0]), float(scale[1]), float(scale[2]))
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return DEFAULT_SCALE

