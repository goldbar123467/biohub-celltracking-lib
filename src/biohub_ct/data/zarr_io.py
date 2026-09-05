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
    require_complete_chunks: bool = False

    @property
    def can_read_chunks(self) -> bool:
        return self._array is not None

    def read_frame(self, t: int) -> np.ndarray:
        if not 0 <= t < self.shape[0]:
            raise IndexError(f"Frame {t} outside {self.shape}")
        if self.require_complete_chunks:
            from itertools import product

            if self.chunks is None:
                raise ValueError("Strict chunk checks require chunk metadata")
            ranges = [range((n + c - 1) // c) for n, c in zip(self.shape[1:], self.chunks[1:])]
            for indices in product(*ranges):
                chunk = self.path / "0" / "c" / str(t // self.chunks[0])
                for index in indices:
                    chunk = chunk / str(index)
                if not chunk.is_file():
                    raise FileNotFoundError(f"Missing competition chunk: {chunk}")
        if self._array is None:
            raise RuntimeError(
                "Zarr package is unavailable or chunks were not opened; only metadata is loaded"
            )
        return np.asarray(self._array[t])


def open_zarr_volume(
    path: Path | str, *, allow_metadata_only: bool = False, require_complete_chunks: bool = False
) -> LazyZarrVolume:
    zarr_path = Path(path)
    if allow_metadata_only:
        return _open_metadata_only(zarr_path)
    try:
        import zarr  # type: ignore

        group = zarr.open_group(str(zarr_path), mode="r")
        arr = group["0"]
        volume = LazyZarrVolume(
            path=zarr_path,
            shape=tuple(int(x) for x in arr.shape),
            chunks=tuple(int(x) for x in arr.chunks) if arr.chunks else None,
            dtype=str(arr.dtype),
            scale=_parse_scale(dict(getattr(group, "attrs", {}))),
            _array=arr,
            require_complete_chunks=require_complete_chunks,
        )
        if len(volume.shape) != 4 or min(volume.shape) <= 0 or volume.dtype != "uint16":
            raise ValueError(
                f"Expected nonempty uint16 TZYX array, got {volume.shape} {volume.dtype}"
            )
        if not all(np.isfinite(s) and s > 0 for s in volume.scale):
            raise ValueError("Spatial scale must be positive and finite")
        if require_complete_chunks:
            meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
            encoding = meta.get("chunk_key_encoding", {})
            if (
                encoding.get("name") != "default"
                or encoding.get("configuration", {}).get("separator", "/") != "/"
            ):
                raise ValueError(
                    "Strict competition chunk checks require v3 default slash encoding"
                )
        return volume
    except ImportError as exc:
        raise ImportError(
            "Real image inference requires zarr; metadata-only mode must be explicit"
        ) from exc


def _open_metadata_only(zarr_path: Path) -> LazyZarrVolume:
    meta_path = zarr_path / "0" / "zarr.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find Zarr array metadata at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
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
            raw = json.loads(p.read_text(encoding="utf-8"))
            return raw.get("attributes", raw)
    return {}


def _parse_scale(attrs: dict[str, Any]) -> tuple[float, float, float]:
    if "multiscales" not in attrs:
        return DEFAULT_SCALE
    try:
        multiscale = attrs["multiscales"][0]
        axes = multiscale.get("axes")
        if axes is not None and [str(a["name"]).lower() for a in axes] != ["t", "z", "y", "x"]:
            raise ValueError("Expected TZYX axes")
        transform = multiscale["datasets"][0]["coordinateTransformations"][0]
        if transform.get("type") != "scale" or len(transform["scale"]) != 4:
            raise ValueError("Expected four-dimensional scale transform")
        scale = tuple(float(v) for v in transform["scale"][-3:])
        if not all(np.isfinite(v) and v > 0 for v in scale):
            raise ValueError("Spatial scale must be positive and finite")
        return scale
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Invalid image axis/scale metadata") from exc
