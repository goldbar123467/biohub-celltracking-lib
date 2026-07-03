from __future__ import annotations


def napari_available() -> bool:
    try:
        import napari  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True

