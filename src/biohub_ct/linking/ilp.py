from __future__ import annotations


def ilp_available() -> bool:
    try:
        import pyscipopt  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True

