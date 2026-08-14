"""Safe path resolution under repo root (no traversal outside)."""
from __future__ import annotations

from pathlib import Path

from graspscope.errors import PathSafetyError


def safe_under_root(path: str | Path, root: Path) -> Path:
    root = root.resolve()
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise PathSafetyError(f"path escapes repo root: {p}") from e
    return p
