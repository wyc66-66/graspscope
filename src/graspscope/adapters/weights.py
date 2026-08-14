"""Resolve model weight paths from config / env."""
from __future__ import annotations

import os
from pathlib import Path

from graspscope.errors import MissingWeightsError


def resolve_weights(config: dict, default_name: str) -> str:
    env_dir = os.environ.get("GRASPSCOPE_WEIGHTS")
    weights = config.get("weights") or config.get("model") or default_name
    path = Path(weights)
    if path.is_file():
        return str(path.resolve())
    if env_dir:
        cand = Path(env_dir) / path.name
        if cand.is_file():
            return str(cand.resolve())
    # Allow ultralytics auto-download names (*.pt without path) only if not forced local
    if config.get("require_local_weights") and not path.is_file():
        raise MissingWeightsError(
            f"weights not found: {weights}. Set path or GRASPSCOPE_WEIGHTS."
        )
    return str(weights)
