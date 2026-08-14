"""Local web console for OpenGate."""

from __future__ import annotations

from pathlib import Path

__all__ = ["create_app", "repo_root"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def create_app():
    from opengate.ui.app import create_app as _create

    return _create()
