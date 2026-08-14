"""Typed errors with stable GRASPSCOPE_E0xx codes."""
from __future__ import annotations


class GraspScopeError(Exception):
    code: str = "GRASPSCOPE_E000"
    http_status: int = 400

    def __init__(self, message: str, *, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(GraspScopeError):
    code = "GRASPSCOPE_E001"


class MissingFileError(GraspScopeError):
    code = "GRASPSCOPE_E002"


class MissingWeightsError(GraspScopeError):
    code = "GRASPSCOPE_E003"


class QCStrictError(GraspScopeError):
    code = "GRASPSCOPE_E004"


class ProfileError(GraspScopeError):
    code = "GRASPSCOPE_E005"


class PathSafetyError(GraspScopeError):
    code = "GRASPSCOPE_E006"
