"""Typed errors with stable OPENGATE_E0xx codes."""
from __future__ import annotations


class OpenGateError(Exception):
    code: str = "OPENGATE_E000"
    http_status: int = 400

    def __init__(self, message: str, *, code: str | None = None):
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(OpenGateError):
    code = "OPENGATE_E001"


class MissingFileError(OpenGateError):
    code = "OPENGATE_E002"


class MissingWeightsError(OpenGateError):
    code = "OPENGATE_E003"


class QCStrictError(OpenGateError):
    code = "OPENGATE_E004"


class ProfileError(OpenGateError):
    code = "OPENGATE_E005"


class PathSafetyError(OpenGateError):
    code = "OPENGATE_E006"
