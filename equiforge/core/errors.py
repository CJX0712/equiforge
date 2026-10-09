"""EquiForge error catalogue (E100-E500)."""


class EquiError(Exception):
    """Base class for all EquiForge errors."""

    code = "E000"

    def __init__(self, message: str = "", code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConfigError(EquiError):
    code = "E100"


class DataError(EquiError):
    code = "E200"


class ModelError(EquiError):
    code = "E300"


class MethodError(EquiError):
    code = "E400"


class EvalError(EquiError):
    code = "E500"
