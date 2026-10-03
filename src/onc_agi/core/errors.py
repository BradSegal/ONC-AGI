"""Domain errors carrying a stable :class:`ErrorCode` for every harness."""

from __future__ import annotations

from onc_agi.core.schema import ArenaErrorPayload, ErrorCode


class ArenaError(Exception):
    """An expected, client-visible failure. Unexpected failures propagate unchanged."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message

    def payload(self) -> ArenaErrorPayload:
        return ArenaErrorPayload(code=self.code, message=self.message)
