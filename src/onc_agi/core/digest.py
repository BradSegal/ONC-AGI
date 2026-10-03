"""Canonical JSON digests used by traces and replay."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_sha256(payload: Any) -> str:
    """SHA-256 of canonical JSON, so replays can verify request and response bytes."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()
