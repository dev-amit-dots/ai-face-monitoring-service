"""Small shared helpers."""

import time
from typing import Any


def utc_timestamp() -> int:
    return int(time.time())


def response_payload(
    session_id: str,
    status: str,
    message: str,
    timestamp: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "message": message,
        "timestamp": timestamp or utc_timestamp(),
    }
    payload.update(extra)
    return payload

