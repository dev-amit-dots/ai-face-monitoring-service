"""In-memory session storage and cleanup."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SessionState:
    session_id: str
    user_id: str
    registered_embedding: np.ndarray | None = None
    last_frame_gray: np.ndarray | None = None
    freeze_started_at: float | None = None
    away_started_at: float | None = None
    last_detected_status: str | None = None
    last_activity_time: float = field(default_factory=time.time)
    warning_counters: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Concurrency-safe in-memory manager for active monitoring sessions."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str, user_id: str) -> SessionState:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState(session_id=session_id, user_id=user_id)
                self._sessions[session_id] = session
            else:
                session.user_id = user_id
            session.last_activity_time = time.time()
            return session

    async def touch(self, session_id: str) -> None:
        async with self._lock:
            if session := self._sessions.get(session_id):
                session.last_activity_time = time.time()

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def cleanup_expired(self) -> int:
        now = time.time()
        async with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if now - session.last_activity_time > self._ttl_seconds
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)
        return len(expired)

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)

