"""Application service for processing WebSocket payloads."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from face_monitor.services.frame_processor import FrameProcessor
from face_monitor.session_manager import SessionManager
from face_monitor.utils.helpers import response_payload
from face_monitor.utils.image_utils import ImageDecodeError, decode_base64_image


class FramePayload(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    timestamp: int | float | None = None
    image: str = Field(min_length=1)


class MonitoringService:
    def __init__(self, sessions: SessionManager, processor: FrameProcessor) -> None:
        self._sessions = sessions
        self._processor = processor

    async def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            frame_payload = FramePayload.model_validate(payload)
            frame = decode_base64_image(frame_payload.image)
            session = await self._sessions.get_or_create(
                frame_payload.session_id, frame_payload.user_id
            )
            result = await asyncio.to_thread(self._processor.process, session, frame)
            await self._sessions.touch(frame_payload.session_id)
            return response_payload(
                session_id=frame_payload.session_id,
                status=result.status,
                message=result.message,
                **result.details,
            )
        except ValidationError as exc:
            return response_payload(
                session_id=str(payload.get("session_id", "")),
                status="ERROR",
                message="Invalid frame payload",
                errors=exc.errors(),
            )
        except ImageDecodeError as exc:
            return response_payload(
                session_id=str(payload.get("session_id", "")),
                status="ERROR",
                message=str(exc),
            )
        except Exception:
            return response_payload(
                session_id=str(payload.get("session_id", "")),
                status="ERROR",
                message="Unexpected processing error",
            )

