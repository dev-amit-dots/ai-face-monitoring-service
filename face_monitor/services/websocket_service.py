"""WebSocket connection helpers."""

from __future__ import annotations

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from face_monitor.services.monitoring_service import MonitoringService
from face_monitor.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketService:
    def __init__(self, monitoring: MonitoringService) -> None:
        self._monitoring = monitoring

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                payload = await websocket.receive_json()
                response = await self._monitoring.handle_payload(payload)
                await websocket.send_json(response)
        except WebSocketDisconnect:
            logger.info("websocket_disconnected")
        except Exception as exc:
            logger.exception("websocket_error", error=str(exc))
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(
                    {
                        "session_id": "",
                        "status": "ERROR",
                        "message": "WebSocket connection error",
                    }
                )

