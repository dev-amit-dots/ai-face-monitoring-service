"""WebSocket routes."""

from fastapi import APIRouter, Depends, WebSocket

from face_monitor.services.websocket_service import WebSocketService

router = APIRouter()


def get_websocket_service(websocket: WebSocket) -> WebSocketService:
    return websocket.app.state.websocket_service


@router.websocket("/ws/monitor")
async def monitor_websocket(
    websocket: WebSocket,
    service: WebSocketService = Depends(get_websocket_service),
) -> None:
    await service.handle(websocket)

