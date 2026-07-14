"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager

_matplotlib_cache = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from face_monitor.config import get_settings
from face_monitor.services.frame_processor import FrameProcessor
from face_monitor.services.monitoring_service import MonitoringService
from face_monitor.services.websocket_service import WebSocketService
from face_monitor.session_manager import SessionManager
from face_monitor.utils.logger import configure_logging, get_logger
from face_monitor.websocket import router as websocket_router

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sessions = SessionManager(settings.session_ttl_seconds)
    processor = FrameProcessor(settings)
    monitoring = MonitoringService(sessions, processor)
    app.state.session_manager = sessions
    app.state.websocket_service = WebSocketService(monitoring)

    stop_event = asyncio.Event()
    cleanup_task = asyncio.create_task(_cleanup_loop(sessions, stop_event))
    logger.info("service_started", environment=settings.environment)
    try:
        yield
    finally:
        stop_event.set()
        cleanup_task.cancel()
        logger.info("service_stopped")


async def _cleanup_loop(sessions: SessionManager, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.cleanup_interval_seconds
            )
        except TimeoutError:
            removed = await sessions.cleanup_expired()
            if removed:
                logger.info("sessions_cleaned", removed=removed)


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(websocket_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
