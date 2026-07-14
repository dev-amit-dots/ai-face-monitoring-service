"""Runtime configuration for the face monitoring service."""

from functools import lru_cache
from typing import Sequence

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "AI Face Monitoring Service"
    environment: str = Field(default="production", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    websocket_path: str = Field(default="/ws/monitor", alias="WEBSOCKET_PATH")
    cors_origins: Sequence[str] = Field(default=("*",), alias="CORS_ORIGINS")

    session_ttl_seconds: int = Field(default=300, alias="SESSION_TTL_SECONDS", ge=30)
    cleanup_interval_seconds: int = Field(
        default=60, alias="CLEANUP_INTERVAL_SECONDS", ge=10
    )

    face_detection_confidence: float = Field(
        default=0.6, alias="FACE_DETECTION_CONFIDENCE", ge=0.0, le=1.0
    )
    face_match_tolerance: float = Field(
        default=0.55, alias="FACE_MATCH_TOLERANCE", ge=0.0, le=2.0
    )
    looking_away_seconds: float = Field(
        default=5.0, alias="LOOKING_AWAY_SECONDS", ge=1.0
    )

    freeze_seconds: float = Field(default=6.0, alias="FREEZE_SECONDS", ge=2.0)
    freeze_difference_threshold: float = Field(
        default=2.5, alias="FREEZE_DIFFERENCE_THRESHOLD", ge=0.0
    )
    dark_mean_threshold: float = Field(
        default=18.0, alias="DARK_MEAN_THRESHOLD", ge=0.0
    )
    dark_std_threshold: float = Field(default=8.0, alias="DARK_STD_THRESHOLD", ge=0.0)

    spoof_model_path: str | None = Field(default=None, alias="SPOOF_MODEL_PATH")
    spoof_score_threshold: float = Field(
        default=0.7, alias="SPOOF_SCORE_THRESHOLD", ge=0.0, le=1.0
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | Sequence[str]) -> Sequence[str]:
        if isinstance(value, str):
            return tuple(origin.strip() for origin in value.split(",") if origin.strip())
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

