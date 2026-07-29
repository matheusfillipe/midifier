"""Runtime configuration, entirely from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the service has. Prefix each with `MIDIFIER_`."""

    model_config = SettingsConfigDict(env_prefix="MIDIFIER_", env_file=".env", extra="ignore")

    # --- service ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    # The hash of the key, never the key itself, so the deployed secret cannot be used to
    # call the service. Generate both with `python -m midifier keygen`.
    api_key_hash: str | None = Field(
        default=None,
        description="SHA-256 of the API key. When set, callers must present the key itself.",
    )

    # --- storage ---
    # Results land in object storage when configured, otherwise on disk. Local is the
    # default so the service runs with no infrastructure at all.
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_dir: Path = Path("./data")

    # Named to match kinesthesia's own MinIO variables, since the two services address
    # the same bucket and sharing the vocabulary avoids a translation layer.
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str | None = None
    minio_public_base: str | None = None
    minio_use_ssl: bool = True
    minio_region: str = "us-east-1"

    # --- queue ---
    # One transcription at a time. A second concurrent decode competes for whatever
    # accelerator is present, and on shared or memory-constrained hardware that is how a
    # working setup turns into a failing one.
    max_concurrent_jobs: int = 1

    # Seconds of processing per second of audio, used for the queue's ETA before any job
    # has finished. It is only a starting point: the real figure is measured from
    # completed jobs, so this need not match any particular machine.
    seconds_per_audio_second: float = 3.0

    # --- transcription ---
    model_size: Literal["small", "medium", "large"] = "large"
    device: str = "auto"
    hf_token: str | None = None

    # Audio is decoded in overlapping segments of this length. The model carries each of its
    # own five-second chunks into the next, so mistakes travel forward; shorter segments cap
    # that reach but give the model less context to place a part in.
    segment_seconds: float = 60.0

    # How many times a single decode is attempted. A hang on the accelerator kills the
    # subprocess without touching the audio, and the retry usually succeeds, so one attempt
    # turns a hardware quirk into a failed job for no reason.
    decode_attempts: int = 3

    # --- input limits ---
    max_upload_bytes: int = 100 * 1024 * 1024
    max_duration_seconds: float = 360.0
    allow_url_input: bool = True

    @property
    def s3_configured(self) -> bool:
        return all(
            (
                self.minio_endpoint,
                self.minio_access_key,
                self.minio_secret_key,
                self.minio_bucket,
            )
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings are read once; tests override by clearing this cache."""
    return Settings()
