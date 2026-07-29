"""Runs transcriptions off the request thread and records what happened on the job."""

from __future__ import annotations

import logging
import tempfile
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from midifier.fetch import FetchError
from midifier.fetch import UnsafeUrlError
from midifier.fetch import fetch_audio
from midifier.jobs import JobState
from midifier.jobs import Stage
from midifier.jobs import Track
from midifier.storage import build_storage
from midifier.transcribe import TranscriptionError
from midifier.transcribe import transcribe

if TYPE_CHECKING:
    from midifier.config import Settings
    from midifier.jobs import JobStore
    from midifier.queue import JobQueue

logger = logging.getLogger(__name__)


def run_job(
    job_id: str,
    store: JobStore,
    settings: Settings,
    payload: bytes | None,
    url: str | None,
    queue: JobQueue | None = None,
) -> None:
    """Fetch, transcribe, store. Every failure ends on the job rather than in a traceback."""
    store.update(job_id, state=JobState.RUNNING, stage=Stage.FETCHING, queue_ahead=0)
    started = time.monotonic()

    try:
        with tempfile.TemporaryDirectory() as workspace:
            audio = Path(workspace) / "input"
            if payload is not None:
                audio.write_bytes(payload)
            else:
                assert url is not None
                audio.write_bytes(fetch_audio(url, settings.max_upload_bytes).content)

            store.update(job_id, stage=Stage.TRANSCRIBING)
            result = transcribe(audio, settings)

            if result.duration > settings.max_duration_seconds:
                raise TranscriptionError(
                    f"audio is {result.duration:.0f}s, longer than the {settings.max_duration_seconds:.0f}s limit"
                )

            store.update(job_id, stage=Stage.STORING)
            midi_url = build_storage(settings).put(f"{job_id}.mid", result.midi)

    except (TranscriptionError, FetchError, UnsafeUrlError, OSError) as error:
        logger.exception("job %s failed", job_id)
        store.update(
            job_id,
            state=JobState.FAILED,
            error=str(error),
            finished_at=datetime.now(UTC),
        )
        return

    if queue is not None:
        queue.observe(job_id, result.duration, time.monotonic() - started)

    store.update(
        job_id,
        state=JobState.SUCCEEDED,
        stage=None,
        midi_url=midi_url,
        duration_seconds=result.duration,
        tracks=[
            Track(
                name=track.name,
                program=track.program,
                is_drum=track.is_drum,
                note_count=track.note_count,
            )
            for track in result.tracks
        ],
        dropped_instruments=[name for name, _ in result.dropped],
        finished_at=datetime.now(UTC),
    )
