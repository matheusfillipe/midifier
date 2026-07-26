"""The job store and queue, shared by both surfaces.

REST and MCP are two doors onto one service: a job started over MCP must be visible over
REST, and both must contend for the same single transcription slot. Keeping these here
rather than in either surface also stops the two importing each other.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from midifier.config import get_settings
from midifier.jobs import JobStore
from midifier.queue import JobQueue

if TYPE_CHECKING:
    from midifier.config import Settings

store = JobStore()
queue = JobQueue(
    seconds_per_audio_second=get_settings().seconds_per_audio_second,
    workers=get_settings().max_concurrent_jobs,
)


def start(job_id: str, settings: Settings, payload: bytes | None = None, url: str | None = None) -> None:
    """Begin a job in the background, waiting for a queue slot first.

    Both surfaces route through here. Doing it in either one alone is how MCP callers
    ended up with jobs that queued and never ran.
    """
    from midifier.worker import run_job

    threading.Thread(
        target=queue.run,
        args=(job_id, lambda: run_job(job_id, store, settings, payload, url, queue)),
        daemon=True,
    ).start()
