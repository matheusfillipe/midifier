"""The job store and queue, shared by both surfaces.

REST and MCP are two doors onto one service: a job started over MCP must be visible over
REST, and both must contend for the same single transcription slot. Keeping these here
rather than in either surface also stops the two importing each other.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING

from midifier.config import get_settings
from midifier.jobs import JobState
from midifier.jobs import JobStore
from midifier.queue import JobQueue

if TYPE_CHECKING:
    from midifier.config import Settings

logger = logging.getLogger(__name__)

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

    def work() -> None:
        # run_job ends every failure it anticipates on the job itself. Anything it did not
        # anticipate would otherwise kill this thread silently, leaving the job "running"
        # for as long as the process lives and a caller polling something that can never
        # change. A job the caller can see failed is always better than one that hangs.
        try:
            run_job(job_id, store, settings, payload, url, queue)
        except BaseException as error:
            logger.exception("job %s died unexpectedly", job_id)
            job = store.get(job_id)
            if job is not None and not job.done:
                store.update(
                    job_id,
                    state=JobState.FAILED,
                    error=f"{type(error).__name__}: {error}",
                    finished_at=datetime.now(UTC),
                )
            raise

    threading.Thread(target=queue.run, args=(job_id, work), daemon=True).start()
