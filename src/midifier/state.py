"""The job store and queue, shared by both surfaces.

REST and MCP are two doors onto one service: a job started over MCP must be visible over
REST, and both must contend for the same single transcription slot. Keeping these here
rather than in either surface also stops the two importing each other.
"""

from __future__ import annotations

from midifier.config import get_settings
from midifier.jobs import JobStore
from midifier.queue import JobQueue

store = JobStore()
queue = JobQueue(
    seconds_per_audio_second=get_settings().seconds_per_audio_second,
    workers=get_settings().max_concurrent_jobs,
)
