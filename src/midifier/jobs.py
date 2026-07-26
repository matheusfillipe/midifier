"""Job lifecycle for transcriptions, which take minutes rather than milliseconds."""

from __future__ import annotations

import uuid
from datetime import UTC
from datetime import datetime
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Iterator


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(StrEnum):
    """Pipeline stages, in order. Reported so a caller can show real progress."""

    FETCHING = "fetching"
    DETECTING = "detecting"
    TRANSCRIBING = "transcribing"
    CLEANING = "cleaning"
    STORING = "storing"


class Track(BaseModel):
    """One instrument in the finished MIDI."""

    name: str
    program: int = Field(ge=0, le=127)
    is_drum: bool
    note_count: int


class Job(BaseModel):
    """A transcription request and everything learned while serving it."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    state: JobState = JobState.QUEUED
    stage: Stage | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    source: str | None = None
    duration_seconds: float | None = None
    tempo: float | None = None

    # Filled while the job waits, so a caller can see the queue rather than guess.
    queue_ahead: int | None = None
    eta_seconds: float | None = None

    midi_url: str | None = None
    tracks: list[Track] = Field(default_factory=list)
    dropped_instruments: list[str] = Field(default_factory=list)
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.state in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobStore:
    """In-memory job registry.

    Deliberately not a queue library. Jobs are long but few, and the deployment target
    runs one Kubernetes Job per request, so durable scheduling is the cluster's problem
    rather than this process's. Swapping in a shared store means implementing these four
    methods, nothing more.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self, source: str | None = None) -> Job:
        job = Job(source=source)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: object) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            updated = job.model_copy(update=fields)
            self._jobs[job_id] = updated
            return updated

    def __iter__(self) -> Iterator[Job]:
        with self._lock:
            return iter(list(self._jobs.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)
