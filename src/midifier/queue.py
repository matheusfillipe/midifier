"""Runs transcriptions one at a time and says how long the wait will be.

Transcription is heavy and, on a single accelerator, two at once is how a working setup
becomes a failing one. Requests are therefore admitted immediately but serialised here,
and a caller can see where it sits and roughly when its turn comes.

The estimate is measured, not configured: each finished job contributes its real ratio of
processing time to audio length, so the queue converges on whatever the machine actually
does rather than on a number written for one particular GPU.
"""

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# How many finished jobs the rate estimate averages over. Short enough to follow a change
# in the machine, long enough not to swing on one unusual song.
RATE_WINDOW = 10

# Assumed length of a song whose duration is not yet known, so a caller still gets a
# usable estimate. Most audio arrives as a URL and is not measured until it is fetched,
# and "no idea" is less useful to a waiting client than "roughly a typical song".
NOMINAL_AUDIO_SECONDS = 210.0


@dataclass(frozen=True)
class Position:
    """Where a job sits, and how long until it finishes."""

    ahead: int
    eta_seconds: float | None


class JobQueue:
    """Serialises work and estimates waiting time from observed throughput."""

    def __init__(self, seconds_per_audio_second: float, workers: int = 1) -> None:
        self._semaphore = threading.Semaphore(workers)
        self._lock = threading.Lock()
        self._waiting: deque[str] = deque()
        self._running: set[str] = set()
        self._rates: deque[float] = deque(maxlen=RATE_WINDOW)
        self._default_rate = seconds_per_audio_second
        self._durations: dict[str, float] = {}

    @property
    def rate(self) -> float:
        """Seconds of processing per second of audio, measured where possible."""
        with self._lock:
            if not self._rates:
                return self._default_rate
            return sum(self._rates) / len(self._rates)

    def submit(self, job_id: str, audio_seconds: float | None = None) -> None:
        with self._lock:
            self._waiting.append(job_id)
            if audio_seconds is not None:
                self._durations[job_id] = audio_seconds

    def observe(self, job_id: str, audio_seconds: float, elapsed_seconds: float) -> None:
        """Record what a finished job actually cost, so later estimates improve."""
        if audio_seconds <= 0:
            return
        with self._lock:
            self._rates.append(elapsed_seconds / audio_seconds)

    def position(self, job_id: str) -> Position | None:
        """How many jobs are ahead, and the estimated seconds until this one is done."""
        with self._lock:
            if job_id in self._running:
                ahead = 0
            elif job_id in self._waiting:
                ahead = list(self._waiting).index(job_id) + len(self._running)
            else:
                return None

            rate = self._default_rate if not self._rates else sum(self._rates) / len(self._rates)
            queued = [*self._running, *self._waiting]
            upto = queued.index(job_id) if job_id in queued else len(queued)
            # Unknown-length jobs still have to be waited for, so they count at the
            # median of what is known rather than as free.
            known = [self._durations[j] for j in queued[: upto + 1] if j in self._durations]
            fallback = sum(known) / len(known) if known else NOMINAL_AUDIO_SECONDS
            pending = sum(self._durations.get(j, fallback) for j in queued[: upto + 1])

        eta = pending * rate if pending > 0 else None
        return Position(ahead=ahead, eta_seconds=eta)

    def run(self, job_id: str, work: Callable[[], None]) -> None:
        """Wait for a slot, then run. Always releases, however the work ends."""
        self._semaphore.acquire()
        with self._lock:
            if job_id in self._waiting:
                self._waiting.remove(job_id)
            self._running.add(job_id)
        try:
            work()
        finally:
            with self._lock:
                self._running.discard(job_id)
                self._durations.pop(job_id, None)
            self._semaphore.release()

    def snapshot(self) -> dict[str, object]:
        """Queue state, for the status endpoint."""
        with self._lock:
            return {
                "running": len(self._running),
                "waiting": len(self._waiting),
                "seconds_per_audio_second": (
                    self._default_rate if not self._rates else sum(self._rates) / len(self._rates)
                ),
                "measured_from": len(self._rates),
            }
