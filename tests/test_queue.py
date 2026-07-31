"""The queue: one job at a time, and an estimate that learns from real jobs."""

import contextlib
import threading
import time

from midifier.queue import JobQueue


class TestSerialisation:
    def test_only_one_job_runs_at_a_time(self) -> None:
        """Two concurrent decodes compete for the accelerator; that is the whole point."""
        queue = JobQueue(seconds_per_audio_second=3.0)
        concurrent = 0
        peak = 0
        lock = threading.Lock()

        def work() -> None:
            nonlocal concurrent, peak
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.05)
            with lock:
                concurrent -= 1

        threads = []
        for index in range(4):
            job = f"job{index}"
            queue.submit(job)
            thread = threading.Thread(target=queue.run, args=(job, work))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        assert peak == 1

    def test_a_failing_job_releases_its_slot(self) -> None:
        """A crash must not wedge the queue for everything behind it."""
        queue = JobQueue(seconds_per_audio_second=3.0)
        queue.submit("bad")
        with contextlib.suppress(RuntimeError):
            queue.run("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        ran = []
        queue.submit("good")
        queue.run("good", lambda: ran.append(True))
        assert ran == [True]


class TestPosition:
    def test_reports_how_many_are_ahead(self) -> None:
        queue = JobQueue(seconds_per_audio_second=3.0)
        for index in range(3):
            queue.submit(f"job{index}", audio_seconds=100.0)

        assert queue.position("job0").ahead == 0  # type: ignore[union-attr]
        assert queue.position("job2").ahead == 2  # type: ignore[union-attr]

    def test_estimates_from_the_configured_rate_before_any_job_finishes(self) -> None:
        queue = JobQueue(seconds_per_audio_second=3.0)
        queue.submit("only", audio_seconds=100.0)
        where = queue.position("only")
        assert where is not None
        assert where.eta_seconds == 300.0

    def test_a_later_job_waits_for_the_ones_ahead(self) -> None:
        queue = JobQueue(seconds_per_audio_second=2.0)
        queue.submit("first", audio_seconds=100.0)
        queue.submit("second", audio_seconds=100.0)
        first = queue.position("first")
        second = queue.position("second")
        assert first is not None and second is not None
        assert second.eta_seconds > first.eta_seconds  # type: ignore[operator]

    def test_unknown_job_has_no_position(self) -> None:
        assert JobQueue(seconds_per_audio_second=3.0).position("nope") is None


class TestLearnedRate:
    def test_the_estimate_follows_the_machine(self) -> None:
        """Seeded with a guess, corrected by what the hardware actually does."""
        queue = JobQueue(seconds_per_audio_second=10.0)
        assert queue.rate == 10.0

        for _ in range(3):
            queue.observe("j", audio_seconds=200.0, elapsed_seconds=400.0)

        assert queue.rate == 2.0

    def test_a_zero_length_job_is_ignored(self) -> None:
        queue = JobQueue(seconds_per_audio_second=3.0)
        queue.observe("j", audio_seconds=0.0, elapsed_seconds=100.0)
        assert queue.rate == 3.0

    def test_snapshot_reports_how_many_jobs_informed_the_rate(self) -> None:
        queue = JobQueue(seconds_per_audio_second=3.0)
        queue.observe("j", audio_seconds=100.0, elapsed_seconds=200.0)
        snapshot = queue.snapshot()
        assert snapshot["measured_from"] == 1
        assert snapshot["seconds_per_audio_second"] == 2.0


class TestUnknownDuration:
    """Audio given as a URL is not measured until it is fetched, but a caller waiting on
    the queue still needs a number."""

    def test_estimates_from_a_nominal_length(self) -> None:
        queue = JobQueue(seconds_per_audio_second=3.0)
        queue.submit("unmeasured")
        where = queue.position("unmeasured")
        assert where is not None
        assert where.eta_seconds is not None
        assert where.eta_seconds > 0
