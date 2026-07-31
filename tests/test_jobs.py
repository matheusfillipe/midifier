"""The job model, and the estimate a caller plans its waiting around."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

from midifier.jobs import Job
from midifier.jobs import JobState


def _running(done: int, total: int, seconds_ago: float, into_current: float = 0.0) -> Job:
    """`seconds_ago` is how long the finished segments took; `into_current` how long the
    running one has been going."""
    now = datetime.now(UTC)
    return Job(
        state=JobState.RUNNING,
        segments_done=done,
        segments_total=total,
        decoding_since=now - timedelta(seconds=seconds_ago + into_current),
        last_segment_at=now - timedelta(seconds=into_current),
    )


class TestMeasuredEta:
    """Decoding cost tracks the notes a song generates, and a dense song costs several times
    what a sparse one does, so the only honest estimate is the pace this song decodes at."""

    def test_extrapolates_from_what_has_already_decoded(self) -> None:
        # two of five segments in 200s is 100s each, so the three left are about 300s
        eta = _running(done=2, total=5, seconds_ago=200.0).measured_eta_seconds
        assert eta is not None
        assert 290.0 < eta < 310.0

    def test_a_dense_song_reports_a_longer_wait_than_a_sparse_one(self) -> None:
        dense = _running(done=1, total=5, seconds_ago=300.0).measured_eta_seconds
        sparse = _running(done=1, total=5, seconds_ago=100.0).measured_eta_seconds
        assert dense is not None and sparse is not None
        assert dense > sparse * 2

    def test_nothing_to_measure_before_the_first_segment_lands(self) -> None:
        assert _running(done=0, total=5, seconds_ago=30.0).measured_eta_seconds is None

    def test_the_last_segment_leaves_nothing_to_wait_for(self) -> None:
        assert _running(done=5, total=5, seconds_ago=500.0).measured_eta_seconds == 0.0

    def test_the_estimate_falls_while_a_segment_runs(self) -> None:
        """An estimate that climbs as you wait is worse than none: it tells a caller to sleep
        longer the longer it has already slept."""
        early = _running(done=2, total=6, seconds_ago=200.0, into_current=10.0).measured_eta_seconds
        later = _running(done=2, total=6, seconds_ago=200.0, into_current=90.0).measured_eta_seconds
        assert early is not None and later is not None
        assert later < early

    def test_it_never_reports_a_negative_wait(self) -> None:
        assert _running(done=1, total=2, seconds_ago=100.0, into_current=9_000.0).measured_eta_seconds == 0.0

    def test_a_finished_job_has_no_estimate(self) -> None:
        job = Job(
            state=JobState.SUCCEEDED,
            segments_done=5,
            segments_total=5,
            decoding_since=datetime.now(UTC),
        )
        assert job.measured_eta_seconds is None
