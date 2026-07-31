"""The worker, with the model mocked at its boundary.

Everything below the `transcribe` call is a multi-gigabyte gated model and a GPU, so the
seam is exactly there: the pipeline's own logic is tested in test_cleanup, test_consolidate and test_segments,
and what matters here is that a job ends in the right state with the right fields.
"""

import time
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from midifier.config import Settings
from midifier.jobs import JobState
from midifier.jobs import JobStore
from midifier.midi.cleanup import CleanupReport
from midifier.transcribe import Result
from midifier.transcribe import Track
from midifier.transcribe import TranscriptionError
from midifier.worker import run_job

MIDI_BYTES = b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x60"


def _result(duration: float = 30.0) -> Result:
    return Result(
        midi=MIDI_BYTES,
        duration=duration,
        tracks=[
            Track(name="electric bass", program=33, is_drum=False, note_count=100),
            Track(name="drums", program=0, is_drum=True, note_count=200),
        ],
        dropped=[("synth_lead", "distorted_electric_guitar")],
        cleanup=CleanupReport(1, 2, 3, 310, 300, []),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage_backend="local", local_storage_dir=tmp_path / "out")


class TestSuccess:
    def test_records_tracks_and_a_url(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("midifier.worker.transcribe", lambda audio, cfg, progress=None: _result())
        store = JobStore()
        job = store.create(source="song.mp3")

        run_job(job.id, store, settings, payload=b"audio", url=None)

        done = store.get(job.id)
        assert done is not None
        assert done.state is JobState.SUCCEEDED
        assert done.midi_url == f"/v1/files/{job.id}.mid"
        assert [track.name for track in done.tracks] == ["electric bass", "drums"]
        assert done.dropped_instruments == ["synth_lead"]
        assert done.finished_at is not None

    def test_stores_the_midi_where_the_url_points(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("midifier.worker.transcribe", lambda audio, cfg, progress=None: _result())
        store = JobStore()
        job = store.create()

        run_job(job.id, store, settings, payload=b"audio", url=None)

        assert (settings.local_storage_dir / f"{job.id}.mid").read_bytes() == MIDI_BYTES

    def test_clears_the_stage_when_finished(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("midifier.worker.transcribe", lambda audio, cfg, progress=None: _result())
        store = JobStore()
        job = store.create()

        run_job(job.id, store, settings, payload=b"audio", url=None)

        done = store.get(job.id)
        assert done is not None and done.stage is None


class TestFailure:
    def test_a_model_failure_lands_on_the_job(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        def explode(audio: Path, cfg: Settings, progress: object = None) -> Result:
            raise TranscriptionError("GPU Hang")

        monkeypatch.setattr("midifier.worker.transcribe", explode)
        store = JobStore()
        job = store.create()

        run_job(job.id, store, settings, payload=b"audio", url=None)

        done = store.get(job.id)
        assert done is not None
        assert done.state is JobState.FAILED
        assert done.error is not None and "GPU Hang" in done.error
        assert done.finished_at is not None

    def test_audio_longer_than_the_limit_is_refused(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        """The check runs after transcription because duration comes from the audio itself."""
        monkeypatch.setattr("midifier.worker.transcribe", lambda audio, cfg, progress=None: _result(duration=99_999.0))
        store = JobStore()
        job = store.create()

        run_job(job.id, store, settings, payload=b"audio", url=None)

        done = store.get(job.id)
        assert done is not None
        assert done.state is JobState.FAILED
        assert "longer than" in (done.error or "")

    def test_a_refused_url_lands_on_the_job(self, settings: Settings) -> None:
        store = JobStore()
        job = store.create()

        run_job(job.id, store, settings, payload=None, url="http://127.0.0.1/song.mp3")

        done = store.get(job.id)
        assert done is not None
        assert done.state is JobState.FAILED


class TestStrandedJobs:
    """A job that stops moving is worse than one that fails: a caller polls it forever."""

    def test_a_dead_link_fails_the_job(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        from midifier.fetch import FetchError

        def dead(url: str, limit: int) -> None:
            raise FetchError("could not fetch https://example.invalid/x.mp3: 404 Not Found")

        monkeypatch.setattr("midifier.worker.fetch_audio", dead)
        store = JobStore()
        job = store.create(source="https://example.invalid/x.mp3")

        run_job(job.id, store, settings, payload=None, url="https://example.invalid/x.mp3")

        done = store.get(job.id)
        assert done is not None
        assert done.state is JobState.FAILED
        assert "404" in (done.error or "")

    def test_an_unexpected_error_still_ends_the_job(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        """Whatever escapes run_job must not leave the caller polling a job that never moves."""
        import midifier.state as state

        def boom(*args: object, **kwargs: object) -> None:
            raise KeyboardInterrupt("something no one predicted")

        monkeypatch.setattr("midifier.state.run_job", boom)
        job = state.store.create(source="song.mp3")
        state.start(job.id, settings, payload=b"audio")

        for _ in range(200):
            current = state.store.get(job.id)
            if current is not None and current.done:
                break
            time.sleep(0.02)

        current = state.store.get(job.id)
        assert current is not None
        assert current.state is JobState.FAILED
        assert "KeyboardInterrupt" in (current.error or "")
