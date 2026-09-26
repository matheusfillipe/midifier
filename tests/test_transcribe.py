"""A quick scout decides the song's instruments, and every segment is decoded held to them."""

from pathlib import Path

import pretty_midi
import pytest

from midifier.config import Settings
from midifier.transcribe import transcribe


def _write(path: Path, lanes: dict[str, int]) -> None:
    midi = pretty_midi.PrettyMIDI()
    for name, count in lanes.items():
        lane = pretty_midi.Instrument(program=52 if name == "voice" else 71, name=name)
        lane.notes = [
            pretty_midi.Note(velocity=100, pitch=60, start=index * 0.5, end=index * 0.5 + 0.2) for index in range(count)
        ]
        midi.instruments.append(lane)
    midi.write(str(path))


def test_every_segment_is_decoded_held_to_what_the_scout_heard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def decoder(args: list[str], timeout: float, settings: Settings) -> None:
        calls.append(args)
        if "--no-prelude-forcing" in args:
            # The whole song: a voice throughout, and a clarinet the scout barely used.
            _write(Path(args[2]), {"voice": 198, "clarinet": 3})
        else:
            _write(Path(args[2]), {"voice": 60})

    monkeypatch.setattr("midifier.transcribe._run", decoder)
    monkeypatch.setattr("midifier.transcribe._audio_duration", lambda path: 100.0)
    monkeypatch.setattr("midifier.transcribe._cut", lambda source, start, length, destination: destination.touch())

    progress: list[tuple[int, int]] = []
    result = transcribe(tmp_path / "song.m4a", Settings(storage_backend="local"), lambda *step: progress.append(step))

    scout, *segments = calls
    assert Path(scout[0]).name == "song.wav"
    assert "--instruments" not in scout
    assert [call[call.index("--instruments") + 1] for call in segments] == ["voice", "voice"]
    assert all("--no-prelude-forcing" not in call for call in segments)
    assert progress[-1] == (3, 3)
    assert [track.name for track in result.tracks] == ["voice"]
