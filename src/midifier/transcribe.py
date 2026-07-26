"""Turn audio into a multi-track General MIDI file.

Two passes. The first decodes freely and is used only to work out which instruments are
really playing; the second decodes again with the invented ones forbidden. The second pass
is what makes the difference: notes filed under a hallucinated instrument are reassigned to
the parts that actually played them, where simply deleting that track loses whole sections.

The model is driven as a subprocess rather than imported. It owns process-wide state, its
CLI is the interface its authors support, and a decode that wedges the GPU takes the
subprocess down instead of the service.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pretty_midi

from midifier.midi import cleanup
from midifier.midi import detect

if TYPE_CHECKING:
    from midifier.config import Settings

# Beyond this a decode is not slow, it is stuck; the model runs at roughly twice the
# length of the song on a GPU and a job that has gone far past that will not recover.
TIMEOUT_MULTIPLIER = 20.0
MIN_TIMEOUT_SECONDS = 300.0


class TranscriptionError(RuntimeError):
    """The model failed, or produced nothing usable."""


@dataclass(frozen=True)
class Track:
    name: str
    program: int
    is_drum: bool
    note_count: int


@dataclass(frozen=True)
class Result:
    midi: bytes
    duration: float
    tracks: list[Track]
    dropped: list[tuple[str, str]]
    cleanup: cleanup.CleanupReport


def _run(args: list[str], timeout: float) -> None:
    process = subprocess.run(
        ["python", "-m", "muscriptor", "transcribe", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        raise TranscriptionError((process.stderr or process.stdout or "muscriptor failed")[-2000:])


def _events_to_notes(events_path: Path) -> list[tuple[str, float, float, int]]:
    """Pair the event stream's starts and ends into notes."""
    events = json.loads(events_path.read_text())
    starts = {event["index"]: event for event in events if event["type"] == "start"}

    notes: list[tuple[str, float, float, int]] = []
    for event in events:
        if event["type"] != "end":
            continue
        start = starts.get(event["start_event_index"])
        if start is not None:
            notes.append((start["instrument"], start["start_time"], event["end_time"], start["pitch"]))
    return notes


def _audio_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(probe.stdout.strip())
    except ValueError as error:
        raise TranscriptionError(f"could not read audio duration: {probe.stderr[-500:]}") from error


def transcribe(audio: Path, settings: Settings) -> Result:
    """Run the pipeline over one file and return a finished MIDI."""
    duration = _audio_duration(audio)
    timeout = max(duration * TIMEOUT_MULTIPLIER, MIN_TIMEOUT_SECONDS)
    common = ["-m", settings.model_size, "-d", settings.device]

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        midi_path = work / "out.mid"
        constraint: list[str] = []
        dropped: list[tuple[str, str]] = []

        if settings.two_pass:
            events_path = work / "pass1.json"
            _run([str(audio), "-o", str(events_path), "-f", "json", *common], timeout)
            parts = detect.summarise(_events_to_notes(events_path), duration)
            verdict = detect.choose(parts)
            dropped = verdict.dropped
            if dropped:
                constraint = ["--instruments", ",".join(verdict.keep)]

        _run([str(audio), "-o", str(midi_path), "-f", "midi", *constraint, *common], timeout)
        if not midi_path.is_file():
            raise TranscriptionError("muscriptor reported success but wrote no file")

        midi = pretty_midi.PrettyMIDI(str(midi_path))
        report = cleanup.clean(midi, duration)
        midi.write(str(midi_path))
        payload = midi_path.read_bytes()

    tracks = [
        Track(
            name=instrument.name or f"Track {index + 1}",
            program=instrument.program,
            is_drum=instrument.is_drum,
            note_count=len(instrument.notes),
        )
        for index, instrument in enumerate(midi.instruments)
        if instrument.notes
    ]
    if not tracks:
        raise TranscriptionError("transcription produced no notes")

    return Result(midi=payload, duration=duration, tracks=tracks, dropped=dropped, cleanup=report)
