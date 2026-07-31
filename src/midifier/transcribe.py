"""Turn audio into a multi-track General MIDI file.

The audio is decoded in overlapping segments rather than in one pass. The model works in
five-second chunks and teacher-forces each chunk's opening from the previous one, which keeps
instruments stable but also carries its mistakes forward: a part it stops hearing stays gone
for the rest of the song, and an ending it starts inventing runs to the last chunk. Segmenting
caps how far either travels. A closing segment also stops on its own rather than spending its
budget inventing an ending, so the extra audio a segmented run decodes largely pays for itself.

What segmenting costs is lane identity, since a segment decoded alone has no reason to name a
part the way its neighbour did. That is repaired afterwards, from the overlap.

The model is driven as a subprocess rather than imported. It owns process-wide state, its CLI
is the interface its authors support, and a decode that wedges the GPU takes the subprocess
down instead of the service.
"""

import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pretty_midi

from midifier.config import Settings
from midifier.midi import cleanup
from midifier.midi import segments
from midifier.midi.consolidate import consolidate

ModelSize = Literal["small", "medium", "large"]
Progress = Callable[[int, int], None]

# Beyond this a decode is not slow, it is stuck; a segment runs at roughly three times its own
# length and one that has gone far past that will not recover.
TIMEOUT_MULTIPLIER = 20.0
MIN_TIMEOUT_SECONDS = 300.0

# Failures worth another attempt: the device fell over, rather than the audio being wrong.
TRANSIENT_MARKERS = ("GPU Hang", "HW Exception", "HIPFFT", "CUDA error", "out of memory", "Aborted")

# Long enough for a driver to settle after a reset before the next process asks for the device.
RETRY_PAUSE_SECONDS = 5.0

# Sizes to fall back through when one keeps hanging. A missing kernel is specific to the shapes
# a size produces, so a smaller model often decodes audio the larger one cannot -- and one
# segment transcribed less well beats a song that fails outright.
SMALLER_MODEL: dict[str, ModelSize] = {"large": "medium", "medium": "small"}

logger = logging.getLogger(__name__)


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


def _environment(settings: Settings) -> dict[str, str]:
    """The model reads its own token from `HF_TOKEN`, under that name and no other.

    The weights are gated, so without this a size whose files are not already cached fails at
    download time rather than at startup -- the service looks healthy until the first job.
    """
    environment = dict(os.environ)
    if settings.hf_token:
        environment["HF_TOKEN"] = settings.hf_token
    return environment


def _run(args: list[str], timeout: float, settings: Settings) -> None:
    process = subprocess.run(
        ["python", "-m", "muscriptor", "transcribe", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=_environment(settings),
    )
    if process.returncode != 0:
        raise TranscriptionError((process.stderr or process.stdout or "muscriptor failed")[-2000:])


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


def _cut(source: Path, start: float, length: float, destination: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(source), str(destination)],
        capture_output=True,
        check=True,
    )


def _decode(audio: Path, destination: Path, settings: Settings, timeout: float) -> pretty_midi.PrettyMIDI:
    """Decode one piece of audio, retrying a decode that died on the accelerator.

    Some accelerators lack kernels for particular matrix shapes and hang rather than erroring,
    which kills the subprocess. It is not the audio: the same segment usually decodes on the
    next attempt, in a fresh process with a fresh device context. Retrying here turns an
    intermittent hardware fault into a slower success instead of a failed job.
    """
    model: ModelSize = settings.model_size
    attempts = settings.decode_attempts
    while True:
        for attempt in range(1, attempts + 1):
            try:
                _run(
                    [str(audio), "-o", str(destination), "-f", "midi", "-m", model, "-d", settings.device],
                    timeout,
                    settings,
                )
            except TranscriptionError as error:
                if not _is_transient(str(error)):
                    raise
                if attempt < attempts:
                    logger.warning(
                        "decode of %s failed (%d/%d on %s), retrying: %s", audio.name, attempt, attempts, model, error
                    )
                    destination.unlink(missing_ok=True)
                    time.sleep(RETRY_PAUSE_SECONDS)
                    continue
                smaller = SMALLER_MODEL.get(model)
                if smaller is None:
                    raise
                logger.warning("%s keeps failing on %s, falling back to %s", audio.name, model, smaller)
                destination.unlink(missing_ok=True)
                time.sleep(RETRY_PAUSE_SECONDS)
                model = smaller
                break

            if not destination.is_file():
                raise TranscriptionError("muscriptor reported success but wrote no file")
            return pretty_midi.PrettyMIDI(str(destination))


def _is_transient(message: str) -> bool:
    """Whether a failure is the accelerator misbehaving rather than the input being wrong."""
    return any(marker in message for marker in TRANSIENT_MARKERS)


def transcribe(audio: Path, settings: Settings, progress: Progress | None = None) -> Result:
    """Run the pipeline over one file and return a finished MIDI.

    `progress` is called with (segments done, segments total) as each lands, so a caller can
    report a wait measured from this song rather than from an average one.
    """
    duration = _audio_duration(audio)
    length = settings.segment_seconds

    with tempfile.TemporaryDirectory() as workspace:
        work = Path(workspace)
        if duration <= length:
            midi = _decode(audio, work / "out.mid", settings, max(duration * TIMEOUT_MULTIPLIER, MIN_TIMEOUT_SECONDS))
        else:
            timeout = max(length * TIMEOUT_MULTIPLIER, MIN_TIMEOUT_SECONDS)
            offsets = segments.plan(duration, length)
            if progress is not None:
                progress(0, len(offsets))
            decoded = []
            for index, offset in enumerate(offsets):
                clip = work / f"s{index}.wav"
                _cut(audio, offset, length, clip)
                decoded.append((offset, _decode(clip, work / f"s{index}.mid", settings, timeout)))
                if progress is not None:
                    progress(index + 1, len(offsets))
            midi = segments.stitch(decoded, length)

        dropped = consolidate(midi)
        report = cleanup.clean(midi, duration)
        output = work / "final.mid"
        midi.write(str(output))
        payload = output.read_bytes()

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
