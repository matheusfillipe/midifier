"""Quality spike: audio -> 6 stems -> per-stem transcription -> one multi-track GM MIDI.

Throwaway. This exists to answer "is any of this good enough to learn from", not to be
the service. Variant B of the plan: htdemucs_6s + basic-pitch, all permissive licences.

    uv run scripts/spike.py ~/tmp/song.mp3
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.signal
import torch

logging.getLogger().setLevel(logging.ERROR)

# Kinesthesia draws only this range and silently drops nothing, so notes outside it
# would render off-keyboard. See claudedocs/research/00-local-context.md.
LOWEST_PITCH = 21
HIGHEST_PITCH = 108

# Its percussion path plays an smplr TR-808 kit addressed by name, and only maps GM
# notes 35-82; anything else silently becomes a snare.
DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_HIHAT = 42


@dataclass(frozen=True)
class StemSpec:
    """How one demucs stem is transcribed and what it becomes in the MIDI."""

    label: str
    program: int
    min_freq: float | None
    max_freq: float | None
    onset_threshold: float
    frame_threshold: float
    min_note_ms: float


# Programs are 0-based. Chosen for the target soundfont, see the plan's mapping table.
STEM_SPECS: dict[str, StemSpec] = {
    "vocals": StemSpec("Vocals", 53, 80, 1400, 0.6, 0.4, 120),
    "bass": StemSpec("Bass", 33, 30, 500, 0.5, 0.3, 100),
    "guitar": StemSpec("Guitar", 25, 70, 2000, 0.6, 0.4, 100),
    "piano": StemSpec("Piano", 0, 50, 3000, 0.6, 0.4, 90),
    "other": StemSpec("Other", 48, 60, 3000, 0.7, 0.5, 120),
}


def separate(path: Path, out_dir: Path, device: str, reuse: bool) -> dict[str, Path]:
    """Split the mix into htdemucs_6s stems, returning one wav path per stem."""
    names = (*STEM_SPECS.keys(), "drums")
    if reuse:
        existing = {n: out_dir / f"{n}.wav" for n in names if (out_dir / f"{n}.wav").exists()}
        if existing:
            print(f"[spike] reusing {len(existing)} stems from {out_dir}")
            return existing

    from demucs.api import Separator
    from demucs.api import save_audio

    separator = Separator(model="htdemucs_6s", device=device, progress=True)
    _, stems = separator.separate_audio_file(path)

    paths: dict[str, Path] = {}
    for name, source in stems.items():
        target = out_dir / f"{name}.wav"
        save_audio(source, str(target), samplerate=separator.samplerate)
        paths[name] = target
    return paths


def note_energy_db(y: np.ndarray, sr: int, start: float, end: float, pitch: int) -> float:
    """Energy near a note's fundamental, in dB.

    basic-pitch's own amplitude is the mean frame-activation posteriorgram, i.e. model
    confidence, which conflates dynamics with detection certainty. Real energy in a band
    around the fundamental is closer to what a listener calls dynamics.
    """
    fundamental = librosa.midi_to_hz(pitch)
    low = max(fundamental * 0.85, 20.0)
    high = min(fundamental * 1.15, sr / 2 - 100)
    if high <= low:
        return -60.0

    segment = y[int(start * sr) : max(int(end * sr), int(start * sr) + 1)]
    if segment.size == 0:
        return -60.0

    sos = scipy.signal.butter(4, [low, high], btype="band", fs=sr, output="sos")
    band = scipy.signal.sosfilt(sos, segment)
    rms = float(np.sqrt(np.mean(band**2)))
    return float(20 * np.log10(max(rms, 1e-6)))


def scale_velocities(energies: list[float]) -> list[int]:
    """Map a track's own energy spread onto the velocity range.

    Absolute dBFS thresholds do not survive contact with real stems: a separated stem is
    quiet, and a narrow band around one fundamental is quieter still, so every note pins
    to the floor. Normalising against the track's own distribution keeps dynamics
    meaningful within an instrument, which is what a learner reads.
    """
    if not energies:
        return []
    quiet, loud = np.percentile(energies, [10, 95])
    if loud - quiet < 6.0:  # genuinely flat playing, don't amplify noise into dynamics
        return [80] * len(energies)
    return [int(np.clip(np.interp(e, [quiet, loud], [45, 120]), 1, 127)) for e in energies]


def merge_held(notes: list[pretty_midi.Note], max_gap: float) -> list[pretty_midi.Note]:
    """Stitch re-onsets of the same pitch back into one sustained note.

    basic-pitch fragments a single held note into repeated onsets, which reads as a
    machine-gun repeat rather than a sustain. Ported from the audio2midi app, which hit
    the same thing and solved it the same way. Velocity of the merged note is the loudest
    fragment, since the attack carries the dynamic.
    """
    if not notes:
        return []

    ordered = sorted(notes, key=lambda n: (n.pitch, n.start))
    merged: list[pretty_midi.Note] = []
    for note in ordered:
        open_note = merged[-1] if merged else None
        if open_note is not None and open_note.pitch == note.pitch and note.start - open_note.end <= max_gap:
            open_note.end = max(open_note.end, note.end)
            open_note.velocity = max(open_note.velocity, note.velocity)
            continue
        merged.append(pretty_midi.Note(note.velocity, note.pitch, note.start, note.end))
    return sorted(merged, key=lambda n: n.start)


def transcribe_pitched(stem_path: Path, spec: StemSpec, velocity_mode: str, merge_gap: float) -> list[pretty_midi.Note]:
    """Run basic-pitch over one melodic stem and return playable notes."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, _, note_events = predict(
        str(stem_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=spec.onset_threshold,
        frame_threshold=spec.frame_threshold,
        minimum_note_length=spec.min_note_ms,
        minimum_frequency=spec.min_freq,
        maximum_frequency=spec.max_freq,
        melodia_trick=True,
    )

    audio = None
    sample_rate = 0
    if velocity_mode == "rms":
        audio, sample_rate = librosa.load(str(stem_path), sr=None, mono=True)

    kept = [
        (start, end, pitch, amplitude)
        for start, end, pitch, amplitude, _ in note_events
        if LOWEST_PITCH <= pitch <= HIGHEST_PITCH and end > start
    ]

    if velocity_mode == "rms" and audio is not None:
        energies = [note_energy_db(audio, sample_rate, s, e, p) for s, e, p, _ in kept]
        velocities = scale_velocities(energies)
    else:
        velocities = [int(np.clip(round(127 * a), 1, 127)) for *_, a in kept]

    notes = [
        pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        for (start, end, pitch, _), velocity in zip(kept, velocities, strict=True)
    ]
    return merge_held(notes, merge_gap)


def transcribe_drums(stem_path: Path) -> list[pretty_midi.Note]:
    """Crude three-band onset split into kick / snare / hi-hat.

    Deliberately not a real drum transcriber. It exists so the spike shows a rhythm
    track at all; a proper ADT model is a separate decision.
    """
    audio, sample_rate = librosa.load(str(stem_path), sr=22050, mono=True)
    bands = ((20.0, 150.0, DRUM_KICK), (180.0, 900.0, DRUM_SNARE), (4000.0, 10000.0, DRUM_HIHAT))

    notes: list[pretty_midi.Note] = []
    for low, high, drum in bands:
        sos = scipy.signal.butter(4, [low, high], btype="band", fs=sample_rate, output="sos")
        band = scipy.signal.sosfilt(sos, audio)
        envelope = librosa.onset.onset_strength(y=band, sr=sample_rate)
        if envelope.max() <= 0:
            continue
        frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sample_rate, backtrack=False, delta=0.25)
        times = librosa.frames_to_time(frames, sr=sample_rate)
        peak = float(envelope.max())
        for frame, onset in zip(frames, times, strict=False):
            strength = float(envelope[frame]) / peak
            velocity = int(np.clip(round(40 + 87 * strength), 1, 127))
            # Kit hits are one-shots and kinesthesia ignores their note-off entirely.
            notes.append(pretty_midi.Note(velocity=velocity, pitch=drum, start=onset, end=onset + 0.05))
    return notes


def build_midi(
    tracks: dict[str, list[pretty_midi.Note]],
    drums: list[pretty_midi.Note],
    tempo: float,
    programs: dict[str, int],
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for stem, notes in tracks.items():
        if not notes:
            continue
        spec = STEM_SPECS[stem]
        program = programs.get(stem, spec.program)
        instrument = pretty_midi.Instrument(program=program, is_drum=False, name=spec.label)
        instrument.notes.extend(notes)
        midi.instruments.append(instrument)
    if drums:
        kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        kit.notes.extend(drums)
        midi.instruments.append(kit)
    return midi


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--velocity", choices=("model", "rms"), default="rms")
    parser.add_argument("--skip-stems", nargs="*", default=[])
    parser.add_argument("--reuse-stems", action="store_true")
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.12,
        help="seconds; re-onsets of one pitch closer than this become a single held note",
    )
    parser.add_argument(
        "--program",
        nargs="*",
        default=[],
        metavar="STEM=GM",
        help="override a stem's General MIDI program, e.g. --program guitar=27 other=51",
    )
    args = parser.parse_args()

    programs = {k: int(v) for k, v in (p.split("=", 1) for p in args.program)}
    device = pick_device(args.device)
    work = args.out_dir / args.audio.stem
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    print(f"[spike] {args.audio.name} on {device}, velocity={args.velocity}")

    started = time.monotonic()
    audio, sample_rate = librosa.load(str(args.audio), sr=None, mono=True)
    duration = len(audio) / sample_rate
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=audio, sr=sample_rate)[0])[0])
    timings["analyse"] = time.monotonic() - started
    print(f"[spike] {duration:.1f}s audio, tempo ~{tempo:.1f} bpm")

    started = time.monotonic()
    stem_paths = separate(args.audio, work, device, args.reuse_stems)
    timings["separate"] = time.monotonic() - started
    print(f"[spike] separated {len(stem_paths)} stems in {timings['separate']:.1f}s")

    tracks: dict[str, list[pretty_midi.Note]] = {}
    for stem, spec in STEM_SPECS.items():
        if stem in args.skip_stems or stem not in stem_paths:
            continue
        started = time.monotonic()
        tracks[stem] = transcribe_pitched(stem_paths[stem], spec, args.velocity, args.merge_gap)
        timings[f"transcribe:{stem}"] = time.monotonic() - started
        print(
            f"[spike]   {spec.label:<8} {len(tracks[stem]):>5} notes  "
            f"GM {programs.get(stem, spec.program):>3}  {timings[f'transcribe:{stem}']:.1f}s"
        )

    drums: list[pretty_midi.Note] = []
    if "drums" in stem_paths and "drums" not in args.skip_stems:
        started = time.monotonic()
        drums = transcribe_drums(stem_paths["drums"])
        timings["transcribe:drums"] = time.monotonic() - started
        print(f"[spike]   {'Drums':<8} {len(drums):>5} hits   ch 10  {timings['transcribe:drums']:.1f}s")

    midi = build_midi(tracks, drums, tempo, programs)
    out_path = work / f"{args.audio.stem}.mid"
    midi.write(str(out_path))

    size = out_path.stat().st_size
    total = sum(timings.values())
    print(f"\n[spike] wrote {out_path} ({size / 1024:.0f} KB)")
    print(f"[spike] total {total:.1f}s for {duration:.1f}s audio = {duration / total:.2f}x realtime")
    if size > 5 * 1024 * 1024:
        print("[spike] WARNING: over kinesthesia's 5 MB limit, it will refuse to load this")


if __name__ == "__main__":
    main()
