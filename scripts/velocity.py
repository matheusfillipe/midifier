"""Give a flat MIDI real dynamics, measured from the audio it was transcribed from.

Transcription models mostly do not emit velocity: MuScriptor writes a constant 100, and
basic-pitch's "amplitude" is really mean frame-activation, i.e. model confidence rather
than loudness. Both are fixable the same way and independently of the transcriber, by
measuring what the recording actually does at each note.

    uv run scripts/velocity.py flat.mid original.mp3 -o dynamic.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import scipy.signal

# Percussion carries the beat, so a wider spread reads better than the melodic range.
DRUM_RANGE = (55, 127)
PITCHED_RANGE = (45, 120)

# Below this dB spread a track is genuinely played flat, and stretching it would
# amplify measurement noise into fake dynamics.
MIN_DB_SPREAD = 6.0


def pitched_energy_db(audio: np.ndarray, sr: int, note: pretty_midi.Note) -> float:
    """Energy in a narrow band around the note's fundamental."""
    fundamental = librosa.midi_to_hz(note.pitch)
    low = max(fundamental * 0.85, 20.0)
    high = min(fundamental * 1.15, sr / 2 - 100)
    if high <= low:
        return -60.0

    start = int(note.start * sr)
    end = max(int(note.end * sr), start + 1)
    segment = audio[start:end]
    if segment.size < 16:
        return -60.0

    sos = scipy.signal.butter(4, [low, high], btype="band", fs=sr, output="sos")
    band = scipy.signal.sosfilt(sos, segment)
    return float(20 * np.log10(max(float(np.sqrt(np.mean(band**2))), 1e-6)))


def drum_energy_db(audio: np.ndarray, sr: int, note: pretty_midi.Note, window: float = 0.05) -> float:
    """Broadband energy in the attack window.

    A drum hit has no fundamental to band around, and its MIDI duration is meaningless
    (kinesthesia ignores percussion note-off entirely), so measure the transient itself.
    """
    start = int(note.start * sr)
    end = min(int((note.start + window) * sr), audio.size)
    segment = audio[start:end]
    if segment.size < 16:
        return -60.0
    return float(20 * np.log10(max(float(np.sqrt(np.mean(segment**2))), 1e-6)))


def scale(energies: list[float], low: int, high: int) -> list[int]:
    """Map a track's own energy spread onto a velocity range.

    Absolute dBFS thresholds do not survive contact with real audio: a quiet mix, or a
    narrow band around one fundamental, pins every note to the floor. Normalising per
    track keeps dynamics meaningful within an instrument, which is what a player reads.
    """
    if not energies:
        return []
    quiet, loud = np.percentile(energies, [10, 95])
    if loud - quiet < MIN_DB_SPREAD:
        return [(low + high) // 2] * len(energies)
    return [int(np.clip(np.interp(e, [quiet, loud], [low, high]), 1, 127)) for e in energies]


def apply_velocity(midi: pretty_midi.PrettyMIDI, audio: np.ndarray, sr: int) -> None:
    for instrument in midi.instruments:
        if not instrument.notes:
            continue
        if instrument.is_drum:
            energies = [drum_energy_db(audio, sr, n) for n in instrument.notes]
            velocities = scale(energies, *DRUM_RANGE)
        else:
            energies = [pitched_energy_db(audio, sr, n) for n in instrument.notes]
            velocities = scale(energies, *PITCHED_RANGE)
        for note, velocity in zip(instrument.notes, velocities, strict=True):
            note.velocity = velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    midi = pretty_midi.PrettyMIDI(str(args.midi))
    audio, sr = librosa.load(str(args.audio), sr=None, mono=True)

    before = {i.name: sorted({n.velocity for n in i.notes}) for i in midi.instruments}
    apply_velocity(midi, audio, sr)

    for instrument in midi.instruments:
        velocities = [n.velocity for n in instrument.notes]
        if not velocities:
            continue
        was = before[instrument.name]
        was_text = str(was[0]) if len(was) == 1 else f"{was[0]}-{was[-1]}"
        print(f"{instrument.name:<28} {len(velocities):>5} notes  {was_text:>7} -> {min(velocities)}-{max(velocities)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.output))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
