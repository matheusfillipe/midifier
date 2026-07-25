"""Snap a transcribed MIDI to the song's real beat grid, and give it a bar structure.

The transformer emits absolute times with no notion of tempo, so the output has no bar
lines and notes sit a few tens of milliseconds off the beat. `beat_this` recovers the grid
from the audio in seconds (MIT, code and weights, CPU), which is enough to quantize, set a
tempo, and declare a time signature.

Quantization is partial by default. Snapping every note hard onto the grid strips the
timing that makes a performance sound human, and for a learning app a rigid grid is easier
to read but no longer the song. Notes far from any grid point are left alone entirely,
since those are usually ornaments or the model being wrong, and neither is improved by
dragging it onto a beat.

    uv run --no-project python scripts/quantize.py in.mid song.mp3 -o out.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pretty_midi

# Candidate subdivisions of a beat: quarter, eighth, sixteenth.
SUBDIVISIONS = (1, 2, 4)

# A note further than this fraction of a grid step from the nearest grid point is treated
# as deliberate, not sloppy, and is not moved.
SNAP_WINDOW = 0.5

# How far to drag a note toward the grid. 1.0 is a hard snap.
DEFAULT_STRENGTH = 0.6

# Fraction of notes that must sit close to a grid point for a subdivision to be believed.
FIT_THRESHOLD = 0.6


def beat_grid(audio: Path) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import File2Beats

    beats, downbeats = File2Beats(device="cpu")(str(audio))
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def subdivide(beats: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return beats
    steps = np.linspace(0.0, 1.0, factor, endpoint=False)
    grid = (beats[:-1, None] + np.diff(beats)[:, None] * steps[None, :]).ravel()
    return np.append(grid, beats[-1])


def pick_subdivision(onsets: np.ndarray, beats: np.ndarray) -> tuple[int, np.ndarray]:
    """Choose the coarsest subdivision the performance actually lands on.

    A finer grid always fits better, so accuracy alone would always pick sixteenths. Taking
    the coarsest grid that explains most of the notes keeps the result readable.
    """
    best = SUBDIVISIONS[-1]
    for factor in SUBDIVISIONS:
        grid = subdivide(beats, factor)
        step = np.median(np.diff(grid))
        distance = np.abs(onsets[:, None] - grid[None, :]).min(axis=1)
        if (distance < step * 0.1).mean() >= FIT_THRESHOLD:
            best = factor
            break
    return best, subdivide(beats, best)


def quantize(midi: pretty_midi.PrettyMIDI, grid: np.ndarray, strength: float) -> tuple[int, int]:
    step = float(np.median(np.diff(grid)))
    moved = untouched = 0
    for instrument in midi.instruments:
        for note in instrument.notes:
            nearest = grid[np.abs(grid - note.start).argmin()]
            offset = nearest - note.start
            if abs(offset) > step * SNAP_WINDOW:
                untouched += 1
                continue
            shift = offset * strength
            length = note.end - note.start
            note.start += shift
            # Percussion length is meaningless downstream, and dragging a sustained note's
            # tail independently of its onset would change how long it is held.
            note.end = note.start + length
            moved += 1
    return moved, untouched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    args = parser.parse_args()

    beats, downbeats = beat_grid(args.audio)
    tempo = 60.0 / float(np.median(np.diff(beats)))
    per_bar = round(len(beats) / max(len(downbeats), 1))
    print(f"grid: {len(beats)} beats, {len(downbeats)} downbeats, {tempo:.1f} bpm, {per_bar}/4")

    source = pretty_midi.PrettyMIDI(str(args.midi))
    onsets = np.array([n.start for i in source.instruments for n in i.notes])
    factor, grid = pick_subdivision(onsets, beats)
    print(f"subdivision: 1/{4 * factor} notes ({len(grid)} grid points)")

    # pretty_midi fixes tempo at construction, so the quantized song is rebuilt rather
    # than edited in place; that is also what writes a usable bar structure downstream.
    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    out.instruments = source.instruments
    out.time_signature_changes = [pretty_midi.TimeSignature(per_bar, 4, 0.0)]

    moved, untouched = quantize(out, grid, args.strength)
    print(f"snapped {moved} notes at strength {args.strength}, left {untouched} off-grid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
