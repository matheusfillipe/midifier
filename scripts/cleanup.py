"""Trim decoder degeneration from a transcribed MIDI, without touching the good parts.

Greedy decoding transcribes better than temperature sampling here, but it degenerates:
on the test track the bass emitted 35 consecutive identical pitches near the end and the
file ran 5s past the audio. Sampling avoids that but makes the rest worse, so the fix is
to keep greedy and delete only the degenerate spans.

Both rules are deliberately conservative. Real music repeats notes, so only runs far
longer than any musical figure are cut, and only the excess is removed.

    uv run scripts/cleanup.py in.mid original.mp3 -o out.mid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import pretty_midi

# Longest repeating figure we look for, in notes. Beyond a bar or so of pattern the
# search stops being about decoder loops and starts matching song structure.
MAX_PATTERN_PERIOD = 8

# How many times a figure may repeat before further repeats are treated as degeneration.
DEFAULT_MAX_CYCLES = 8

# Same-pitch notes closer than this are one sustained note the decoder re-articulated.
DEFAULT_MERGE_GAP = 0.05

# A merge is only ever fusing re-articulations of one sustained note, so the result should
# still look like a note somebody played. Past this the join is inventing a sustain.
DEFAULT_MAX_MERGED = 1.0

# The decoder can emit notes past the end of the audio. Allow a little ring-out.
END_TOLERANCE = 0.5


def trim_overrun(midi: pretty_midi.PrettyMIDI, duration: float) -> int:
    limit = duration + END_TOLERANCE
    removed = 0
    for instrument in midi.instruments:
        keep = [n for n in instrument.notes if n.start < limit]
        removed += len(instrument.notes) - len(keep)
        for note in keep:
            note.end = min(note.end, limit)
        instrument.notes = keep
    return removed


def merge_held(midi: pretty_midi.PrettyMIDI, max_gap: float, max_length: float) -> int:
    """Rejoin a sustained note that was emitted as a stream of re-articulations.

    The decoder re-states a held note on every subdivision instead of letting it ring, so a
    sustained brass chord arrives as the same chord struck every eighth. Measured on the
    test track, roughly 40% of consecutive same-pitch pairs sat under 50ms apart, which is
    contiguous rather than re-struck. Merging those restores the sustain and removes what
    reads as machine-gun repetition.

    Percussion is skipped: a drum note's length carries no meaning and repeated hits are
    the point.
    """
    removed = 0
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        by_pitch: dict[int, list[pretty_midi.Note]] = {}
        for note in instrument.notes:
            by_pitch.setdefault(note.pitch, []).append(note)

        kept: list[pretty_midi.Note] = []
        for notes in by_pitch.values():
            notes.sort(key=lambda n: n.start)
            open_note = notes[0]
            kept.append(open_note)
            for note in notes[1:]:
                joined_length = max(open_note.end, note.end) - open_note.start
                if note.start - open_note.end <= max_gap and joined_length <= max_length:
                    open_note.end = max(open_note.end, note.end)
                    open_note.velocity = max(open_note.velocity, note.velocity)
                    removed += 1
                    continue
                open_note = note
                kept.append(open_note)
        instrument.notes = sorted(kept, key=lambda n: n.start)
    return removed


def trim_repeat_runs(midi: pretty_midi.PrettyMIDI, max_cycles: int) -> int:
    """Cut looping patterns, not just single repeated pitches.

    Counting one pitch repeated in a row misses the common case: the decoder loops on a
    short figure (A B A B A B ...), which resets a same-pitch counter on every note and so
    survives untouched however long it runs. Looking for periodicity instead catches both,
    since a stuck single pitch is just a pattern of length one.

    A note is cut when it continues a period-p repetition that has already played
    `max_cycles` times over. Real music is repetitive, so the allowance is generous and the
    shortest period wins, which is the one a listener actually perceives as a loop.
    """
    removed = 0
    for instrument in midi.instruments:
        ordered = sorted(instrument.notes, key=lambda n: n.start)
        pitches = [n.pitch for n in ordered]
        doomed: set[int] = set()

        for period in range(1, MAX_PATTERN_PERIOD + 1):
            run = 0
            for index in range(len(pitches)):
                if index >= period and pitches[index] == pitches[index - period]:
                    run += 1
                else:
                    run = 0
                if run > period * max_cycles:
                    doomed.add(index)

        if doomed:
            instrument.notes = [n for i, n in enumerate(ordered) if i not in doomed]
            removed += len(doomed)
        else:
            instrument.notes = ordered
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=DEFAULT_MERGE_GAP,
        help="seconds; same-pitch notes closer than this rejoin into one held note",
    )
    parser.add_argument(
        "--max-merged",
        type=float,
        default=DEFAULT_MAX_MERGED,
        help="seconds; never merge fragments into a note longer than this",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=DEFAULT_MAX_CYCLES,
        help="how many times a figure may repeat before it counts as a decoder loop",
    )
    args = parser.parse_args()

    midi = pretty_midi.PrettyMIDI(str(args.midi))
    duration = float(librosa.get_duration(path=str(args.audio)))
    before = sum(len(i.notes) for i in midi.instruments)

    overrun = trim_overrun(midi, duration)
    merged = merge_held(midi, args.merge_gap, args.max_merged)
    repeats = trim_repeat_runs(midi, args.max_cycles)

    after = sum(len(i.notes) for i in midi.instruments)
    print(f"audio {duration:.1f}s, midi ended {midi.get_end_time():.1f}s")
    print(f"removed {overrun} past-the-end, {repeats} runaway repeats  ({before} -> {after} notes)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
