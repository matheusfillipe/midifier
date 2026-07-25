"""Repair the decoder's characteristic defects in a transcribed MIDI.

Three defects, all observed on real output and all fixed by arithmetic on the notes, so
this stage costs nothing next to transcription:

* notes written past the end of the audio, because the decoder does not know where the
  song stops;
* a sustained note emitted as a stream of re-articulations, one per subdivision, which
  reads as machine-gun repetition;
* a short figure repeated far past the point any player would, which is the decoder
  looping rather than the music repeating.

Every rule is conservative. Real music repeats notes and holds them, so each threshold
sits well outside normal playing and only the excess is touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pretty_midi

# Longest repeating figure searched for, in notes. Past roughly a bar the search stops
# finding decoder loops and starts matching song structure.
MAX_PATTERN_PERIOD = 8

DEFAULT_MAX_CYCLES = 8
DEFAULT_MERGE_GAP = 0.05
DEFAULT_MAX_MERGED = 1.0

# Slack past the audio for a final chord to ring out.
END_TOLERANCE = 0.5


@dataclass(frozen=True)
class CleanupReport:
    """What the pass changed, for logging and for the job result."""

    trimmed_past_end: int
    merged_rearticulations: int
    cut_loops: int
    notes_before: int
    notes_after: int


def trim_overrun(midi: pretty_midi.PrettyMIDI, duration: float) -> int:
    """Drop notes starting after the audio ended, and clamp any still sounding."""
    limit = duration + END_TOLERANCE
    removed = 0
    for instrument in midi.instruments:
        keep = [note for note in instrument.notes if note.start < limit]
        removed += len(instrument.notes) - len(keep)
        for note in keep:
            note.end = min(note.end, limit)
        instrument.notes = keep
    return removed


def merge_held(
    midi: pretty_midi.PrettyMIDI,
    max_gap: float = DEFAULT_MERGE_GAP,
    max_length: float = DEFAULT_MAX_MERGED,
) -> int:
    """Rejoin re-articulations of one sustained note.

    `max_length` matters as much as `max_gap`: the fragments are usually touching, so the
    gap alone would happily fuse a whole phrase into one implausible sustain. Capping the
    result keeps a merged note the length of something a player would hold.

    Percussion is skipped, where repeated hits are the point and note length is ignored
    downstream anyway.
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
            notes.sort(key=lambda note: note.start)
            open_note = notes[0]
            kept.append(open_note)
            for note in notes[1:]:
                joined = max(open_note.end, note.end) - open_note.start
                if note.start - open_note.end <= max_gap and joined <= max_length:
                    open_note.end = max(open_note.end, note.end)
                    open_note.velocity = max(open_note.velocity, note.velocity)
                    removed += 1
                    continue
                open_note = note
                kept.append(open_note)
        instrument.notes = sorted(kept, key=lambda note: note.start)
    return removed


def trim_loops(midi: pretty_midi.PrettyMIDI, max_cycles: int = DEFAULT_MAX_CYCLES) -> int:
    """Cut figures that repeat past `max_cycles`.

    Counting one pitch repeated in a row misses the usual case, where the decoder loops on
    a short figure (A B A B ...) and resets a same-pitch counter on every note. Testing
    periodicity instead catches both, a stuck single pitch being the period-one case.
    """
    removed = 0
    for instrument in midi.instruments:
        ordered = sorted(instrument.notes, key=lambda note: note.start)
        pitches = [note.pitch for note in ordered]
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

        instrument.notes = [note for index, note in enumerate(ordered) if index not in doomed]
        removed += len(doomed)
    return removed


def clean(
    midi: pretty_midi.PrettyMIDI,
    duration: float,
    *,
    merge_gap: float = DEFAULT_MERGE_GAP,
    max_merged: float = DEFAULT_MAX_MERGED,
    max_cycles: int = DEFAULT_MAX_CYCLES,
) -> CleanupReport:
    """Run every repair, in the order they depend on each other."""
    before = sum(len(instrument.notes) for instrument in midi.instruments)
    trimmed = trim_overrun(midi, duration)
    merged = merge_held(midi, merge_gap, max_merged)
    loops = trim_loops(midi, max_cycles)
    after = sum(len(instrument.notes) for instrument in midi.instruments)
    return CleanupReport(
        trimmed_past_end=trimmed,
        merged_rearticulations=merged,
        cut_loops=loops,
        notes_before=before,
        notes_after=after,
    )
