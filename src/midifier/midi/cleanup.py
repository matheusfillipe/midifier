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

import itertools
import statistics
from dataclasses import dataclass

import pretty_midi

# Longest repeating figure searched for, in notes. Past roughly a bar the search stops
# finding decoder loops and starts matching song structure.
MAX_PATTERN_PERIOD = 8

DEFAULT_MAX_CYCLES = 8
DEFAULT_MERGE_GAP = 0.05
DEFAULT_MAX_MERGED = 1.0

# Slack past the audio for a final chord to ring out.
END_TOLERANCE = 0.5

# Only the last stretch of a decode is trimmed for repetition. The decoder degenerates as it
# runs out of song; a figure repeated in the middle is the music doing it.
DEFAULT_TAIL_SECONDS = 20.0

# Notes starting this close together are one chord.
CHORD_WINDOW = 0.05

# A chord this wide, repeated identically to the end, is the decoder locked on it. Width is
# what separates the two: measured across real output the locked chord is 13 notes, while
# honest repetition in the same window is a drum figure two notes wide.
MIN_STUCK_CHORD = 4
MIN_STUCK_CYCLES = 4

# A part counts as playing a steady figure when this share of its onsets sit within
# PULSE_TOLERANCE of its own median spacing. Note *lengths* cannot make this judgement:
# they are note values at the song's tempo, so any fixed threshold is a tempo threshold and
# picks a different answer on every song. Spacing regularity does not move with tempo.
PULSE_TOLERANCE = 0.35
PULSE_SHARE = 0.40
MIN_ONSETS_FOR_PULSE = 10
MAX_PULSE_GAP = 2.0


@dataclass(frozen=True)
class CleanupReport:
    """What the pass changed, for logging and for the job result."""

    trimmed_past_end: int
    merged_rearticulations: int
    cut_loops: int
    notes_before: int
    notes_after: int
    pulse_tracks: list[str]
    cut_stuck_chord: int = 0


def plays_a_pulse(instrument: pretty_midi.Instrument) -> bool:
    """Whether this part repeats a steady figure, so its repeats are music and not a defect.

    A driving bass is indistinguishable from decoder stutter by note geometry alone -- both are
    the same pitch, repeated, touching. What separates them is that the player keeps time.
    """
    ordered = sorted(instrument.notes, key=lambda note: note.start)
    gaps = [
        later.start - earlier.start
        for earlier, later in itertools.pairwise(ordered)
        if 0.0 < later.start - earlier.start < MAX_PULSE_GAP
    ]
    if len(gaps) < MIN_ONSETS_FOR_PULSE:
        return False
    spacing = statistics.median(gaps)
    steady = sum(1 for gap in gaps if abs(gap - spacing) < PULSE_TOLERANCE * spacing)
    return steady / len(gaps) >= PULSE_SHARE


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
    downstream anyway, and so is any part keeping a steady pulse -- fusing those turns a
    driving bass into a drone, which reads as the part having gone missing.
    """
    removed = 0
    for instrument in midi.instruments:
        if instrument.is_drum or plays_a_pulse(instrument):
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


def trim_loops(
    midi: pretty_midi.PrettyMIDI,
    duration: float,
    max_cycles: int = DEFAULT_MAX_CYCLES,
    tail: float = DEFAULT_TAIL_SECONDS,
) -> int:
    """Cut figures that repeat past `max_cycles`, in the closing stretch only.

    Counting one pitch repeated in a row misses the usual case, where the decoder loops on
    a short figure (A B A B ...) and resets a same-pitch counter on every note. Testing
    periodicity instead catches both, a stuck single pitch being the period-one case.

    The window matters as much as the test. Applied to a whole song this deletes real
    playing: a repetitive section reads as periodic because it is. Degeneration happens as
    the decode runs out of audio, so only the closing seconds are eligible.
    """
    cutoff = duration - tail
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
                if run > period * max_cycles and ordered[index].start >= cutoff:
                    doomed.add(index)

        instrument.notes = [note for index, note in enumerate(ordered) if index not in doomed]
        removed += len(doomed)
    return removed


def _chord_events(notes: list[pretty_midi.Note]) -> list[tuple[frozenset[int], list[pretty_midi.Note]]]:
    ordered = sorted(notes, key=lambda note: note.start)
    events = []
    for _, group in itertools.groupby(ordered, key=lambda note: round(note.start / CHORD_WINDOW)):
        together = list(group)
        events.append((frozenset(note.pitch for note in together), together))
    return events


def _repeating_tail(events: list[tuple[frozenset[int], list[pretty_midi.Note]]]) -> tuple[int, int]:
    """The (period, matches) of the longest identical block repeating up to the last event."""
    best = (0, 0)
    for period in range(1, MAX_PATTERN_PERIOD + 1):
        matches = 0
        index = len(events) - 1
        while index - period >= 0 and events[index][0] == events[index - period][0]:
            matches += 1
            index -= 1
        if matches // period > best[1] // max(best[0], 1):
            best = (period, matches)
    return best


def trim_stuck_chord(
    midi: pretty_midi.PrettyMIDI,
    duration: float,
    tail: float = DEFAULT_TAIL_SECONDS,
) -> int:
    """Cut a wide chord the decoder repeats to the end of the file, leaving one of them.

    `trim_loops` cannot see this. It tests a flat sequence of pitches, where notes sounding
    together make the period as wide as the chord and any note interleaved between repeats
    resets the count -- so a thirteen-note chord alternating with a single note scores a run
    of zero at every period it searches.

    One cycle survives so the song still ends on the chord, and so that misjudging honest
    playing costs the repeats rather than the passage.
    """
    cutoff = duration - tail
    removed = 0
    for instrument in midi.instruments:
        events = _chord_events([note for note in instrument.notes if note.start >= cutoff])
        period, matches = _repeating_tail(events)
        if not period or matches // period < MIN_STUCK_CYCLES:
            continue
        run = events[len(events) - matches - period :]
        if max(len(pitches) for pitches, _ in run) < MIN_STUCK_CHORD:
            continue
        doomed = {id(note) for _, group in run[period:] for note in group}
        instrument.notes = [note for note in instrument.notes if id(note) not in doomed]
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
    # Recorded before merging, since merging is what would flatten the evidence.
    pulses = [i.name for i in midi.instruments if i.notes and not i.is_drum and plays_a_pulse(i)]
    merged = merge_held(midi, merge_gap, max_merged)
    loops = trim_loops(midi, duration, max_cycles)
    stuck = trim_stuck_chord(midi, duration)
    after = sum(len(instrument.notes) for instrument in midi.instruments)
    return CleanupReport(
        trimmed_past_end=trimmed,
        merged_rearticulations=merged,
        cut_loops=loops,
        notes_before=before,
        notes_after=after,
        pulse_tracks=pulses,
        cut_stuck_chord=stuck,
    )
