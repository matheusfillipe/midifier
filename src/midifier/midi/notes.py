"""Reading groups of notes out of a decode, for the passes that repair one."""

import pretty_midi

# Onsets this close are one chord. Chaining each note against the one before it rather than
# bucketing the timeline keeps two notes a millisecond apart together wherever they fall.
CHORD_WINDOW = 0.06


def chords(notes: list[pretty_midi.Note]) -> list[list[pretty_midi.Note]]:
    """Notes struck together, each group ordered lowest first."""
    grouped: list[list[pretty_midi.Note]] = []
    for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if grouped and note.start - grouped[-1][0].start <= CHORD_WINDOW:
            grouped[-1].append(note)
        else:
            grouped.append([note])
    return grouped
