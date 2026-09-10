"""Decide the song's parts once, from the whole song, and put every note in one of them.

The decoder names its lanes per segment and independently, and it is not consistent about it:
on one song it split a five piece band into nine lanes in one segment and crushed it into a
single lane three segments later. Carrying those names across the joins, then folding the
duplicates afterwards, repairs the symptom twice and never decides what the parts are.

What is stable is the parts. A bass keeps to its register for the length of a song, a strummed
part keeps playing chords, a sung line keeps to one note at a time, a pad holds its chords where
a rhythm guitar cuts them short. So the parts are learned from the segments where the decoder
did separate the band, and every note in the song is then placed in the part it fits. One rule
both splits a lane that swallowed the band and merges the lanes that are one part under several
names, because those are the same mistake seen from either side.
"""

import itertools
import statistics
from typing import Self

import pretty_midi

from midifier.midi.notes import chords

# Every note we have placed so far, each with the segment it was decoded in.
type Owned = list[tuple[pretty_midi.Note, int]]

# Above this many notes sounding at once a part is playing chords rather than a single line.
CHORDAL = 2.5

# A lane holding more than this many semitones is covering more than one part. An octave either
# side of a line is ordinary; three octaves is the whole band in one lane, and its own texture
# and register then describe nothing.
COLLAPSED_SEMITONES = 36

# Two chordal parts in one register are still two parts when one holds its chords and the other
# strums, and note length is all that separates them once register and texture agree.
HELD_RATIO = 2.0

# How far outside a part's register a note may fall and still belong to it.
REGISTER_SLACK = 6

# A chord narrower than this is one instrument's, however many notes are in it: two hands reach
# about two octaves. Wider than that and it is a bass note, harmony, and a melody note together.
SPREAD_CHORD = 24

# A note this far from the rest of a chord is a line of its own rather than part of the voicing.
LINE_GAP = 7

# How far a note's distance from a part's centre counts against it, next to falling outside
# the part's register altogether, which is the stronger signal.
CENTRE_PULL = 4

# What playing the wrong texture costs a note, in semitones of register. Above two octaves so
# that a part never takes a note whose texture says it belongs to another.
TEXTURE_PENALTY = 12

# A part whose middle half sits inside this many semitones is a fragment of another one.
MIN_PART_SPREAD = 5

# A lane reaching across fewer than this many semitones is not an instrument playing a line.
MIN_PART_RANGE = 5

# Under this many notes in a whole song a lane is not worth a track of its own.
MIN_PART_NOTES = 12


def _polyphony(notes: list[pretty_midi.Note]) -> float:
    ordered = sorted(notes, key=lambda note: note.start)
    return statistics.mean(sum(1 for other in ordered if other.start <= note.start < other.end) for note in ordered)


def _quartiles(pitches: list[int]) -> tuple[int, int, int]:
    ranked = sorted(pitches)
    return ranked[len(ranked) // 4], ranked[len(ranked) // 2], ranked[3 * len(ranked) // 4]


def _is_collapsed(notes: list[pretty_midi.Note]) -> bool:
    pitches = [int(note.pitch) for note in notes]
    return max(pitches) - min(pitches) > COLLAPSED_SEMITONES


class Label:
    """How much of a lane one of the decoder's names accounts for.

    A name from a lane holding the whole band counts its notes but not its segment, so it ranks
    below every name the decoder gave a part it had actually separated. It still has to count
    for something: a song the decoder never separates at all -- a solo piano, where one hand
    reaches well past a part's register -- has no other name to offer.
    """

    def __init__(self, program: int) -> None:
        self.program = program
        self.segments: set[int] = set()
        self.notes = 0

    def saw(self, segment: int, notes: int, *, separated: bool) -> None:
        if separated:
            self.segments.add(segment)
        self.notes += notes


class Part:
    """One instrument of the song: where it sits and how it plays."""

    def __init__(self, notes: list[pretty_midi.Note]) -> None:
        self.low, self.middle, self.high = _quartiles([note.pitch for note in notes])
        self.chordal = _polyphony(notes) >= CHORDAL
        self.held = statistics.median(note.end - note.start for note in notes)

    def is_twin(self, other: Self) -> bool:
        return (
            self.chordal == other.chordal
            and min(self.high, other.high) - max(self.low, other.low) > -REGISTER_SLACK
            and max(self.held, other.held) / min(self.held, other.held) < HELD_RATIO
        )

    def absorb(self, other: Self) -> None:
        self.low = min(self.low, other.low)
        self.high = max(self.high, other.high)

    def distance(self, pitch: int, chordal: bool) -> float:
        """How badly a note fits, in semitones, with a penalty for the wrong texture."""
        outside = max(0, self.low - REGISTER_SLACK - pitch, pitch - self.high - REGISTER_SLACK)
        return outside + abs(pitch - self.middle) / CENTRE_PULL + (0 if chordal == self.chordal else TEXTURE_PENALTY)


class Fragment:
    """One lane of one segment, moved to where it belongs in the song."""

    def __init__(self, lane: pretty_midi.Instrument, offset: float, segment: int) -> None:
        self.notes = [
            pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start + offset, end=note.end + offset)
            for note in lane.notes
        ]
        self.name = lane.name or "unnamed"
        self.program = lane.program
        self.is_drum = lane.is_drum
        self.segment = segment
        self.collapsed = _is_collapsed(self.notes)


def _learn(fragments: list[Fragment]) -> list[Part]:
    """The song's parts, from the lanes the decoder did keep separate."""
    separated = [fragment for fragment in fragments if not fragment.is_drum and not fragment.collapsed]
    parts: list[Part] = []
    for fragment in sorted(separated, key=lambda f: -len(f.notes)):
        candidate = Part(fragment.notes)
        twin = next((part for part in parts if part.is_twin(candidate)), None)
        if twin is None:
            parts.append(candidate)
        else:
            twin.absorb(candidate)

    return [part for part in parts if part.high - part.low >= MIN_PART_SPREAD]


def _name_lanes(lanes: list[pretty_midi.Instrument], sources: list[dict[str, Label]]) -> None:
    """Name each lane after the label that put the most of the song into it.

    Naming from the parts that formed a lane instead names it after evidence that may have
    ended up somewhere else. What a lane is, is what went into it. A label the decoder used in
    several segments outranks one it used once, however many notes that one carried, and a
    label belongs to a single lane: two lanes under one name describes at most one of them.
    """
    taken: set[str] = set()
    for index in sorted(range(len(lanes)), key=lambda position: -len(lanes[position].notes)):
        ranked = sorted(sources[index].items(), key=lambda item: (len(item[1].segments), item[1].notes), reverse=True)
        for name, label in ranked:
            if name in taken:
                continue
            lanes[index].name = name
            lanes[index].program = label.program
            taken.add(name)
            break


def _roles(chord: list[pretty_midi.Note], parts: list[Part]) -> list[tuple[pretty_midi.Note, Part]]:
    """Hand a chord's notes out by the job each is doing in it.

    Only a chord reaching across the band is several parts at once. One an instrument could
    play with two hands is one instrument's chord, and taking its lowest note for the bass and
    its highest for the singer leaves three parts each holding a third of it, all playing all
    the time. A note is only a line of its own when a gap separates it from the rest.
    """
    lines = [part for part in parts if not part.chordal]
    lowest = min(lines, key=lambda p: p.middle) if lines else None
    highest = max(lines, key=lambda p: p.middle) if lines else None

    left = list(chord)
    placed: list[tuple[pretty_midi.Note, Part]] = []
    if left[-1].pitch - left[0].pitch > SPREAD_CHORD:
        below = len(left) > 1 and left[1].pitch - left[0].pitch >= LINE_GAP
        if below and lowest is not None and left[0].pitch <= lowest.high + REGISTER_SLACK:
            placed.append((left.pop(0), lowest))
        above = len(left) > 1 and left[-1].pitch - left[-2].pitch >= LINE_GAP
        if above and highest is not None and highest is not lowest and left[-1].pitch >= highest.low - REGISTER_SLACK:
            placed.append((left.pop(), highest))

    if not placed and len(left) > 1:
        # The chord stays whole, in the part whose register its body sits in.
        middle = left[len(left) // 2].pitch
        together = min(parts, key=lambda p: p.distance(middle, True))
        return [(note, together) for note in left]

    placed.extend((note, min(parts, key=lambda p: p.distance(note.pitch, len(left) > 1))) for note in left)
    return placed


def _resolve_overlap(owned: Owned, seams: list[float]) -> list[pretty_midi.Note]:
    """Both neighbours transcribe the overlap, so only one of them may keep it.

    The arriving segment wins where it has this part, because a decode degenerates as it runs
    and is at its freshest where its neighbour is at its worst. Where it does not have the part,
    the outgoing segment keeps playing: a segment that lost a part has nothing fresher to put
    there, and dropping the notes anyway is how a drum kit falls silent at a seam and stays
    silent for the rest of the song.
    """
    # Every note a segment holds starts inside that segment, so a segment appearing here at all
    # is one that transcribed this part and can take over the overlap before it.
    present = {segment for _, segment in owned}
    kept: list[pretty_midi.Note] = []
    for note, segment in owned:
        seam = seams[segment] if segment < len(seams) else None
        if seam is None or note.start < seam or segment + 1 not in present:
            kept.append(note)
    return kept


def _route(fragments: list[Fragment], parts: list[Part]) -> tuple[list[Owned], list[dict[str, Label]], Owned]:
    """Put every note in a part, and record which of the decoder's labels put it there."""
    owned: list[Owned] = [[] for _ in parts]
    sources: list[dict[str, Label]] = [{} for _ in parts]
    drums: Owned = []

    for fragment in fragments:
        if fragment.is_drum:
            drums.extend((note, fragment.segment) for note in fragment.notes)
            continue

        placed: list[tuple[int, pretty_midi.Note]] = []
        if fragment.collapsed:
            index_of = {id(part): index for index, part in enumerate(parts)}
            for chord in chords(fragment.notes):
                placed.extend((index_of[id(part)], note) for note, part in _roles(chord, parts))
        else:
            # A lane the decoder kept to one part stays whole: splitting it again trades its
            # judgement for ours, and ours is only better where its own broke down.
            texture = _polyphony(fragment.notes) >= CHORDAL
            middle = _quartiles([note.pitch for note in fragment.notes])[1]
            best = min(range(len(parts)), key=lambda index: parts[index].distance(middle, texture))
            placed.extend((best, note) for note in fragment.notes)

        carried: dict[int, int] = {}
        for index, note in placed:
            owned[index].append((note, fragment.segment))
            carried[index] = carried.get(index, 0) + 1
        for index, count in carried.items():
            label = sources[index].setdefault(fragment.name, Label(fragment.program))
            label.saw(fragment.segment, count, separated=not fragment.collapsed)

    return owned, sources, drums


def assemble(
    decoded: list[tuple[float, pretty_midi.PrettyMIDI]],
) -> tuple[pretty_midi.PrettyMIDI, list[tuple[str, str]]]:
    """Build one song from independently decoded segments; returns it and the names folded away."""
    fragments = [
        Fragment(lane, offset, index)
        for index, (offset, segment) in enumerate(decoded)
        for lane in segment.instruments
        if lane.notes
    ]
    if not fragments:
        return pretty_midi.PrettyMIDI(), []

    seams = [offset for offset, _ in decoded[1:]]
    parts = _learn(fragments) or [Part(fragments[0].notes)]
    owned, sources, drums = _route(fragments, parts)

    lanes: list[pretty_midi.Instrument] = []
    kept: list[dict[str, Label]] = []
    for notes, labels in zip(owned, sources, strict=True):
        lane = pretty_midi.Instrument(program=0, name="")
        lane.notes = _resolve_overlap(notes, seams)
        if not lane.notes or not labels:
            continue
        pitches = [note.pitch for note in lane.notes]
        # Judged on what it ended up holding, since a part is what it plays.
        if len(lane.notes) >= MIN_PART_NOTES and max(pitches) - min(pitches) >= MIN_PART_RANGE:
            lanes.append(lane)
            kept.append(labels)

    _name_lanes(lanes, kept)

    # A name can still fall to two lanes when one of them has no other to offer.
    for name, group in itertools.groupby(sorted(lanes, key=lambda lane: lane.name), key=lambda lane: lane.name):
        for number, lane in enumerate(list(group)[1:], start=2):
            lane.name = f"{name} {number}"

    kit = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    kit.notes = _resolve_overlap(drums, seams)
    if kit.notes:
        lanes.append(kit)

    song = pretty_midi.PrettyMIDI()
    for lane in lanes:
        lane.notes.sort(key=lambda note: note.start)
        song.instruments.append(lane)

    # A label can put notes in several lanes; it is reported against the one that took most of
    # it, so the caller sees each name the decoder used at most once.
    surviving = {lane.name for lane in lanes}
    biggest: dict[str, tuple[int, str]] = {}
    for lane, labels in zip(lanes, kept, strict=False):
        for name, label in labels.items():
            if name in surviving or name == lane.name:
                continue
            if label.notes > biggest.get(name, (0, ""))[0]:
                biggest[name] = (label.notes, lane.name)
    return song, [(name, into) for name, (_, into) in biggest.items()]
