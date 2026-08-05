"""Stitch independently decoded segments back into one song.

Long decodes drift. The model works in five-second chunks and carries each chunk's opening
from the previous one, so a part it loses stays lost and an invented ending runs to the last
chunk. Decoding in segments caps how far either can travel, and the closing segment stops on
its own rather than filling its budget with noise.

The price is the joins. A segment decoded alone has no reason to name a part the way its
neighbour did, so without repair the entire band appears to change instrument at every seam.
The overlap pays for it: both segments transcribe those seconds, so a lane can be recognised
by what it plays there and inherit the earlier name.
"""

from collections.abc import Iterable

import pretty_midi

# Shared seconds between neighbouring segments, used only to match lanes across the join.
OVERLAP_SECONDS = 5.0

# Slack for calling two notes the same event, and how many matches make a confident pairing.
ONSET_TOLERANCE = 0.15
MIN_AGREEMENT = 3


def plan(duration: float, length: float) -> list[float]:
    """Start offsets covering `duration`, each segment overlapping the last."""
    step = length - OVERLAP_SECONDS
    offsets = [0.0]
    while offsets[-1] + length < duration:
        offsets.append(offsets[-1] + step)
    return offsets


def _name(instrument: pretty_midi.Instrument) -> str:
    return instrument.name or "unnamed"


def _agreement(head: list[pretty_midi.Note], tail: list[pretty_midi.Note]) -> int:
    return sum(1 for n in head if any(x.pitch == n.pitch and abs(x.start - n.start) < ONSET_TOLERANCE for x in tail))


def _inherited_names(
    midi: pretty_midi.PrettyMIDI,
    previous: dict[str, list[pretty_midi.Note]],
    is_drum: dict[str, bool],
) -> dict[str, str]:
    """Map this segment's lane names onto the previous segment's, by what they play in common."""
    scored = sorted(
        (
            (_agreement([n for n in inst.notes if n.start < OVERLAP_SECONDS], tail), _name(inst), name, inst.is_drum)
            for inst in midi.instruments
            for name, tail in previous.items()
        ),
        reverse=True,
    )
    renames: dict[str, str] = {}
    claimed: set[str] = set()
    for score, mine, theirs, drum in scored:
        if score < MIN_AGREEMENT or mine in renames or theirs in claimed or is_drum.get(theirs) != drum:
            continue
        renames[mine] = theirs
        claimed.add(theirs)

    # Inheriting a name is a permutation, not a substitution. If this segment already has a
    # lane under an inherited name, that lane is a different part and needs its own.
    for inst in midi.instruments:
        if _name(inst) in renames or _name(inst) not in claimed:
            continue
        spare = f"{_name(inst)} 2"
        while spare in claimed:
            spare += " 2"
        renames[_name(inst)] = spare
        claimed.add(spare)
    return renames


def stitch(parts: Iterable[tuple[float, pretty_midi.PrettyMIDI]], length: float) -> pretty_midi.PrettyMIDI:
    """Shift each segment into place and merge, keeping lane identity across the joins."""
    merged = pretty_midi.PrettyMIDI()
    lanes: dict[str, pretty_midi.Instrument] = {}
    is_drum: dict[str, bool] = {}
    programs: dict[str, int] = {}
    previous: dict[str, list[pretty_midi.Note]] = {}

    for index, (offset, segment) in enumerate(parts):
        renames = _inherited_names(segment, previous, is_drum) if previous else {}
        # Both segments transcribe the overlap, and a decode degenerates as it runs: parts
        # drop out one by one and never come back. The arriving segment is at its freshest
        # exactly where the previous one is at its worst, so it replaces those seconds.
        if index:
            for lane in lanes.values():
                lane.notes = [note for note in lane.notes if note.start < offset]

        for inst in segment.instruments:
            name = renames.get(_name(inst), _name(inst))
            is_drum.setdefault(name, inst.is_drum)
            programs.setdefault(name, inst.program)
            lane = lanes.get(name)
            if lane is None:
                lane = pretty_midi.Instrument(program=programs[name], is_drum=is_drum[name], name=name)
                lanes[name] = lane
                merged.instruments.append(lane)
            lane.notes.extend(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + offset,
                    end=note.end + offset,
                )
                for note in inst.notes
            )

        previous = {}
        for inst in segment.instruments:
            tail = [
                pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start - (length - OVERLAP_SECONDS),
                    end=n.end - (length - OVERLAP_SECONDS),
                )
                for n in inst.notes
                if n.start >= length - OVERLAP_SECONDS
            ]
            if tail:
                previous[renames.get(_name(inst), _name(inst))] = tail

    for lane in merged.instruments:
        lane.notes.sort(key=lambda note: note.start)
    return merged
