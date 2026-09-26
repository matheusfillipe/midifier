"""Decide the song's instruments once, then build the song from segments decoded with them.

The decoder picks instruments per segment and is not consistent about it: one song's sung line
came out as voice, clarinet and flutes in three segments, a rhythm guitar as piano in one and
guitar in the next, and a whole segment once collapsed into a single piano lane. So a quick
scout of the whole song decides its instruments, and every segment is then decoded held to
that list. The labels agree across the song, and we build one lane per label.
"""

import collections
import statistics

import pretty_midi

# A part the song really has plays in more than one segment. An instrument the scout gives
# at least this many notes in only one segment is that segment going wrong, and allowing it
# lets the full decode wander onto it.
MIN_NOTES = 20
RECURS_IN = 2

# The decoder transcribes in chunks of this many seconds and names each chunk's parts afresh,
# so one part named two ways shows up as two names that almost never share a chunk. Two
# parts that really play together share a good share of them.
CHUNK_SECONDS = 5.0
TOGETHER_SHARE = 1 / 3

# Above this many notes sounding at once a part plays chords rather than a single line. A sung
# line and a flute line are easy for the decoder to confuse; a line and a strummed guitar are not.
CHORDAL = 2.5

# A recurring lead line in a song is far more often sung than blown, and the decoder takes one
# for the other, so when two single lines turn out to be one part it keeps this name.
VOICE = "voice"


def _family(program: int) -> int:
    """General MIDI numbers instruments in groups of eight."""
    return program // 8


def _polyphony(notes: list[pretty_midi.Note]) -> float:
    return statistics.mean(sum(1 for other in notes if other.start <= note.start < other.end) for note in notes)


def choose(
    scout: pretty_midi.PrettyMIDI, offsets: list[float], length: float
) -> tuple[list[str], list[tuple[str, str]]]:
    """The instruments to hold the full decode to, and each name folded into another."""
    notes: dict[str, list[pretty_midi.Note]] = collections.defaultdict(list)
    programs: dict[str, int] = {}
    drums: set[str] = set()
    for lane in scout.instruments:
        notes[lane.name].extend(lane.notes)
        programs[lane.name] = lane.program
        if lane.is_drum:
            drums.add(lane.name)

    def recurs(name: str) -> bool:
        segments = sum(
            1 for offset in offsets if sum(1 for n in notes[name] if offset <= n.start < offset + length) >= MIN_NOTES
        )
        return segments >= min(RECURS_IN, len(offsets))

    chunks = {
        name: collections.Counter(int(note.start // CHUNK_SECONDS) for note in played) for name, played in notes.items()
    }

    def together(first: str, second: str) -> bool:
        heard_first = {chunk for chunk, count in chunks[first].items() if count >= 2}
        heard_second = {chunk for chunk, count in chunks[second].items() if count >= 2}
        rarer = min(len(heard_first), len(heard_second))
        return rarer > 0 and len(heard_first & heard_second) / rarer >= TOGETHER_SHARE

    candidates = sorted((name for name in notes if notes[name] and recurs(name)), key=lambda n: -len(notes[n]))
    line = {name: name not in drums and _polyphony(notes[name]) < CHORDAL for name in candidates}

    def one_part(first: str, second: str) -> bool:
        if first in drums or second in drums or together(first, second):
            return False
        return _family(programs[first]) == _family(programs[second]) or (line[first] and line[second])

    kept: list[str] = []
    folded: list[tuple[str, str]] = []
    for name in candidates:
        rival = next((other for other in kept if one_part(other, name)), None)
        if rival is None:
            kept.append(name)
        elif name == VOICE:
            kept[kept.index(rival)] = name
            folded = [(dropped, name if into == rival else into) for dropped, into in folded]
            folded.append((rival, name))
        else:
            folded.append((name, rival))
    return kept, folded


def instrument_list(names: list[str]) -> str:
    """The decoder's `--instruments` argument: it names instruments as it writes them, with
    underscores for spaces."""
    return ",".join(name.replace(" ", "_") for name in names)


def assemble(decoded: list[tuple[float, pretty_midi.PrettyMIDI]]) -> pretty_midi.PrettyMIDI:
    """One lane per instrument, each segment's notes moved to where they fall in the song.

    Both neighbours transcribe the overlap, so only one of them may keep it. The arriving
    segment wins where it has the instrument, because a decode degenerates as it runs and is
    at its freshest where its neighbour is at its worst. Where it does not have the instrument,
    the outgoing segment keeps playing: a segment that lost a part has nothing fresher to put
    there, and dropping the notes anyway is how a drum kit falls silent at a seam.
    """
    lanes: dict[tuple[str, bool], pretty_midi.Instrument] = {}
    present: dict[tuple[str, bool], set[int]] = collections.defaultdict(set)
    for index, (_, midi) in enumerate(decoded):
        for lane in midi.instruments:
            if lane.notes:
                present[(lane.name, lane.is_drum)].add(index)

    for index, (offset, midi) in enumerate(decoded):
        seam = decoded[index + 1][0] if index + 1 < len(decoded) else None
        for lane in midi.instruments:
            key = (lane.name, lane.is_drum)
            if not lane.notes:
                continue
            cutoff = seam if index + 1 in present[key] else None
            song_lane = lanes.setdefault(
                key, pretty_midi.Instrument(program=lane.program, is_drum=lane.is_drum, name=lane.name)
            )
            song_lane.notes.extend(
                pretty_midi.Note(
                    velocity=note.velocity, pitch=note.pitch, start=note.start + offset, end=note.end + offset
                )
                for note in lane.notes
                if cutoff is None or note.start + offset < cutoff
            )

    song = pretty_midi.PrettyMIDI()
    for lane in lanes.values():
        lane.notes.sort(key=lambda note: note.start)
        song.instruments.append(lane)
    return song
