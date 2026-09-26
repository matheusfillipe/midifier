"""Choosing the song's instruments, and building the song from segments decoded with them."""

import pretty_midi

from midifier.midi.parts import assemble
from midifier.midi.parts import choose
from midifier.midi.parts import instrument_list

SEAM = 45.0


def _lane(
    name: str, count: int, program: int = 0, *, is_drum: bool = False, start: float = 0.0
) -> pretty_midi.Instrument:
    lane = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    lane.notes = [
        pretty_midi.Note(velocity=100, pitch=40 + index % 12, start=start + index * 0.5, end=start + index * 0.5 + 0.2)
        for index in range(count)
    ]
    return lane


def _segment(*lanes: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(lanes)
    return midi


CLEAN = 27
DISTORTED = 30
VOICE = 52
STRINGS = 48


# A four-segment song, as the scout sees it: one decode of the whole thing.
OFFSETS = [0.0, 45.0, 90.0, 135.0]
LENGTH = 60.0


def _chords(name: str, count: int, program: int, *, start: float = 0.0) -> pretty_midi.Instrument:
    """A strummed part: three notes struck together, held past the next strum."""
    lane = pretty_midi.Instrument(program=program, name=name)
    lane.notes = [
        pretty_midi.Note(velocity=100, pitch=48 + offset, start=start + index * 0.5, end=start + index * 0.5 + 1.2)
        for index in range(count)
        for offset in (0, 4, 7)
    ]
    return lane


def _scout(*lanes: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    return _segment(*lanes)


def _taking_turns(name: str, other: str, program: int, other_program: int) -> list[pretty_midi.Instrument]:
    """Two names that alternate every 20 seconds, never sharing a chunk, across the song."""
    first = pretty_midi.Instrument(program=program, name=name)
    second = pretty_midi.Instrument(program=other_program, name=other)
    for turn in range(9):
        lane = first if turn % 2 == 0 else second
        lane.notes += _lane(lane.name, 40, lane.program, start=turn * 20.0).notes
    return [first, second]


class TestChoose:
    def test_one_guitar_named_two_ways_becomes_one_instrument(self) -> None:
        """A guitar flipping between clean and distorted from one chunk to the next is one part."""
        kept, folded = choose(
            _scout(*_taking_turns("distorted electric guitar", "clean electric guitar", DISTORTED, CLEAN)),
            OFFSETS,
            LENGTH,
        )
        assert kept == ["distorted electric guitar"]
        assert folded == [("clean electric guitar", "distorted electric guitar")]

    def test_two_guitars_playing_together_stay_two(self) -> None:
        scout = _scout(_lane("clean electric guitar", 360, CLEAN), _lane("distorted electric guitar", 300, DISTORTED))
        kept, _ = choose(scout, OFFSETS, LENGTH)
        assert sorted(kept) == ["clean electric guitar", "distorted electric guitar"]

    def test_an_instrument_the_scout_barely_used_is_left_out(self) -> None:
        kept, _ = choose(_scout(_lane("voice", 360, VOICE), _lane("string ensemble", 3, STRINGS)), OFFSETS, LENGTH)
        assert kept == ["voice"]

    def test_an_instrument_heard_in_one_segment_only_is_left_out(self) -> None:
        scout = _scout(_lane("voice", 360, VOICE), _lane("brass section", 40, 61, start=62.0))
        kept, _ = choose(scout, OFFSETS, LENGTH)
        assert kept == ["voice"]

    def test_a_sung_line_the_decoder_also_calls_flutes_is_the_voice(self) -> None:
        """Two single lines taking turns are one part, and the sung name wins over the blown one."""
        lanes = _taking_turns("flutes", "voice", 73, VOICE)
        kept, folded = choose(_scout(*lanes), OFFSETS, LENGTH)
        assert kept == ["voice"]
        assert folded == [("flutes", "voice")]

    def test_a_guitar_playing_between_the_verses_is_not_the_voice(self) -> None:
        """Taking turns is not enough across families: a strummed part and a sung line are two."""
        voice = pretty_midi.Instrument(program=VOICE, name="voice")
        guitar = _chords("clean electric guitar", 0, CLEAN)
        for turn in range(9):
            if turn % 2 == 0:
                voice.notes += _lane("voice", 40, VOICE, start=turn * 20.0).notes
            else:
                guitar.notes += _chords("clean electric guitar", 40, CLEAN, start=turn * 20.0).notes
        kept, _ = choose(_scout(voice, guitar), OFFSETS, LENGTH)
        assert sorted(kept) == ["clean electric guitar", "voice"]

    def test_drums_are_always_their_own_instrument(self) -> None:
        scout = _scout(_lane("drums", 360, is_drum=True), _lane("acoustic piano", 360, 0))
        kept, _ = choose(scout, OFFSETS, LENGTH)
        assert sorted(kept) == ["acoustic piano", "drums"]

    def test_the_list_uses_the_decoder_s_own_spelling(self) -> None:
        assert instrument_list(["distorted electric guitar", "voice"]) == "distorted_electric_guitar,voice"


class TestAssemble:
    def test_one_lane_per_instrument_placed_at_its_time_in_the_song(self) -> None:
        first = _segment(_lane("voice", 4, VOICE, start=1.0))
        second = _segment(_lane("voice", 4, VOICE, start=20.0))
        song = assemble([(0.0, first), (SEAM, second)])
        assert [lane.name for lane in song.instruments] == ["voice"]
        assert song.instruments[0].notes[-1].start == SEAM + 21.5

    def test_the_arriving_segment_takes_over_the_overlap(self) -> None:
        """Both segments transcribe the overlap; keeping both would double every note in it."""
        first = _segment(_lane("voice", 2, VOICE, start=SEAM + 1.0))
        second = _segment(_lane("voice", 2, VOICE, start=1.0))
        song = assemble([(0.0, first), (SEAM, second)])
        assert len(song.instruments[0].notes) == 2

    def test_a_part_the_arriving_segment_lost_keeps_playing(self) -> None:
        """A segment that lost the drums has nothing fresher to put in their place."""
        first = _segment(_lane("drums", 4, is_drum=True, start=SEAM + 1.0), _lane("voice", 2, VOICE))
        second = _segment(_lane("voice", 2, VOICE, start=1.0))
        song = assemble([(0.0, first), (SEAM, second)])
        drums = next(lane for lane in song.instruments if lane.is_drum)
        assert min(note.start for note in drums.notes) > SEAM

    def test_nothing_decoded_is_an_empty_song(self) -> None:
        assert assemble([(0.0, pretty_midi.PrettyMIDI())]).instruments == []
