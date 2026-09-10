"""Deciding the song's parts from every segment at once."""

import pretty_midi

from midifier.midi.parts import assemble

SEGMENT = 60.0
SEAM = 45.0


def _lane(name: str, notes: list[tuple[float, int]], program: int = 0, is_drum: bool = False) -> pretty_midi.Instrument:
    lane = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    lane.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.2) for start, pitch in notes]
    return lane


def _segment(*lanes: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(lanes)
    return midi


# A scale's worth of pitches, so a part covers a register rather than sitting on one note.
SCALE = (0, 2, 4, 5, 7, 9, 11, 12)


def _line(start: float, count: int, low: int, step: float = 0.5) -> list[tuple[float, int]]:
    """A single-note part: one pitch at a time, over an octave."""
    return [(start + index * step, low + SCALE[index % len(SCALE)]) for index in range(count)]


def _chords(start: float, count: int, low: int, step: float = 0.5) -> list[tuple[float, int]]:
    """A chordal part: three notes struck together."""
    return [
        (start + index * step, low + SCALE[index % len(SCALE)] + offset)
        for index in range(count)
        for offset in (0, 4, 7)
    ]


def _named(song: pretty_midi.PrettyMIDI) -> list[str]:
    return sorted(lane.name for lane in song.instruments if lane.notes)


class TestParts:
    def test_a_lane_holding_the_whole_band_is_split_into_its_parts(self) -> None:
        """The decoder sometimes emits one lane spanning the band; its notes still belong apart."""
        good = _segment(
            _lane("electric bass", _line(0.0, 40, 32), program=33),
            _lane("voice", _line(0.0, 40, 64), program=52),
            _lane("clean electric guitar", _chords(0.0, 30, 48), program=26),
        )
        collapsed = _segment(
            _lane(
                "acoustic piano",
                _line(0.0, 20, 32) + _line(0.0, 20, 64) + _chords(0.0, 20, 48),
                program=0,
            )
        )
        song, _ = assemble([(0.0, good), (SEAM, collapsed)])
        assert _named(song) == ["clean electric guitar", "electric bass", "voice"]
        bass = next(lane for lane in song.instruments if lane.name == "electric bass")
        assert max(note.pitch for note in bass.notes) < 48

    def test_one_part_under_two_names_becomes_one_lane(self) -> None:
        first = _segment(_lane("voice", _line(0.0, 40, 64), program=52))
        second = _segment(_lane("flutes", _line(0.0, 40, 64), program=72))
        song, folded = assemble([(0.0, first), (SEAM, second)])
        assert _named(song) == ["voice"]
        assert ("flutes", "voice") in folded

    def test_parts_that_play_together_stay_apart(self) -> None:
        """Two lines an octave apart sounding at once are two parts, whatever they are called."""
        segment = _segment(
            _lane("electric bass", _line(0.0, 40, 30), program=33),
            _lane("voice", _line(0.0, 40, 70), program=52),
        )
        song, _ = assemble([(0.0, segment)])
        assert _named(song) == ["electric bass", "voice"]

    def test_a_part_the_arriving_segment_lost_keeps_playing(self) -> None:
        """A segment that lost the drums has nothing fresher to put in their place."""
        # Both lanes play into the overlap, which is where the arriving segment would take over.
        first = _segment(
            _lane("voice", _line(0.0, 110, 64), program=52),
            _lane("drums", _line(0.0, 110, 38), is_drum=True),
        )
        second = _segment(_lane("voice", _line(0.0, 110, 64), program=52))
        song, _ = assemble([(0.0, first), (SEAM, second)])
        drums = next(lane for lane in song.instruments if lane.is_drum)
        assert max(note.start for note in drums.notes) > SEAM

    def test_the_name_the_decoder_repeats_wins_over_the_one_it_guessed_once(self) -> None:
        first = _segment(_lane("voice", _line(0.0, 30, 64), program=52))
        second = _segment(_lane("clarinet", _line(0.0, 60, 64), program=71))
        third = _segment(_lane("voice", _line(0.0, 30, 64), program=52))
        song, _ = assemble([(0.0, first), (SEAM, second), (2 * SEAM, third)])
        assert _named(song) == ["voice"]

    def test_a_lane_holding_the_band_does_not_name_a_part(self) -> None:
        """Its label describes whichever part the decoder noticed, so it names nothing."""
        good = _segment(_lane("voice", _line(0.0, 30, 64), program=52))
        collapsed = _segment(
            _lane("acoustic piano", _line(0.0, 60, 64) + _line(0.0, 30, 30) + _chords(0.0, 30, 48), program=0)
        )
        song, _ = assemble([(0.0, good), (SEAM, collapsed)])
        assert "voice" in _named(song)
        assert "acoustic piano" not in _named(song)

    def test_one_instrument_s_chord_is_not_shared_out_between_parts(self) -> None:
        """Taking a chord's lowest note for the bass and its highest for the singer leaves three
        parts each holding a third of it, and every part sounding all the time."""
        good = _segment(
            _lane("electric bass", _line(0.0, 40, 30), program=33),
            _lane("voice", _line(0.0, 40, 70), program=52),
            _lane("clean electric guitar", _chords(0.0, 30, 50), program=26),
        )
        held = [(0.5 * index, 50 + offset) for index in range(40) for offset in (0, 4, 7, 12)]
        collapsed = _segment(_lane("acoustic piano", held + _line(0.0, 20, 28), program=0))
        song, _ = assemble([(0.0, good), (SEAM, collapsed)])

        voice = next(lane for lane in song.instruments if lane.name == "voice")
        # The chord sits inside one instrument's reach, so none of it belongs to the singer.
        assert [note for note in voice.notes if note.start >= SEAM] == []

    def test_a_chord_reaching_across_the_band_is_shared_out(self) -> None:
        good = _segment(
            _lane("electric bass", _line(0.0, 40, 30), program=33),
            _lane("voice", _line(0.0, 40, 76), program=52),
            _lane("clean electric guitar", _chords(0.0, 30, 50), program=26),
        )
        spread = [(0.5 * index, pitch) for index in range(40) for pitch in (30, 50, 54, 57, 78)]
        collapsed = _segment(_lane("acoustic piano", spread, program=0))
        song, _ = assemble([(0.0, good), (SEAM, collapsed)])

        bass = next(lane for lane in song.instruments if lane.name == "electric bass")
        voice = next(lane for lane in song.instruments if lane.name == "voice")
        assert [note for note in bass.notes if note.start >= SEAM]
        assert [note for note in voice.notes if note.start >= SEAM]

    def test_drums_are_never_folded_into_an_instrument(self) -> None:
        segment = _segment(
            _lane("drums", _line(0.0, 40, 38), is_drum=True),
            _lane("electric bass", _line(0.0, 40, 36), program=33),
        )
        song, _ = assemble([(0.0, segment)])
        assert sorted(lane.is_drum for lane in song.instruments) == [False, True]

    def test_a_song_the_decoder_never_separates_still_comes_out(self) -> None:
        """A solo piano reaches past any one part's register, so every lane looks like the band.

        Nothing is left to learn the song's parts from, and nothing has the standing to name
        one. The notes are still the recording, and dropping them fails the job outright.
        """
        both_hands = _line(0.0, 60, 36) + _line(0.0, 60, 72)
        segments = [_segment(_lane("acoustic piano", both_hands, program=0)) for _ in range(3)]
        song, _ = assemble([(index * SEAM, part) for index, part in enumerate(segments)])
        assert song.instruments
        assert sum(len(lane.notes) for lane in song.instruments) > 0

    def test_nothing_decoded_is_not_an_error(self) -> None:
        song, folded = assemble([(0.0, pretty_midi.PrettyMIDI())])
        assert song.instruments == []
        assert folded == []
