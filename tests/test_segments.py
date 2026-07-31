"""Stitching segments back together, where lane identity is the thing that can go wrong."""

import pretty_midi

from midifier.midi import segments


def _instrument(
    name: str, notes: list[tuple[float, int]], program: int = 0, is_drum: bool = False
) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    inst.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.2) for start, pitch in notes]
    return inst


def _segment(*instruments: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(instruments)
    return midi


class TestPlan:
    def test_a_short_file_is_one_segment(self) -> None:
        assert segments.plan(40.0, 60.0) == [0.0]

    def test_segments_overlap_their_neighbour(self) -> None:
        offsets = segments.plan(200.0, 60.0)
        assert offsets[1] - offsets[0] == 60.0 - segments.OVERLAP_SECONDS

    def test_the_last_segment_reaches_the_end(self) -> None:
        offsets = segments.plan(200.0, 60.0)
        assert offsets[-1] + 60.0 >= 200.0


class TestStitch:
    def test_notes_land_at_their_place_in_the_song(self) -> None:
        first = _segment(_instrument("bass", [(1.0, 40)]))
        second = _segment(_instrument("bass", [(10.0, 40)]))
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        starts = sorted(note.start for inst in merged.instruments for note in inst.notes)
        assert starts == [1.0, 65.0]

    def test_the_overlap_is_not_transcribed_twice(self) -> None:
        """Both segments cover the seam; taking both would double every note in it."""
        first = _segment(_instrument("bass", [(56.0, 40)]))
        second = _segment(_instrument("bass", [(1.0, 40)]))
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        assert sum(len(inst.notes) for inst in merged.instruments) == 1

    def test_a_relabelled_part_keeps_the_earlier_name(self) -> None:
        """The whole band appearing to change instrument at a seam is what this prevents."""
        shared = [(56.0, 40), (56.5, 43), (57.0, 45)]
        first = _segment(_instrument("distorted electric guitar", shared))
        second = _segment(_instrument("synth lead", [(1.0, 40), (1.5, 43), (2.0, 45), (20.0, 47)]))
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        assert [inst.name for inst in merged.instruments] == ["distorted electric guitar"]

    def test_an_unrelated_part_is_not_given_a_name_already_in_use(self) -> None:
        """Inheriting a name is a permutation: two different parts cannot share one lane."""
        shared = [(56.0, 40), (56.5, 43), (57.0, 45)]
        first = _segment(_instrument("distorted electric guitar", shared))
        second = _segment(
            _instrument("synth lead", [(1.0, 40), (1.5, 43), (2.0, 45)]),
            _instrument("distorted electric guitar", [(30.0, 70), (31.0, 72)]),
        )
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        names = sorted(inst.name for inst in merged.instruments)
        assert len(names) == 2
        assert len(set(names)) == 2

    def test_a_part_that_shares_nothing_keeps_its_own_name(self) -> None:
        first = _segment(_instrument("bass", [(56.0, 40)]))
        second = _segment(_instrument("voice", [(1.0, 70), (2.0, 72), (3.0, 74)]))
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        assert sorted(inst.name for inst in merged.instruments) == ["bass", "voice"]

    def test_drums_are_never_matched_onto_a_melodic_lane(self) -> None:
        shared = [(56.0, 40), (56.5, 40), (57.0, 40)]
        first = _segment(_instrument("drums", shared, is_drum=True))
        second = _segment(_instrument("electric bass", [(1.0, 40), (1.5, 40), (2.0, 40)]))
        merged = segments.stitch([(0.0, first), (55.0, second)], 60.0)
        assert sorted(inst.name for inst in merged.instruments) == ["drums", "electric bass"]
