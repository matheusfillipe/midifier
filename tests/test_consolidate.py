"""Folding lanes that are one part the decoder renamed partway through."""

import pretty_midi

from midifier.midi.consolidate import consolidate


def _instrument(
    name: str, notes: list[tuple[float, int]], is_drum: bool = False, program: int = 0
) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    inst.notes = [pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.2) for start, pitch in notes]
    return inst


def _midi(*instruments: pretty_midi.Instrument) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend(instruments)
    return midi


def _line(start: float, count: int, low: int = 40) -> list[tuple[float, int]]:
    return [(start + index * 0.5, low + index % 12) for index in range(count)]


class TestFolding:
    def test_a_part_relabelled_partway_becomes_one_lane(self) -> None:
        midi = _midi(
            _instrument("distorted electric guitar", _line(0.0, 40)),
            _instrument("synth lead", _line(40.0, 40)),
        )
        folded = consolidate(midi)
        assert folded == [("synth lead", "distorted electric guitar")]
        assert len(midi.instruments) == 1
        assert len(midi.instruments[0].notes) == 80

    def test_the_earlier_lane_is_the_one_kept(self) -> None:
        midi = _midi(
            _instrument("synth lead", _line(40.0, 40)),
            _instrument("distorted electric guitar", _line(0.0, 40)),
        )
        consolidate(midi)
        assert midi.instruments[0].name == "distorted electric guitar"

    def test_parts_that_play_together_are_left_alone(self) -> None:
        """Two instruments in the same register are not one part just because they overlap."""
        midi = _midi(
            _instrument("distorted electric guitar", _line(0.0, 40)),
            _instrument("clean electric guitar", _line(0.05, 40)),
        )
        assert consolidate(midi) == []
        assert len(midi.instruments) == 2

    def test_parts_in_different_registers_are_left_alone(self) -> None:
        midi = _midi(
            _instrument("electric bass", _line(0.0, 40, low=24)),
            _instrument("voice", _line(40.0, 40, low=64)),
        )
        assert consolidate(midi) == []

    def test_drums_are_never_folded(self) -> None:
        midi = _midi(
            _instrument("drums", _line(0.0, 40), is_drum=True),
            _instrument("distorted electric guitar", _line(40.0, 40)),
        )
        assert consolidate(midi) == []

    def test_a_bass_is_not_folded_into_a_piano(self) -> None:
        """Two instruments taking turns is what a song sounds like, not one part renamed."""
        midi = _midi(
            _instrument("electric piano", _line(0.0, 40, low=48), program=2),
            _instrument("electric bass", _line(40.0, 40, low=48), program=33),
        )
        assert consolidate(midi) == []

    def test_a_narrow_part_inside_a_wide_one_is_left_alone(self) -> None:
        """Register share is measured against the wider lane, or a broad lane swallows anything
        that happens to sit inside its range -- and each fold widens it further."""
        wide = _instrument("electric piano", _line(0.0, 40, low=36), program=2)
        wide.notes.append(pretty_midi.Note(velocity=100, pitch=100, start=1.0, end=1.2))
        midi = _midi(wide, _instrument("acoustic piano", _line(40.0, 40, low=40), program=0))
        assert consolidate(midi) == []

    def test_three_fragments_of_one_part_all_collapse(self) -> None:
        midi = _midi(
            _instrument("distorted electric guitar", _line(0.0, 30)),
            _instrument("synth lead", _line(30.0, 30)),
            _instrument("clean electric guitar", _line(60.0, 30)),
        )
        consolidate(midi)
        assert len(midi.instruments) == 1
        assert len(midi.instruments[0].notes) == 90
