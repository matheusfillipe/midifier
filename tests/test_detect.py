"""Hallucinated-instrument detection, driven by the shape the real failure had."""

from __future__ import annotations

import pytest

from midifier.midi.detect import Part
from midifier.midi.detect import choose
from midifier.midi.detect import summarise


def _rows(instrument: str, count: int, duration: float, step: float = 0.25) -> list[tuple[str, float, float, int]]:
    return [(instrument, index * step, index * step + duration, 60) for index in range(count)]


class TestSummarise:
    def test_measures_each_instrument_separately(self) -> None:
        parts = {part.instrument: part for part in summarise(_rows("bass", 10, 0.23) + _rows("brass", 4, 0.34), 10.0)}
        assert parts["bass"].note_count == 10
        assert parts["bass"].median_duration == pytest.approx(0.23)
        assert parts["brass"].median_duration == pytest.approx(0.34)

    def test_counts_overlapping_notes_once_for_coverage(self) -> None:
        rows = [("pad", 0.0, 5.0, 60), ("pad", 1.0, 4.0, 64)]
        (part,) = summarise(rows, span=10.0)
        assert part.coverage == 0.5

    def test_ignores_zero_length_notes(self) -> None:
        assert summarise([("x", 1.0, 1.0, 60)], span=10.0) == []


class TestChoose:
    def test_drops_the_class_with_abnormally_short_notes(self) -> None:
        """The real failure: the invented track was the biggest, so count cannot find it."""
        parts = [
            Part("acoustic_guitar", note_count=3430, median_duration=0.120, coverage=0.68, median_pitch=60),
            Part("electric_bass", note_count=771, median_duration=0.230, coverage=0.97, median_pitch=38),
            Part("distorted_electric_guitar", note_count=751, median_duration=0.230, coverage=0.74, median_pitch=62),
            Part("brass_section", note_count=199, median_duration=0.340, coverage=0.11, median_pitch=65),
        ]
        verdict = choose(parts)
        assert "acoustic_guitar" not in verdict.keep
        assert verdict.dropped[0][0] == "acoustic_guitar"
        assert "electric_bass" in verdict.keep

    def test_drops_a_class_that_barely_sounds(self) -> None:
        parts = [
            Part("bass", note_count=500, median_duration=0.23, coverage=0.9, median_pitch=40),
            Part("triangle", note_count=3, median_duration=0.25, coverage=0.001, median_pitch=90),
        ]
        assert "triangle" not in choose(parts).keep

    def test_drums_are_never_judged_on_note_length(self) -> None:
        """Drum hits are momentary by nature and would always look hallucinated."""
        parts = [
            Part("drums", note_count=2070, median_duration=0.010, coverage=0.07, median_pitch=42),
            Part("bass", note_count=771, median_duration=0.230, coverage=0.97, median_pitch=38),
        ]
        assert "drums" in choose(parts).keep

    def test_never_returns_an_empty_melodic_set(self) -> None:
        parts = [Part("solo", note_count=10, median_duration=0.23, coverage=0.0001, median_pitch=60)]
        assert choose(parts).keep == ["solo"]

    def test_keeps_everything_when_nothing_looks_wrong(self) -> None:
        parts = [
            Part("bass", note_count=700, median_duration=0.23, coverage=0.9, median_pitch=38),
            Part("guitar", note_count=600, median_duration=0.23, coverage=0.7, median_pitch=62),
        ]
        verdict = choose(parts)
        assert verdict.dropped == []
        assert len(verdict.keep) == 2
