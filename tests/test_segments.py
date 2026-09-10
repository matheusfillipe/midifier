"""Where a long decode is cut."""

from midifier.midi import segments

SEGMENT = 60.0

# Where the next segment takes over. Derived, because it moves with the overlap.
SEAM = SEGMENT - segments.OVERLAP_SECONDS


class TestPlan:
    def test_a_short_file_is_one_segment(self) -> None:
        assert segments.plan(40.0, SEGMENT) == [0.0]

    def test_segments_overlap_their_neighbour(self) -> None:
        offsets = segments.plan(200.0, SEGMENT)
        assert offsets[1] - offsets[0] == SEAM

    def test_the_last_segment_reaches_the_end(self) -> None:
        offsets = segments.plan(200.0, SEGMENT)
        assert offsets[-1] + SEGMENT >= 200.0

    def test_every_second_of_the_song_is_covered(self) -> None:
        """A gap between segments is audio nothing transcribes, which is silence in the result."""
        duration = 200.0
        offsets = segments.plan(duration, SEGMENT)
        second = 0.0
        while second < duration:
            assert any(offset <= second < offset + SEGMENT for offset in offsets), f"{second}s falls in no segment"
            second += 1.0
