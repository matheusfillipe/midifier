"""The cleanup rules are pure functions over notes, so they are tested exactly."""

import pretty_midi

from midifier.midi.cleanup import clean
from midifier.midi.cleanup import merge_held
from midifier.midi.cleanup import plays_a_pulse
from midifier.midi.cleanup import trim_loops
from midifier.midi.cleanup import trim_overrun
from midifier.midi.cleanup import trim_stuck_chord

from .conftest import note


def _midi(notes: list[pretty_midi.Note], *, is_drum: bool = False) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, is_drum=is_drum, name="test")
    instrument.notes.extend(notes)
    midi.instruments.append(instrument)
    return midi


class TestTrimOverrun:
    def test_drops_notes_starting_after_the_audio(self) -> None:
        midi = _midi([note(60, 1.0, 1.5), note(60, 30.0, 30.5)])
        assert trim_overrun(midi, duration=10.0) == 1
        assert len(midi.instruments[0].notes) == 1

    def test_clamps_a_note_still_sounding_at_the_end(self) -> None:
        midi = _midi([note(60, 9.0, 30.0)])
        trim_overrun(midi, duration=10.0)
        assert midi.instruments[0].notes[0].end == 10.5  # duration + END_TOLERANCE

    def test_keeps_a_final_chord_ringing_within_tolerance(self) -> None:
        midi = _midi([note(60, 10.2, 10.4)])
        assert trim_overrun(midi, duration=10.0) == 0


class TestMergeHeld:
    def test_joins_touching_notes_of_the_same_pitch(self) -> None:
        midi = _midi([note(60, 0.0, 0.23), note(60, 0.23, 0.46), note(60, 0.46, 0.69)])
        assert merge_held(midi, max_gap=0.05, max_length=1.0) == 2
        (merged,) = midi.instruments[0].notes
        assert merged.start == 0.0
        assert merged.end == 0.69

    def test_leaves_a_real_gap_alone(self) -> None:
        midi = _midi([note(60, 0.0, 0.2), note(60, 1.0, 1.2)])
        assert merge_held(midi, max_gap=0.05, max_length=1.0) == 0
        assert len(midi.instruments[0].notes) == 2

    def test_refuses_to_invent_a_long_sustain(self) -> None:
        """The cap is what stops a whole phrase fusing into one implausible note."""
        notes = [note(60, index * 0.2, index * 0.2 + 0.2) for index in range(10)]
        merge_held(_midi(notes), max_gap=0.05, max_length=0.5)
        rebuilt = _midi([note(60, index * 0.2, index * 0.2 + 0.2) for index in range(10)])
        merge_held(rebuilt, max_gap=0.05, max_length=0.5)
        assert all(n.end - n.start <= 0.5 + 1e-9 for n in rebuilt.instruments[0].notes)

    def test_different_pitches_never_merge(self) -> None:
        midi = _midi([note(60, 0.0, 0.2), note(62, 0.2, 0.4)])
        assert merge_held(midi) == 0

    def test_percussion_is_left_alone(self) -> None:
        midi = _midi([note(36, 0.0, 0.05), note(36, 0.05, 0.1)], is_drum=True)
        assert merge_held(midi) == 0

    def test_a_steady_pulse_is_left_alone(self) -> None:
        """Fusing a driving bass into one drone is heard as the part having disappeared."""
        notes = [note(40, index * 0.2, index * 0.2 + 0.19) for index in range(40)]
        assert merge_held(_midi(notes)) == 0


class TestPulseDetection:
    def test_evenly_spaced_playing_is_a_pulse(self) -> None:
        notes = [note(40, index * 0.2, index * 0.2 + 0.19) for index in range(40)]
        assert plays_a_pulse(_midi(notes).instruments[0])

    def test_free_playing_is_not(self) -> None:
        starts = [0.0, 0.31, 0.9, 1.05, 2.4, 2.5, 4.2, 4.9, 5.05, 7.7, 9.1, 9.15]
        assert not plays_a_pulse(_midi([note(60, s, s + 0.1) for s in starts]).instruments[0])

    def test_too_few_notes_to_tell(self) -> None:
        notes = [note(40, index * 0.2, index * 0.2 + 0.19) for index in range(4)]
        assert not plays_a_pulse(_midi(notes).instruments[0])


class TestTrimLoops:
    def test_cuts_a_stuck_single_pitch(self) -> None:
        notes = [note(60, index * 0.25, index * 0.25 + 0.2) for index in range(40)]
        assert trim_loops(_midi(notes), duration=10.0, max_cycles=8) > 0

    def test_cuts_a_repeating_two_note_figure(self) -> None:
        """A same-pitch counter never fires on A-B-A-B; periodicity does."""
        notes = []
        for index in range(60):
            pitch = 60 if index % 2 == 0 else 64
            notes.append(note(pitch, index * 0.25, index * 0.25 + 0.2))
        assert trim_loops(_midi(notes), duration=10.0, max_cycles=8) > 0

    def test_leaves_ordinary_playing_alone(self) -> None:
        notes = [note(60 + (index % 7), index * 0.25, index * 0.25 + 0.2) for index in range(30)]
        assert trim_loops(_midi(notes), duration=10.0, max_cycles=8) == 0

    def test_a_repeated_section_mid_song_is_music(self) -> None:
        """Only a decode running out of audio degenerates; a repetitive verse is playing."""
        notes = [note(60, index * 0.25, index * 0.25 + 0.2) for index in range(40)]
        assert trim_loops(_midi(notes), duration=300.0, max_cycles=8) == 0


class TestTrimStuckChord:
    def _locked(self, width: int, cycles: int, start: float = 40.0) -> list[pretty_midi.Note]:
        """A wide chord alternating with a single note, the shape real decodes produce."""
        notes: list[pretty_midi.Note] = []
        for cycle in range(cycles):
            at = start + cycle * 0.44
            notes.extend(note(57 + step * 4, at, at + 0.4) for step in range(width))
            notes.append(note(52, at + 0.22, at + 0.42))
        return notes

    def test_cuts_a_chord_repeated_to_the_end(self) -> None:
        midi = _midi(self._locked(width=13, cycles=9))
        assert trim_stuck_chord(midi, duration=60.0) > 0

    def test_takes_the_whole_run(self) -> None:
        """Leaving one copy strands a chord in the silence the decoder stopped playing in."""
        midi = _midi(self._locked(width=13, cycles=9))
        trim_stuck_chord(midi, duration=60.0)
        assert midi.instruments[0].notes == []

    def test_playing_before_the_run_survives(self) -> None:
        played = [note(60 + index % 5, 30.0 + index * 0.5, 30.4 + index * 0.5) for index in range(12)]
        midi = _midi(played + self._locked(width=13, cycles=9))
        trim_stuck_chord(midi, duration=60.0)
        assert len(midi.instruments[0].notes) == len(played)

    def test_a_narrow_figure_is_playing_not_locking(self) -> None:
        """Two notes repeating is a drum figure; width is what tells them apart."""
        midi = _midi(self._locked(width=2, cycles=9))
        assert trim_stuck_chord(midi, duration=60.0) == 0

    def test_a_few_repeats_are_left_alone(self) -> None:
        midi = _midi(self._locked(width=13, cycles=3))
        assert trim_stuck_chord(midi, duration=60.0) == 0

    def test_the_same_chord_mid_song_is_music(self) -> None:
        midi = _midi(self._locked(width=13, cycles=9, start=40.0))
        assert trim_stuck_chord(midi, duration=300.0) == 0

    def test_trim_loops_cannot_see_it(self) -> None:
        """Why this rule exists: a flat pitch sequence scores nothing on a repeated chord."""
        midi = _midi(self._locked(width=13, cycles=9))
        assert trim_loops(midi, duration=60.0, max_cycles=8) == 0


class TestClean:
    def test_reports_what_it_changed(self) -> None:
        notes = [note(60, 0.0, 0.23), note(60, 0.23, 0.46), note(60, 99.0, 99.5)]
        report = clean(_midi(notes), duration=10.0)
        assert report.trimmed_past_end == 1
        assert report.merged_rearticulations == 1
        assert report.notes_after < report.notes_before

    def test_is_safe_on_an_empty_file(self) -> None:
        report = clean(pretty_midi.PrettyMIDI(), duration=10.0)
        assert report.notes_before == report.notes_after == 0
