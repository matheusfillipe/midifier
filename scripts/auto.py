"""Two-pass automatic transcription: detect the real instruments, then re-decode.

Left unconstrained, the model hands uncertainty to a plausible-sounding class that then
acts as a dumping ground: on the test track it invented a 3430-note "acoustic guitar"
whose median note was 0.120s while every genuine part sat at 0.230s or longer. Note count
does not catch this (it was the largest track); abnormally short notes do.

Pass 1 decodes freely and measures each class. Pass 2 forbids the classes that look
hallucinated, which both removes them and lets their notes be reassigned to real parts.

    uv run scripts/auto.py song.mp3 -o out/song.mid
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MUSCRIPTOR_VENV = Path(".venv-muscriptor")

# A hallucinated class emits many abnormally short notes. Judged against the median of
# the other melodic classes rather than an absolute, since real note length varies with
# tempo and genre.
SHORT_NOTE_RATIO = 0.6

# Classes sounding for almost none of the track are noise, not parts.
MIN_COVERAGE = 0.01

# Roughly where each class sits when played in its normal register, as a MIDI pitch.
# Used only by --fit-octaves, to pull a part that was decoded an octave out back into
# place. Deliberately coarse: only whole-octave shifts are ever applied.
EXPECTED_MEDIAN_PITCH: dict[str, int] = {
    "acoustic_piano": 60,
    "electric_piano": 60,
    "organ": 60,
    "chromatic_percussion": 72,
    "acoustic_guitar": 52,
    "clean_electric_guitar": 52,
    "distorted_electric_guitar": 52,
    "acoustic_bass": 40,
    "electric_bass": 40,
    "violin": 74,
    "viola": 67,
    "cello": 52,
    "contrabass": 40,
    "orchestral_harp": 60,
    "string_ensemble": 64,
    "synth_strings": 64,
    "synth_pad": 64,
    "synth_lead": 72,
    "trumpet": 70,
    "trombone": 55,
    "tuba": 40,
    "french_horn": 60,
    "brass_section": 60,
    "soprano_and_alto_sax": 68,
    "tenor_sax": 58,
    "baritone_sax": 48,
    "oboe": 74,
    "english_horn": 66,
    "bassoon": 48,
    "clarinet": 62,
    "flutes": 79,
    "voice": 62,
    "timpani": 45,
    "orchestra_hit": 60,
}


@dataclass
class Part:
    instrument: str
    count: int
    median_duration: float
    coverage: float
    median_pitch: float


def run_muscriptor(args: list[str]) -> None:
    command = [str(MUSCRIPTOR_VENV / "bin" / "python"), "-m", "muscriptor", "transcribe", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"muscriptor failed:\n{result.stderr[-2000:]}")


def load_parts(events_path: Path, span: float) -> list[Part]:
    """Pair start/end events from the JSON stream into per-instrument statistics."""
    events = json.loads(events_path.read_text())
    starts = {e["index"]: e for e in events if e["type"] == "start"}

    notes: dict[str, list[tuple[float, float, int]]] = {}
    for event in events:
        if event["type"] != "end":
            continue
        start = starts.get(event["start_event_index"])
        if start is None:
            continue
        notes.setdefault(start["instrument"], []).append((start["start_time"], event["end_time"], start["pitch"]))

    parts: list[Part] = []
    for instrument, rows in notes.items():
        durations = [end - start for start, end, _ in rows if end > start]
        if not durations:
            continue
        covered, reach = 0.0, -1.0
        for start, end, _ in sorted(rows):
            if start > reach:
                covered += end - start
                reach = end
            elif end > reach:
                covered += end - reach
                reach = end
        parts.append(
            Part(
                instrument=instrument,
                count=len(rows),
                median_duration=statistics.median(durations),
                coverage=covered / span if span > 0 else 0.0,
                median_pitch=statistics.median([p for *_, p in rows]),
            )
        )
    return sorted(parts, key=lambda p: -p.count)


def choose_instruments(parts: list[Part]) -> tuple[list[str], list[tuple[str, str]]]:
    """Keep the parts that look like real playing; explain every rejection."""
    melodic = [p for p in parts if p.instrument != "drums"]
    if not melodic:
        return [p.instrument for p in parts], []

    reference = statistics.median([p.median_duration for p in melodic])
    keep: list[str] = []
    dropped: list[tuple[str, str]] = []

    for part in parts:
        if part.instrument == "drums":
            keep.append(part.instrument)
            continue
        if part.median_duration < SHORT_NOTE_RATIO * reference:
            dropped.append(
                (part.instrument, f"notes too short ({part.median_duration:.3f}s vs {reference:.3f}s median)")
            )
            continue
        if part.coverage < MIN_COVERAGE:
            dropped.append((part.instrument, f"sounds for only {100 * part.coverage:.1f}% of the track"))
            continue
        keep.append(part.instrument)

    # Never let the filter empty the result; the loudest part is real by definition.
    if not [k for k in keep if k != "drums"] and melodic:
        best = max(melodic, key=lambda p: p.coverage)
        keep.append(best.instrument)
        dropped = [d for d in dropped if d[0] != best.instrument]
    return keep, dropped


def fit_octaves(midi_path: Path) -> list[str]:
    """Shift whole parts by octaves so each sits in its instrument's normal register."""
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes_moved: list[str] = []
    for instrument in midi.instruments:
        if instrument.is_drum or not instrument.notes:
            continue
        key = instrument.name.replace(" ", "_")
        expected = EXPECTED_MEDIAN_PITCH.get(key)
        if expected is None:
            continue
        median = statistics.median([n.pitch for n in instrument.notes])
        shift = round((expected - median) / 12) * 12
        if shift == 0:
            continue
        pitches = [n.pitch + shift for n in instrument.notes]
        if min(pitches) < 21 or max(pitches) > 108:
            continue
        for note in instrument.notes:
            note.pitch += shift
        notes_moved.append(f"{instrument.name} {shift:+d} semitones (median {median:.0f} -> {median + shift:.0f})")
    if notes_moved:
        midi.write(str(midi_path))
    return notes_moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-m", "--model", default="medium")
    parser.add_argument("-d", "--device", default="auto")
    parser.add_argument("--fit-octaves", action="store_true")
    args = parser.parse_args()

    import librosa

    args.output.parent.mkdir(parents=True, exist_ok=True)
    span = float(librosa.get_duration(path=str(args.audio)))
    common = ["-m", args.model, "-d", args.device]

    print(f"[auto] {args.audio.name}, {span:.0f}s, model={args.model}")

    started = time.monotonic()
    events_path = args.output.with_suffix(".pass1.json")
    run_muscriptor([str(args.audio), "-o", str(events_path), "-f", "json", *common])
    pass1 = time.monotonic() - started

    parts = load_parts(events_path, span)
    print(f"[auto] pass 1 ({pass1:.0f}s) found {len(parts)} instruments:")
    for part in parts:
        print(
            f"[auto]   {part.instrument:<28}{part.count:>6} notes  "
            f"med {part.median_duration:.3f}s  cover {100 * part.coverage:>5.1f}%  pitch {part.median_pitch:>3.0f}"
        )

    keep, dropped = choose_instruments(parts)
    for instrument, reason in dropped:
        print(f"[auto]   DROP {instrument}: {reason}")

    # Pass 2 always runs, because pass 1 only produced events, not a MIDI file. It is
    # constrained only when pass 1 turned up something worth forbidding.
    constraint = [] if not dropped else ["--instruments", ",".join(keep)]
    if not dropped:
        print("[auto] nothing looks hallucinated; decoding unconstrained")
    started = time.monotonic()
    run_muscriptor([str(args.audio), "-o", str(args.output), "-f", "midi", *constraint, *common])
    print(f"[auto] pass 2 ({time.monotonic() - started:.0f}s) kept: {', '.join(keep)}")

    if args.fit_octaves:
        for line in fit_octaves(args.output) or ["nothing to shift"]:
            print(f"[auto]   octave: {line}")

    print(f"[auto] wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
