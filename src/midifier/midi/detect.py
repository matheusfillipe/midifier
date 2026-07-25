"""Spot instruments the model invented rather than heard.

Left unconstrained the decoder hands its uncertainty to a plausible-sounding instrument,
which then collects fragments from everywhere. On the reference track that produced an
"acoustic guitar" holding 3430 notes -- the largest track in the file -- that was not in
the recording at all.

Note count cannot find it, precisely because the invented track tends to be the biggest.
Note *length* can: every genuine part had a median note of 0.230s while the invented one
sat at 0.120s, because it was scattering fragments rather than playing.

Forbidding those classes and decoding a second time is what makes the difference, since
the notes are then reassigned to the instruments that really played them rather than
simply discarded.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# A melodic class whose median note is this much shorter than its peers is scattering
# fragments, not playing.
SHORT_NOTE_RATIO = 0.6

# Sounding for less of the track than this is noise, not a part.
MIN_COVERAGE = 0.01

DRUM_INSTRUMENT = "drums"


@dataclass(frozen=True)
class Part:
    """One instrument's statistics from the exploratory pass."""

    instrument: str
    note_count: int
    median_duration: float
    coverage: float
    median_pitch: float


@dataclass(frozen=True)
class Verdict:
    """Which instruments to decode, and why the others were refused."""

    keep: list[str]
    dropped: list[tuple[str, str]]


def summarise(notes: Iterable[tuple[str, float, float, int]], span: float) -> list[Part]:
    """Turn `(instrument, start, end, pitch)` rows into per-instrument statistics."""
    grouped: dict[str, list[tuple[float, float, int]]] = {}
    for instrument, start, end, pitch in notes:
        if end > start:
            grouped.setdefault(instrument, []).append((start, end, pitch))

    parts: list[Part] = []
    for instrument, rows in grouped.items():
        durations = [end - start for start, end, _ in rows]
        parts.append(
            Part(
                instrument=instrument,
                note_count=len(rows),
                median_duration=statistics.median(durations),
                coverage=_covered(rows) / span if span > 0 else 0.0,
                median_pitch=statistics.median([pitch for *_, pitch in rows]),
            )
        )
    return sorted(parts, key=lambda part: -part.note_count)


def _covered(rows: list[tuple[float, float, int]]) -> float:
    """Total time the part is sounding, counting overlaps once."""
    covered = 0.0
    reach = float("-inf")
    for start, end in sorted((start, end) for start, end, _ in rows):
        if start > reach:
            covered += end - start
            reach = end
        elif end > reach:
            covered += end - reach
            reach = end
    return covered


def choose(parts: list[Part]) -> Verdict:
    """Keep the parts that look like real playing, and say why the rest were refused."""
    melodic = [part for part in parts if part.instrument != DRUM_INSTRUMENT]
    if not melodic:
        return Verdict(keep=[part.instrument for part in parts], dropped=[])

    reference = statistics.median([part.median_duration for part in melodic])
    keep: list[str] = []
    dropped: list[tuple[str, str]] = []

    for part in parts:
        if part.instrument == DRUM_INSTRUMENT:
            keep.append(part.instrument)
        elif part.median_duration < SHORT_NOTE_RATIO * reference:
            dropped.append(
                (
                    part.instrument,
                    f"notes too short ({part.median_duration:.3f}s against a {reference:.3f}s median)",
                )
            )
        elif part.coverage < MIN_COVERAGE:
            dropped.append((part.instrument, f"sounds for only {100 * part.coverage:.1f}% of the track"))
        else:
            keep.append(part.instrument)

    # Never let the filter empty the result: whatever sounds for longest is real by
    # definition, even if its notes look odd.
    if not [name for name in keep if name != DRUM_INSTRUMENT]:
        loudest = max(melodic, key=lambda part: part.coverage)
        keep.append(loudest.instrument)
        dropped = [entry for entry in dropped if entry[0] != loudest.instrument]

    return Verdict(keep=keep, dropped=dropped)
