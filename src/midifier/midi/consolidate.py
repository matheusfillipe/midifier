"""Fold together lanes that are one part wearing two labels.

The decoder relabels a continuing line -- sometimes where two segments join, sometimes
mid-phrase. Both leave the same fingerprint: two lanes covering the same register that hand
over rather than play together. A listener hears the whole band change instrument; on a
falling-notes display the part appears to stop and a new one start.

Identity is decided by behaviour, not by the name the model chose, because the name is the
thing that is unreliable. Two parts that genuinely coexist are never folded, however similar
their ranges.
"""

import pretty_midi

# How much of the narrower part's register the two must share before they are candidates.
REGISTER_SHARE = 0.5

# Above this much shared airtime they are two parts playing together, not one relabelled.
MAX_TIME_SHARE = 0.25

# Slack for "these two notes are the same event", and how many such coincidences are still
# consistent with a handover rather than a duet.
SIMULTANEOUS_WINDOW = 0.25
MAX_SIMULTANEOUS_SHARE = 0.05


def _span(instrument: pretty_midi.Instrument) -> tuple[float, float]:
    return min(n.start for n in instrument.notes), max(n.end for n in instrument.notes)


def _register(instrument: pretty_midi.Instrument) -> tuple[int, int]:
    pitches = [n.pitch for n in instrument.notes]
    return min(pitches), max(pitches)


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def _coincidences(a: pretty_midi.Instrument, b: pretty_midi.Instrument) -> int:
    return sum(1 for n in a.notes if any(abs(x.start - n.start) < SIMULTANEOUS_WINDOW for x in b.notes))


def _is_one_part(a: pretty_midi.Instrument, b: pretty_midi.Instrument) -> bool:
    low_a, high_a = _register(a)
    low_b, high_b = _register(b)
    narrower = min(high_a - low_a, high_b - low_b)
    if narrower <= 0:
        return False
    if _overlap((low_a, high_a), (low_b, high_b)) / narrower < REGISTER_SHARE:
        return False

    span_a, span_b = _span(a), _span(b)
    shortest = min(span_a[1] - span_a[0], span_b[1] - span_b[0])
    if shortest <= 0 or _overlap(span_a, span_b) / shortest > MAX_TIME_SHARE:
        return False

    return _coincidences(a, b) <= MAX_SIMULTANEOUS_SHARE * min(len(a.notes), len(b.notes))


def consolidate(midi: pretty_midi.PrettyMIDI) -> list[tuple[str, str]]:
    """Merge relabelled lanes in place; returns each `(folded, into)` pair."""
    folded: list[tuple[str, str]] = []
    while True:
        lanes = [i for i in midi.instruments if i.notes and not i.is_drum]
        pair = next(
            ((a, b) for index, a in enumerate(lanes) for b in lanes[index + 1 :] if _is_one_part(a, b)),
            None,
        )
        if pair is None:
            return folded

        first, second = pair
        keep, drop = (first, second) if _span(first)[0] <= _span(second)[0] else (second, first)
        keep.notes.extend(drop.notes)
        keep.notes.sort(key=lambda note: note.start)
        midi.instruments.remove(drop)
        folded.append((drop.name, keep.name))
