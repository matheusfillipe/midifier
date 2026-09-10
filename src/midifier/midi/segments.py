"""Where to cut a long decode.

Long decodes drift. The model works in five-second chunks and carries each chunk's opening
from the previous one, so a part it loses stays lost and an invented ending runs to the last
chunk. Decoding in segments caps how far either can travel, and the closing segment stops on
its own rather than filling its budget with noise.

Putting them back together is `parts`, which decides the song's instruments from all the
segments at once.
"""

# Shared seconds between neighbouring segments. Read it as `length - OVERLAP_SECONDS`: how far
# into a decode we trust it before the next segment takes over. That is what governs quality,
# since a decode degenerates as it runs and the tail we give up is the part that has degenerated.
OVERLAP_SECONDS = 5.0


def plan(duration: float, length: float) -> list[float]:
    """Start offsets covering `duration`, each segment overlapping the last."""
    step = length - OVERLAP_SECONDS
    offsets = [0.0]
    while offsets[-1] + length < duration:
        offsets.append(offsets[-1] + step)
    return offsets
