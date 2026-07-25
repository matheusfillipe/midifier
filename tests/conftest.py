"""Shared fixtures. Nothing here touches the network or a transcription model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pretty_midi
import pytest
from fastapi.testclient import TestClient

from midifier.api import create_app
from midifier.api import store
from midifier.config import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(storage_backend="local", local_storage_dir=tmp_path / "data", api_key=None)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    store._jobs.clear()


def note(pitch: int, start: float, end: float, velocity: int = 100) -> pretty_midi.Note:
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)


@pytest.fixture
def midi_factory() -> object:
    """Builds a small PrettyMIDI from `(name, is_drum, notes)` triples."""

    def build(tracks: list[tuple[str, bool, list[pretty_midi.Note]]]) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        for name, is_drum, notes in tracks:
            instrument = pretty_midi.Instrument(program=0, is_drum=is_drum, name=name)
            instrument.notes.extend(notes)
            midi.instruments.append(instrument)
        return midi

    return build
