"""The model runs as a subprocess, so what it inherits is part of the contract."""

import subprocess

import pytest

from midifier.config import Settings
from midifier.transcribe import _environment
from midifier.transcribe import _run


class TestModelEnvironment:
    def test_the_configured_token_reaches_the_model(self) -> None:
        """The weights are gated; without this a job fails only when it needs a new size."""
        assert _environment(Settings(hf_token="hf_secret"))["HF_TOKEN"] == "hf_secret"

    def test_no_token_configured_leaves_the_environment_alone(self) -> None:
        assert "HF_TOKEN" not in _environment(Settings(hf_token=None))


class TestModelCommand:
    def test_tempo_detection_is_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """It runs a beat tracker per segment and stitching discards the result."""
        seen: list[str] = []

        def record(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            seen.extend(args)
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr("midifier.transcribe.subprocess.run", record)
        _run(["in.wav", "-o", "out.mid"], 60.0, Settings())
        assert seen[seen.index("--detect-tempo") + 1] == "false"
