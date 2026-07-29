"""Retrying a decode, which is what stands between a flaky accelerator and a failed job."""

from __future__ import annotations

from pathlib import Path

import pytest

from midifier.config import Settings
from midifier.transcribe import TranscriptionError
from midifier.transcribe import _decode
from midifier.transcribe import _is_transient

HANG = "Loading model…\nHW Exception by GPU node-1 (Agent handle: 0x1) reason :GPU Hang\n"


class TestTransientDetection:
    @pytest.mark.parametrize(
        "message",
        [HANG, "HIPFFT_PARSE_ERROR", "CUDA error: unspecified launch failure", "Aborted (core dumped)"],
    )
    def test_device_failures_are_worth_retrying(self, message: str) -> None:
        assert _is_transient(message)

    @pytest.mark.parametrize(
        "message",
        ["could not read audio duration", "the file is not audio", "gated and require a HuggingFace account"],
    )
    def test_input_failures_are_not(self, message: str) -> None:
        assert not _is_transient(message)


class TestDecodeRetry:
    def test_a_hang_is_retried_and_can_succeed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def flaky(args: list[str], timeout: float, settings: Settings) -> None:
            calls.append(1)
            if len(calls) < 3:
                raise TranscriptionError(HANG)
            Path(args[2]).write_bytes(_MINIMAL_MIDI)

        monkeypatch.setattr("midifier.transcribe._run", flaky)
        monkeypatch.setattr("midifier.transcribe.RETRY_PAUSE_SECONDS", 0.0)

        _decode(tmp_path / "in.wav", tmp_path / "out.mid", Settings(storage_backend="local"), 10.0)
        assert len(calls) == 3

    def test_a_segment_that_keeps_hanging_falls_back_to_a_smaller_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing kernel is specific to the shapes one size produces, so a smaller model
        often decodes audio the larger one cannot. One weaker segment beats a failed song."""
        sizes: list[str] = []

        def hangs_on_large(args: list[str], timeout: float, settings: Settings) -> None:
            model = args[args.index("-m") + 1]
            sizes.append(model)
            if model == "large":
                raise TranscriptionError(HANG)
            Path(args[2]).write_bytes(_MINIMAL_MIDI)

        monkeypatch.setattr("midifier.transcribe._run", hangs_on_large)
        monkeypatch.setattr("midifier.transcribe.RETRY_PAUSE_SECONDS", 0.0)

        _decode(tmp_path / "in.wav", tmp_path / "out.mid", Settings(storage_backend="local", model_size="large"), 10.0)
        assert sizes[: Settings().decode_attempts] == ["large"] * Settings().decode_attempts
        assert sizes[-1] == "medium"

    def test_it_gives_up_once_no_smaller_model_is_left(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sizes: list[str] = []

        def always_hangs(args: list[str], timeout: float, settings: Settings) -> None:
            sizes.append(args[args.index("-m") + 1])
            raise TranscriptionError(HANG)

        monkeypatch.setattr("midifier.transcribe._run", always_hangs)
        monkeypatch.setattr("midifier.transcribe.RETRY_PAUSE_SECONDS", 0.0)

        with pytest.raises(TranscriptionError, match="GPU Hang"):
            _decode(
                tmp_path / "in.wav", tmp_path / "out.mid", Settings(storage_backend="local", model_size="large"), 10.0
            )
        assert set(sizes) == {"large", "medium", "small"}

    def test_a_real_error_fails_at_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retrying bad input just spends minutes arriving at the same answer."""
        calls: list[int] = []

        def bad_input(args: list[str], timeout: float, settings: Settings) -> None:
            calls.append(1)
            raise TranscriptionError("the file is not audio")

        monkeypatch.setattr("midifier.transcribe._run", bad_input)

        with pytest.raises(TranscriptionError, match="not audio"):
            _decode(tmp_path / "in.wav", tmp_path / "out.mid", Settings(storage_backend="local"), 10.0)
        assert len(calls) == 1


_MINIMAL_MIDI = bytes.fromhex("4d546864000000060000000100604d54726b0000000400ff2f00")
