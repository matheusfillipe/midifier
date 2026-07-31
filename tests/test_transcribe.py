"""The model runs as a subprocess, so what it inherits is part of the contract."""

from midifier.config import Settings
from midifier.transcribe import _environment


class TestModelEnvironment:
    def test_the_configured_token_reaches_the_model(self) -> None:
        """The weights are gated; without this a job fails only when it needs a new size."""
        assert _environment(Settings(hf_token="hf_secret"))["HF_TOKEN"] == "hf_secret"

    def test_no_token_configured_leaves_the_environment_alone(self) -> None:
        assert "HF_TOKEN" not in _environment(Settings(hf_token=None))
