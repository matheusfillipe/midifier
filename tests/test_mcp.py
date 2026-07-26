"""MCP tools. Kinesthesia discovers these over `tools/list`, so names and shapes matter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

from midifier.config import Settings
from midifier.mcp import create_mcp
from midifier.state import store

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastmcp import FastMCP
    from fastmcp.tools import ToolResult


def payload_of(result: ToolResult) -> dict[str, Any]:
    """`structured_content` is optional on the protocol, but every tool here returns one."""
    content = result.structured_content
    assert content is not None
    return dict(content)


@pytest.fixture
def mcp_server(monkeypatch: pytest.MonkeyPatch) -> FastMCP:
    # The tool starts real work now; these tests cover the tool surface, not the pipeline.
    monkeypatch.setattr("midifier.mcp.start", lambda *args, **kwargs: None)
    return create_mcp(Settings(storage_backend="local"))


@pytest.fixture(autouse=True)
def _clear_jobs() -> Iterator[None]:
    yield
    store._jobs.clear()


class TestToolSurface:
    async def test_exposes_the_expected_tools(self, mcp_server: FastMCP) -> None:
        """Renaming a tool silently breaks the bot, which binds by name."""
        names = {tool.name for tool in await mcp_server.list_tools()}
        assert names == {"transcribe_audio", "transcription_status", "transcription_settings"}

    async def test_every_tool_is_described(self, mcp_server: FastMCP) -> None:
        """A tool's description is the only context the model gets about it."""
        for tool in await mcp_server.list_tools():
            assert tool.description
            assert len(tool.description.strip()) > 20


class TestTranscribeAudio:
    async def test_returns_a_job_that_can_be_polled(self, mcp_server: FastMCP) -> None:
        started = await mcp_server.call_tool("transcribe_audio", {"url": "https://example.com/song.mp3"})
        job_id = payload_of(started)["job_id"]

        # wait_seconds=0 so the long poll returns at once rather than holding the test open
        status = await mcp_server.call_tool("transcription_status", {"job_id": job_id, "wait_seconds": 0})
        assert payload_of(status)["state"] == "queued"

    async def test_unknown_job_reports_an_error(self, mcp_server: FastMCP) -> None:
        result = await mcp_server.call_tool("transcription_status", {"job_id": "missing", "wait_seconds": 0})
        assert "no such job" in payload_of(result)["error"]


class TestSettingsTool:
    async def test_reports_how_the_instance_is_configured(self, mcp_server: FastMCP) -> None:
        payload = payload_of(await mcp_server.call_tool("transcription_settings", {}))
        assert payload["model_size"] == "medium"
        assert payload["storage_backend"] == "local"
        assert "queued" in payload["states"]


class TestWorkActuallyStarts:
    """Regression: the MCP tool once queued a job and started nothing, so callers polled
    a job that would never move. Both surfaces must go through the same start path."""

    async def test_transcribe_audio_starts_the_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        started: list[str] = []
        monkeypatch.setattr("midifier.mcp.start", lambda job_id, *a, **k: started.append(job_id))
        server = create_mcp(Settings(storage_backend="local"))

        result = await server.call_tool("transcribe_audio", {"url": "https://example.com/a.mp3"})
        job_id = payload_of(result)["job_id"]

        assert started == [job_id]
        store._jobs.clear()
