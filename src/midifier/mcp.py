"""MCP surface, so an agent can transcribe a song as a tool call.

Tool descriptions are the only context a model gets, so they say what the tool does and
what it costs. Kinesthesia's own MCP server discovers tools over `tools/list`, meaning a
tool added here appears to the bot after a restart with no client change.
"""

from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from midifier.api import store
from midifier.config import Settings
from midifier.config import get_settings
from midifier.jobs import JobState

INSTRUCTIONS = """
midifier turns a recording into a multi-track General MIDI file. It identifies which
instruments are playing, transcribes each one, and names and assigns the tracks, so the
result can be played or practised directly.

Transcription runs at roughly twice the length of the song, so start a job and poll it
rather than waiting on a single call.
""".strip()


def create_mcp(settings: Settings | None = None) -> FastMCP:
    resolved = settings or get_settings()
    mcp: FastMCP = FastMCP(name="midifier", instructions=INSTRUCTIONS)

    @mcp.tool
    def transcribe_audio(
        url: Annotated[str, Field(description="Publicly reachable URL of the audio to transcribe.")],
    ) -> dict[str, str]:
        """Start transcribing a song into a multi-track MIDI file.

        Returns a job id to poll with `transcription_status`. Expect a few minutes.
        """
        job = store.create(source=url)
        return {"job_id": job.id, "state": str(job.state)}

    @mcp.tool
    def transcription_status(
        job_id: Annotated[str, Field(description="Job id returned by transcribe_audio.")],
    ) -> dict[str, object]:
        """Check a transcription, and get the MIDI URL and track list once it is done."""
        job = store.get(job_id)
        if job is None:
            return {"error": f"no such job: {job_id}"}
        return {
            "state": str(job.state),
            "stage": str(job.stage) if job.stage else None,
            "midi_url": job.midi_url,
            "tracks": [track.model_dump() for track in job.tracks],
            "error": job.error,
        }

    @mcp.tool
    def transcription_settings() -> dict[str, object]:
        """Report how this instance is configured, including model size and storage."""
        return {
            "model_size": resolved.model_size,
            "two_pass": resolved.two_pass,
            "storage_backend": resolved.storage_backend,
            "max_duration_seconds": resolved.max_duration_seconds,
            "states": [str(state) for state in JobState],
        }

    return mcp


mcp = create_mcp()
