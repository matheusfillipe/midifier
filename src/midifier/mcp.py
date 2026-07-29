"""MCP surface, so an agent can transcribe a song as a tool call.

Tool descriptions are the only context a model gets, so they say what the tool does and
what it costs. Kinesthesia's own MCP server discovers tools over `tools/list`, meaning a
tool added here appears to the bot after a restart with no client change.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from midifier.auth import verify
from midifier.config import Settings
from midifier.config import get_settings
from midifier.jobs import JobState
from midifier.state import queue
from midifier.state import start
from midifier.state import store

KEY_FIELD = Field(description="API key. Not needed when the request already carried it as a header.")

# Set when the transport authenticated the caller, which is the case for every HTTP
# client. Tools then need no key of their own; a stdio client, which has no headers to
# present, still supplies one as an argument.
authenticated: ContextVar[bool] = ContextVar("authenticated", default=False)


# A status call waits this long for something to change before answering. An agent has no
# way to sleep between calls, so without this it polls as fast as it can think, which
# wastes its context and tells it nothing new. Waiting server-side paces the loop for it.
class NotAuthorizedError(RuntimeError):
    """The supplied key does not match."""


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

    def _check(api_key: str | None) -> None:
        if authenticated.get():
            return
        if not verify(api_key, resolved.api_key_hash):
            raise NotAuthorizedError("invalid or missing api_key")

    @mcp.tool
    def transcribe_audio(
        url: Annotated[str, Field(description="Publicly reachable URL of the audio to transcribe.")],
        api_key: Annotated[str | None, KEY_FIELD] = None,
    ) -> dict[str, str]:
        """Start transcribing a song into a multi-track MIDI file.

        Returns a job id to poll with `transcription_status`. Transcription runs at
        roughly three times the length of the song, and one job runs at a time.
        """
        _check(api_key)
        job = store.create(source=url)
        queue.submit(job.id)
        start(job.id, resolved, url=url)
        return {"job_id": job.id, "state": str(job.state)}

    @mcp.tool
    def transcription_status(
        job_id: Annotated[str, Field(description="Job id returned by transcribe_audio.")],
        api_key: Annotated[str | None, KEY_FIELD] = None,
    ) -> dict[str, object]:
        """Check a transcription, including its place in the queue and estimated wait.

        This answers at once. A caller with minutes to wait should sleep on its own side and
        ask again: holding the request open instead puts the wait behind every proxy on the
        path, where it fails in ways that look like the service being broken.
        """
        _check(api_key)
        job = store.get(job_id)
        if job is None:
            return {"error": f"no such job: {job_id}"}

        where = queue.position(job_id) if not job.done else None
        return {
            "queue_ahead": where.ahead if where else None,
            "eta_seconds": where.eta_seconds if where else None,
            "state": str(job.state),
            "stage": str(job.stage) if job.stage else None,
            "midi_url": job.midi_url,
            "tracks": [track.model_dump() for track in job.tracks],
            "error": job.error,
        }

    @mcp.tool
    def transcription_settings(
        api_key: Annotated[str | None, KEY_FIELD] = None,
    ) -> dict[str, object]:
        """Report how this instance is configured, and how busy it currently is."""
        _check(api_key)
        return {
            "queue": queue.snapshot(),
            "model_size": resolved.model_size,
            "segment_seconds": resolved.segment_seconds,
            "storage_backend": resolved.storage_backend,
            "max_duration_seconds": resolved.max_duration_seconds,
            "states": [str(state) for state in JobState],
        }

    return mcp


mcp = create_mcp()
