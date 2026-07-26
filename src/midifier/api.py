"""The REST surface. Transcription takes minutes, so every request is a job."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal

from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Header
from fastapi import HTTPException
from fastapi import Response
from fastapi import UploadFile
from fastapi import status
from pydantic import BaseModel
from pydantic import Field

from midifier import __version__
from midifier.auth import verify
from midifier.config import Settings
from midifier.config import get_settings
from midifier.fetch import UnsafeUrlError
from midifier.jobs import Job
from midifier.jobs import JobState
from midifier.mcp import authenticated
from midifier.mcp import create_mcp
from midifier.state import queue
from midifier.state import store
from midifier.storage import StorageError
from midifier.storage import build_storage
from midifier.worker import run_job

if TYPE_CHECKING:
    from starlette.types import ASGIApp
    from starlette.types import Receive
    from starlette.types import Scope
    from starlette.types import Send

MIDI_MEDIA_TYPE = "audio/midi"

# MCP has no notion of headers of its own, but clients reaching it over HTTP do send
# them, and an agent framework configured with a bearer token or an api-key header is far
# easier to wire than one that must thread a key through every tool call. Both are
# accepted, and the tool argument still works for stdio clients.
BEARER_PREFIX = "Bearer "


class JobAccepted(BaseModel):
    id: str
    state: JobState


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    storage: str


class CreateJob(BaseModel):
    """A transcription request pointing at audio already on the public internet."""

    url: str = Field(description="Publicly reachable audio URL. Private addresses are refused.")


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """No-op until a key hash is configured, so local runs need no credentials."""
    if not verify(x_api_key, settings.api_key_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    # The MCP app manages a task group in its lifespan, so the parent app has to run it;
    # mounting alone leaves it uninitialised and every call fails at request time.
    mcp_app = create_mcp(resolved).http_app(path="/")

    app = FastAPI(
        lifespan=mcp_app.lifespan,
        title="midifier",
        version=__version__,
        summary="Turn a song into a multi-track General MIDI file.",
        description=(
            "Upload audio or point at a URL. The service separates the instruments it hears, "
            "transcribes each one, and returns a General MIDI file with the tracks named and "
            "assigned. Transcription takes minutes, so requests return a job to poll."
        ),
    )

    # Depends(get_settings) would otherwise read the process-wide cached settings,
    # ignoring whatever this app was constructed with.
    app.dependency_overrides[get_settings] = lambda: resolved

    @app.get("/v1/health", response_model=Health, tags=["service"])
    def health() -> Health:
        return Health(status="ok", version=__version__, storage=resolved.storage_backend)

    @app.post(
        "/v1/jobs",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
        tags=["jobs"],
    )
    async def create_job(
        background: BackgroundTasks,
        file: Annotated[UploadFile | None, File(description="Audio file to transcribe.")] = None,
        url: Annotated[str | None, Form(description="Audio URL to transcribe.")] = None,
    ) -> JobAccepted:
        """Accept audio and start transcribing it."""
        if (file is None) == (url is None):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "provide exactly one of 'file' or 'url'")

        if url is not None and not resolved.allow_url_input:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "url input is disabled")

        payload: bytes | None = None
        if file is not None:
            payload = await file.read()
            if len(payload) > resolved.max_upload_bytes:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"upload exceeds {resolved.max_upload_bytes} bytes",
                )
            source = file.filename or "upload"
        else:
            source = url or ""

        job = store.create(source=source)
        queue.submit(job.id)
        # Transcription takes minutes, so the response returns now and the work continues
        # after it. Losing the pod loses the job, which is why one Kubernetes Job per
        # request is the deployment shape rather than a long-lived queue in here.
        background.add_task(queue.run, job.id, lambda: run_job(job.id, store, resolved, payload, url, queue))
        return JobAccepted(id=job.id, state=job.state)

    @app.get("/v1/queue", dependencies=[Depends(require_api_key)], tags=["jobs"])
    def queue_status() -> dict[str, object]:
        """How busy the service is, and how fast it is currently working."""
        return queue.snapshot()

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=Job,
        dependencies=[Depends(require_api_key)],
        tags=["jobs"],
    )
    def get_job(job_id: str) -> Job:
        """Current state of a transcription, including per-track results once finished."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
        if not job.done:
            where = queue.position(job_id)
            if where is not None:
                return job.model_copy(update={"queue_ahead": where.ahead, "eta_seconds": where.eta_seconds})
        return job

    @app.delete(
        "/v1/jobs/{job_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(require_api_key)],
        tags=["jobs"],
    )
    def cancel_job(job_id: str) -> Response:
        """Cancel a job that has not finished."""
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
        if not job.done:
            store.update(job_id, state=JobState.CANCELLED)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/v1/files/{key:path}", dependencies=[Depends(require_api_key)], tags=["files"])
    def get_file(key: str) -> Response:
        """Serve a stored MIDI. Only used by the local storage backend."""
        try:
            payload = build_storage(resolved).get(key)
        except StorageError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        return Response(content=payload, media_type=MIDI_MEDIA_TYPE)

    # The MCP surface is served from the same app, so one URL and one key cover both.
    class McpAuth:
        """Checks the key on the way in, so MCP callers authenticate like REST ones."""

        def __init__(self, inner: ASGIApp) -> None:
            self._inner = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
                presented = headers.get("x-api-key")
                authorization = headers.get("authorization", "")
                if not presented and authorization.startswith(BEARER_PREFIX):
                    presented = authorization[len(BEARER_PREFIX) :]
                if not verify(presented, resolved.api_key_hash):
                    await Response("invalid or missing API key", status_code=401)(scope, receive, send)
                    return
                # The transport has authenticated, so tools need no key of their own.
                authenticated.set(True)
            await self._inner(scope, receive, send)

    app.mount("/mcp", McpAuth(mcp_app))

    @app.exception_handler(UnsafeUrlError)
    def _unsafe_url(_: object, error: UnsafeUrlError) -> Response:
        return Response(content=str(error), status_code=status.HTTP_400_BAD_REQUEST)

    return app


app = create_app()
