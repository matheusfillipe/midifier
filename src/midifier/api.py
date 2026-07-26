"""The REST surface. Transcription takes minutes, so every request is a job."""

from __future__ import annotations

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
from midifier.config import Settings
from midifier.config import get_settings
from midifier.fetch import UnsafeUrlError
from midifier.jobs import Job
from midifier.jobs import JobState
from midifier.jobs import JobStore
from midifier.storage import StorageError
from midifier.storage import build_storage
from midifier.worker import run_job

MIDI_MEDIA_TYPE = "audio/midi"

store = JobStore()


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
    """No-op until MIDIFIER_API_KEY is set, so local runs need no credentials."""
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    app = FastAPI(
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
        # Transcription takes minutes, so the response returns now and the work continues
        # after it. Losing the pod loses the job, which is why one Kubernetes Job per
        # request is the deployment shape rather than a long-lived queue in here.
        background.add_task(run_job, job.id, store, resolved, payload, url)
        return JobAccepted(id=job.id, state=job.state)

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

    @app.get("/v1/files/{key:path}", tags=["files"])
    def get_file(key: str) -> Response:
        """Serve a stored MIDI. Only used by the local storage backend."""
        try:
            payload = build_storage(resolved).get(key)
        except StorageError as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
        return Response(content=payload, media_type=MIDI_MEDIA_TYPE)

    @app.exception_handler(UnsafeUrlError)
    def _unsafe_url(_: object, error: UnsafeUrlError) -> Response:
        return Response(content=str(error), status_code=status.HTTP_400_BAD_REQUEST)

    return app


app = create_app()
