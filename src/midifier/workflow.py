"""Transcription as a workflows.h4ks.com job type.

workflows reads the manifest to list the job, dispatches paid jobs here, and learns how each
one goes from the events we post back to its callback URL.
"""

import logging
import time
from pathlib import PurePosixPath
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import HttpUrl

from midifier.config import Settings
from midifier.jobs import Job
from midifier.jobs import JobState
from midifier.jobs import Stage

logger = logging.getLogger(__name__)

STEPS = {Stage.FETCHING: "fetch", Stage.TRANSCRIBING: "transcribe", Stage.STORING: "store"}
CALLBACK_TIMEOUT_SECONDS = 10.0
FINAL_EVENT_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 5.0

type Event = dict[str, object]


class MidiParams(BaseModel):
    model_config = ConfigDict(extra="forbid", title="MidiParams")

    url: HttpUrl = Field(
        title="Song",
        description="A direct link to an audio or video file, or upload one. Up to 6 minutes.",
        json_schema_extra={"x-upload": "audio/*,video/*"},
    )


class Manifest(BaseModel):
    name: str = Field(description="Job type id in workflows URLs and its API.")
    title: str = Field(description="Job type title.")
    description: str = Field(description="What the job does.")
    pricing: str = Field(description="The price explained in words.")
    price: str = Field(description="Price expression in credits over the params.")
    steps: list[tuple[str, int]] = Field(description="Step names in order, each with a progress weight.")
    params_schema: dict[str, object] = Field(description="JSON schema of the params.")


class Dispatch(BaseModel):
    job_id: int = Field(description="The job's id in workflows.")
    type: str = Field(description="The job type name.")
    params: MidiParams = Field(description="The validated params.")
    steps: list[str] = Field(description="Step names in order.")
    callback_url: HttpUrl = Field(description="Where to post the job's events.")
    callback_token: str = Field(description="Bearer token for the callbacks.")


MANIFEST = Manifest(
    name="midi",
    title="Song to MIDI",
    description="Turns a song into a multi-track MIDI file with every instrument it hears, "
    "and opens it in kinesthesia.",
    pricing="1.5 credits per second of audio, plus 60",
    price="1.5 * duration(url) + 60",
    steps=[("fetch", 5), ("transcribe", 90), ("store", 5)],
    params_schema=MidiParams.model_json_schema(),
)


def _song_name(url: str) -> str:
    return PurePosixPath(unquote(urlparse(url).path)).stem or "song"


class Reporter:
    """Watches one job and posts what changes to the workflows callback."""

    def __init__(self, dispatch: Dispatch, settings: Settings) -> None:
        self._url = str(dispatch.callback_url)
        self._headers = {"Authorization": f"Bearer {dispatch.callback_token}"}
        self._song = _song_name(str(dispatch.params.url))
        self._player_url = settings.player_url
        self._last: Event | None = None

    def __call__(self, job: Job) -> None:
        event = self._event(job)
        if event is None or event == self._last:
            return
        self._last = event
        final = event["kind"] in ("result", "error")
        self._post(event, FINAL_EVENT_ATTEMPTS if final else 1)

    def _event(self, job: Job) -> Event | None:
        if job.state is JobState.SUCCEEDED and job.midi_url:
            return self._result(job.midi_url)
        if job.state is JobState.FAILED:
            return {"kind": "error", "message": job.error or "transcription failed"}
        if job.state is not JobState.RUNNING or job.stage not in STEPS:
            return None
        event: Event = {"kind": "step", "step": STEPS[job.stage]}
        if job.stage is Stage.TRANSCRIBING and job.segments_total:
            event |= {"done": job.segments_done, "total": job.segments_total}
        return event

    def _result(self, midi_url: str) -> Event:
        player = f"{self._player_url}?url={quote(midi_url, safe='')}&name={quote(self._song, safe='')}"
        return {
            "kind": "result",
            "title": self._song,
            "files": [{"url": midi_url, "name": f"{self._song}.mid", "mime": "audio/midi"}],
            "links": [{"label": "open in kinesthesia", "url": player}],
        }

    def _post(self, event: Event, attempts: int) -> None:
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(self._url, json=event, headers=self._headers, timeout=CALLBACK_TIMEOUT_SECONDS)
            except httpx.HTTPError as error:
                logger.warning("callback %s failed: %s", event["kind"], error)
            else:
                if response.is_success or response.is_client_error:
                    return
                logger.warning("callback %s got %s", event["kind"], response.status_code)
            if attempt < attempts:
                time.sleep(RETRY_DELAY_SECONDS)
