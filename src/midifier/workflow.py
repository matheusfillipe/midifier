"""Transcription as a workflows.h4ks.com job type.

workflows reads the manifest to list the job, dispatches paid jobs here, and learns how each
one goes from the events we post back to its callback URL.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Literal
from typing import NotRequired
from typing import TypedDict
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

STEPS = {Stage.FETCHING: ("fetch", 5), Stage.TRANSCRIBING: ("transcribe", 90), Stage.STORING: ("store", 5)}
CALLBACK_TIMEOUT_SECONDS = 10.0
FINAL_EVENT_ATTEMPTS = 5
RETRY_DELAY_SECONDS = 5.0
# workflows rejects longer text with a 422, which would leave its job running until it times out.
MESSAGE_LIMIT = 500
TITLE_LIMIT = 200


class StepEvent(TypedDict):
    kind: Literal["step"]
    step: str
    done: NotRequired[int]
    total: NotRequired[int]


class LogEvent(TypedDict):
    kind: Literal["log"]
    message: str


class ResultFile(TypedDict):
    url: str
    name: str
    mime: str


class ResultLink(TypedDict):
    label: str
    url: str


class ResultEvent(TypedDict):
    kind: Literal["result"]
    title: str
    files: list[ResultFile]
    links: list[ResultLink]


class ErrorEvent(TypedDict):
    kind: Literal["error"]
    message: str


type Event = StepEvent | LogEvent | ResultEvent | ErrorEvent


class MidiParams(BaseModel):
    model_config = ConfigDict(extra="forbid", title="MidiParams")

    url: HttpUrl = Field(
        title="Song",
        description="A link (YouTube, SoundCloud, an audio file) or an upload.",
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


class Media(BaseModel):
    title: str = Field("", description="Title of the page the link pointed at.")
    duration: float | None = Field(None, description="Length of the downloaded audio in seconds.")


class Dispatch(BaseModel):
    job_id: int = Field(description="The job's id in workflows.")
    type: str = Field(description="The job type name.")
    params: MidiParams = Field(description="The validated params.")
    steps: list[str] = Field(description="Step names in order.")
    callback_url: HttpUrl = Field(description="Where to post the job's events.")
    callback_token: str = Field(description="Bearer token for the callbacks.")
    media: dict[str, Media] = Field(
        default_factory=dict, description="What workflows learned about each link it downloaded for us."
    )


def manifest(settings: Settings) -> Manifest:
    params_schema = MidiParams.model_json_schema()
    params_schema["properties"]["url"]["description"] += f" Up to {settings.max_duration_seconds / 60:g} minutes."
    return Manifest(
        name="midi",
        title="Song to MIDI",
        description="Transcribe audio into multi-track MIDI.",
        pricing="1.5 credits per second of audio, plus 60",
        price="1.5 * duration(url) + 60",
        steps=list(STEPS.values()),
        params_schema=params_schema,
    )


def _song_name(dispatch: Dispatch) -> str:
    media = dispatch.media.get("url")
    if media is not None and media.title:
        return media.title[:TITLE_LIMIT]
    return PurePosixPath(unquote(urlparse(str(dispatch.params.url)).path)).stem[:TITLE_LIMIT] or "song"


class Reporter:
    """Watches one job and posts what changes to the workflows callback, in order, off the worker thread.

    `on_ended` runs when workflows answers 409, meaning it already ended the job, so we stop
    spending the GPU on it.
    """

    def __init__(self, dispatch: Dispatch, settings: Settings, on_ended: Callable[[], None]) -> None:
        self._url = str(dispatch.callback_url)
        self._headers = {"Authorization": f"Bearer {dispatch.callback_token}"}
        self._song = _song_name(dispatch)
        self._player_url = settings.player_url
        self._on_ended = on_ended
        self._last: Event | None = None
        self._outbox: queue.Queue[Event | None] = queue.Queue()
        self.sender = threading.Thread(target=self._send_all, daemon=True)
        self.sender.start()

    def send(self, event: Event) -> None:
        self._outbox.put(event)

    def __call__(self, job: Job) -> None:
        event = self._event(job)
        if event is not None and event != self._last:
            self._last = event
            self.send(event)
        if job.done and (event is None or event["kind"] not in ("result", "error")):
            self._outbox.put(None)

    def _event(self, job: Job) -> Event | None:
        if job.state is JobState.SUCCEEDED and job.midi_url:
            return self._result(job.midi_url)
        if job.state is JobState.FAILED:
            return ErrorEvent(kind="error", message=(job.error or "transcription failed")[:MESSAGE_LIMIT])
        if job.state is not JobState.RUNNING or job.stage not in STEPS:
            return None
        step = StepEvent(kind="step", step=STEPS[job.stage][0])
        if job.stage is Stage.TRANSCRIBING and job.segments_total:
            step["done"] = job.segments_done
            step["total"] = job.segments_total
        return step

    def _result(self, midi_url: str) -> ResultEvent:
        player = f"{self._player_url}?url={quote(midi_url, safe='')}&name={quote(self._song, safe='')}"
        return ResultEvent(
            kind="result",
            title=self._song,
            files=[ResultFile(url=midi_url, name=f"{self._song[: TITLE_LIMIT - 4]}.mid", mime="audio/midi")],
            links=[ResultLink(label="open in kinesthesia", url=player)],
        )

    def _send_all(self) -> None:
        while (event := self._outbox.get()) is not None:
            final = event["kind"] in ("result", "error")
            if not self._post(event, FINAL_EVENT_ATTEMPTS if final else 1) or final:
                return

    def _post(self, event: Event, attempts: int) -> bool:
        """Post one event. False when workflows has ended the job and wants nothing more."""
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(self._url, json=event, headers=self._headers, timeout=CALLBACK_TIMEOUT_SECONDS)
            except httpx.HTTPError as error:
                logger.warning("callback %s failed: %s", event["kind"], error)
            else:
                if response.status_code == httpx.codes.CONFLICT:
                    self._on_ended()
                    return False
                if response.is_client_error:
                    logger.warning("callback %s refused: %s %s", event["kind"], response.status_code, response.text)
                if response.is_success or response.is_client_error:
                    return True
                logger.warning("callback %s got %s", event["kind"], response.status_code)
            if attempt < attempts:
                time.sleep(RETRY_DELAY_SECONDS)
        return True
