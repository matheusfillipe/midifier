"""The workflows.h4ks.com job type: its manifest, dispatch endpoint and callback events."""

from collections.abc import Iterator
from urllib.parse import parse_qs
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from midifier.config import Settings
from midifier.jobs import JobState
from midifier.jobs import JobStore
from midifier.jobs import Stage
from midifier.state import store as shared_store
from midifier.workflow import Dispatch
from midifier.workflow import Reporter

CALLBACK = "https://workflows.test/api/jobs/7/events"
SONG = "https://s3.test/workflows/uploads/My%20Song.mp3"
MIDI = "https://s3.test/workflows/midi/abc.mid"

type Answer = httpx.Response | httpx.HTTPError


def dispatch_body(url: str = SONG) -> dict[str, object]:
    return {
        "job_id": 7,
        "type": "midi",
        "params": {"url": url},
        "steps": ["fetch", "transcribe", "store"],
        "callback_url": CALLBACK,
        "callback_token": "secret",
    }


class Workflows:
    """Stands in for the workflows callback endpoint, answering each post from a script."""

    def __init__(self, answers: list[Answer]) -> None:
        self.answers = answers
        self.events: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object], headers: dict[str, str], timeout: float) -> httpx.Response:
        assert url == CALLBACK
        assert headers["Authorization"] == "Bearer secret"
        self.events.append(json)
        answer = self.answers.pop(0) if self.answers else httpx.Response(204)
        if isinstance(answer, httpx.HTTPError):
            raise answer
        return answer


@pytest.fixture
def workflows(monkeypatch: MonkeyPatch) -> Iterator[Workflows]:
    fake = Workflows([])
    monkeypatch.setattr("midifier.workflow.httpx.post", fake.post)
    monkeypatch.setattr("midifier.workflow.time.sleep", lambda _: None)
    yield fake


def watched(settings: Settings, song: str = SONG) -> tuple[JobStore, str, Reporter]:
    store = JobStore()
    job = store.create()
    reporter = Reporter(Dispatch.model_validate(dispatch_body(song)), settings, lambda: store.cancel(job.id))
    store.watch(job.id, reporter)
    return store, job.id, reporter


def finish(reporter: Reporter) -> None:
    reporter.sender.join(timeout=5)
    assert not reporter.sender.is_alive()


class TestReporter:
    def test_posts_each_step_and_segment_progress_once(self, settings: Settings, workflows: Workflows) -> None:
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.RUNNING, stage=Stage.FETCHING)
        store.update(job_id, stage=Stage.TRANSCRIBING)
        store.update(job_id, segments_done=1, segments_total=3)
        store.update(job_id, segments_done=1, segments_total=3, last_segment_at=None)
        store.update(job_id, stage=Stage.STORING)
        store.cancel(job_id)
        finish(reporter)

        assert workflows.events == [
            {"kind": "step", "step": "fetch"},
            {"kind": "step", "step": "transcribe"},
            {"kind": "step", "step": "transcribe", "done": 1, "total": 3},
            {"kind": "step", "step": "store"},
        ]

    def test_posts_the_midi_with_a_kinesthesia_link(self, settings: Settings, workflows: Workflows) -> None:
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.SUCCEEDED, stage=None, midi_url=MIDI)
        finish(reporter)

        [result] = workflows.events
        assert result["title"] == "My Song"
        assert result["files"] == [{"url": MIDI, "name": "My Song.mid", "mime": "audio/midi"}]
        links = result["links"]
        assert isinstance(links, list)
        [link] = links
        query = parse_qs(urlparse(link["url"]).query)
        assert query == {"url": [MIDI], "name": ["My Song"]}

    def test_long_text_fits_what_workflows_accepts(self, settings: Settings, workflows: Workflows) -> None:
        store, job_id, reporter = watched(settings, f"https://s3.test/{'a' * 300}.mp3")

        store.update(job_id, state=JobState.SUCCEEDED, midi_url=MIDI)
        finish(reporter)

        [result] = workflows.events
        assert len(str(result["title"])) == 200
        assert result["files"] == [{"url": MIDI, "name": f"{'a' * 196}.mid", "mime": "audio/midi"}]

    def test_posts_a_trimmed_error_and_stops_watching(self, settings: Settings, workflows: Workflows) -> None:
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.FAILED, error="x" * 2000)
        store.update(job_id, error="later")
        finish(reporter)

        assert workflows.events == [{"kind": "error", "message": "x" * 500}]

    def test_retries_the_result_until_workflows_answers(self, settings: Settings, workflows: Workflows) -> None:
        workflows.answers = [httpx.ConnectError("down"), httpx.Response(503), httpx.Response(204)]
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.SUCCEEDED, midi_url=MIDI)
        finish(reporter)

        assert len(workflows.events) == 3
        assert workflows.answers == []

    def test_a_job_workflows_ended_is_cancelled_here(self, settings: Settings, workflows: Workflows) -> None:
        workflows.answers = [httpx.Response(409)]
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.RUNNING, stage=Stage.FETCHING)
        finish(reporter)

        job = store.get(job_id)
        assert job is not None and job.state is JobState.CANCELLED

    def test_a_cancelled_job_sends_nothing(self, settings: Settings, workflows: Workflows) -> None:
        store, job_id, reporter = watched(settings)

        store.update(job_id, state=JobState.CANCELLED)
        finish(reporter)

        assert workflows.events == []


class TestEndpoints:
    def test_manifest_describes_the_form_and_the_limit(self, client: TestClient) -> None:
        manifest = client.get("/v1/workflow").json()

        assert manifest["name"] == "midi"
        assert manifest["steps"] == [["fetch", 5], ["transcribe", 90], ["store", 5]]
        assert manifest["params_schema"]["properties"]["url"]["description"].endswith("Up to 6 minutes.")
        assert manifest["params_schema"]["required"] == ["url"]
        assert manifest["params_schema"]["properties"]["url"]["x-upload"] == "audio/*,video/*"

    def test_dispatch_starts_a_watched_job_and_says_it_waits(
        self, client: TestClient, workflows: Workflows, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("midifier.api.assert_public_url", lambda url: url)

        response = client.post("/v1/workflow/jobs", json=dispatch_body())
        job_id = response.json()["id"]
        reporter = shared_store._watchers[job_id]
        assert isinstance(reporter, Reporter)
        shared_store.cancel(job_id)
        finish(reporter)

        assert response.status_code == 202
        assert workflows.events[0] == {"kind": "log", "message": "waiting for the GPU"}

    def test_dispatch_refuses_a_private_address(self, client: TestClient) -> None:
        response = client.post("/v1/workflow/jobs", json=dispatch_body("http://127.0.0.1/song.mp3"))

        assert response.status_code == 400

    def test_dispatch_refuses_unknown_params(self, client: TestClient) -> None:
        body = dispatch_body()
        body["params"] = {"url": SONG, "extra": 1}

        assert client.post("/v1/workflow/jobs", json=body).status_code == 422
