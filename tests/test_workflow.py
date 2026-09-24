"""The workflows.h4ks.com job type: its manifest, dispatch endpoint and callback events."""

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
from midifier.workflow import Dispatch
from midifier.workflow import Reporter

CALLBACK = "https://workflows.test/api/jobs/7/events"
SONG = "https://s3.test/workflows/uploads/My%20Song.mp3"
MIDI = "https://s3.test/workflows/midi/abc.mid"


def dispatch_body(url: str = SONG) -> dict[str, object]:
    return {
        "job_id": 7,
        "type": "midi",
        "params": {"url": url},
        "steps": ["fetch", "transcribe", "store"],
        "callback_url": CALLBACK,
        "callback_token": "secret",
    }


@pytest.fixture
def posted(monkeypatch: MonkeyPatch) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    def post(url: str, json: dict[str, object], headers: dict[str, str], timeout: float) -> httpx.Response:
        assert url == CALLBACK
        assert headers["Authorization"] == "Bearer secret"
        events.append(json)
        return httpx.Response(204)

    monkeypatch.setattr("midifier.workflow.httpx.post", post)
    return events


def watched(settings: Settings) -> tuple[JobStore, str]:
    store = JobStore()
    job = store.create()
    store.watch(job.id, Reporter(Dispatch.model_validate(dispatch_body()), settings))
    return store, job.id


class TestReporter:
    def test_posts_each_step_and_segment_progress_once(
        self, settings: Settings, posted: list[dict[str, object]]
    ) -> None:
        store, job_id = watched(settings)

        store.update(job_id, state=JobState.RUNNING, stage=Stage.FETCHING)
        store.update(job_id, stage=Stage.TRANSCRIBING)
        store.update(job_id, segments_done=1, segments_total=3)
        store.update(job_id, segments_done=1, segments_total=3, last_segment_at=None)
        store.update(job_id, stage=Stage.STORING)

        assert posted == [
            {"kind": "step", "step": "fetch"},
            {"kind": "step", "step": "transcribe"},
            {"kind": "step", "step": "transcribe", "done": 1, "total": 3},
            {"kind": "step", "step": "store"},
        ]

    def test_posts_the_midi_with_a_kinesthesia_link(self, settings: Settings, posted: list[dict[str, object]]) -> None:
        store, job_id = watched(settings)

        store.update(job_id, state=JobState.SUCCEEDED, stage=None, midi_url=MIDI)

        [result] = posted
        assert result["title"] == "My Song"
        assert result["files"] == [{"url": MIDI, "name": "My Song.mid", "mime": "audio/midi"}]
        links = result["links"]
        assert isinstance(links, list)
        [link] = links
        query = parse_qs(urlparse(link["url"]).query)
        assert query == {"url": [MIDI], "name": ["My Song"]}

    def test_posts_the_error_and_stops_watching(self, settings: Settings, posted: list[dict[str, object]]) -> None:
        store, job_id = watched(settings)

        store.update(job_id, state=JobState.FAILED, error="audio is too long")
        store.update(job_id, error="later")

        assert posted == [{"kind": "error", "message": "audio is too long"}]

    def test_retries_the_result_until_workflows_answers(self, settings: Settings, monkeypatch: MonkeyPatch) -> None:
        answers: list[httpx.Response | httpx.HTTPError] = [
            httpx.ConnectError("down"),
            httpx.Response(503),
            httpx.Response(204),
        ]

        def post(url: str, **_: object) -> httpx.Response:
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr("midifier.workflow.httpx.post", post)
        monkeypatch.setattr("midifier.workflow.time.sleep", lambda _: None)
        store, job_id = watched(settings)

        store.update(job_id, state=JobState.SUCCEEDED, midi_url=MIDI)

        assert answers == []

    def test_a_cancelled_job_sends_nothing(self, settings: Settings, posted: list[dict[str, object]]) -> None:
        store, job_id = watched(settings)

        store.update(job_id, state=JobState.CANCELLED)

        assert posted == []


class TestEndpoints:
    def test_manifest_describes_the_form(self, client: TestClient) -> None:
        manifest = client.get("/v1/workflow").json()

        assert manifest["name"] == "midi"
        assert manifest["params_schema"]["required"] == ["url"]
        assert manifest["params_schema"]["properties"]["url"]["x-upload"] == "audio/*,video/*"

    def test_dispatch_starts_a_job(self, client: TestClient, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr("midifier.api.assert_public_url", lambda url: url)

        response = client.post("/v1/workflow/jobs", json=dispatch_body())

        assert response.status_code == 202

    def test_dispatch_refuses_a_private_address(self, client: TestClient) -> None:
        response = client.post("/v1/workflow/jobs", json=dispatch_body("http://127.0.0.1/song.mp3"))

        assert response.status_code == 400

    def test_dispatch_refuses_unknown_params(self, client: TestClient) -> None:
        body = dispatch_body()
        body["params"] = {"url": SONG, "extra": 1}

        assert client.post("/v1/workflow/jobs", json=body).status_code == 422
