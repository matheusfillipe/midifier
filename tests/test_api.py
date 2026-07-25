"""The HTTP contract, including the OpenAPI schema the TypeScript client is built from."""

from __future__ import annotations

from fastapi.testclient import TestClient

from midifier.api import create_app
from midifier.config import Settings


class TestHealth:
    def test_reports_version_and_backend(self, client: TestClient) -> None:
        body = client.get("/v1/health").json()
        assert body["status"] == "ok"
        assert body["storage"] == "local"


class TestCreateJob:
    def test_accepts_an_upload_and_returns_a_job(self, client: TestClient) -> None:
        response = client.post("/v1/jobs", files={"file": ("song.mp3", b"ID3fake", "audio/mpeg")})
        assert response.status_code == 202
        assert response.json()["state"] == "queued"

    def test_accepts_a_url(self, client: TestClient) -> None:
        response = client.post("/v1/jobs", data={"url": "https://example.com/song.mp3"})
        assert response.status_code == 202

    def test_rejects_both_file_and_url(self, client: TestClient) -> None:
        response = client.post(
            "/v1/jobs",
            data={"url": "https://example.com/a.mp3"},
            files={"file": ("song.mp3", b"x", "audio/mpeg")},
        )
        assert response.status_code == 400

    def test_rejects_neither(self, client: TestClient) -> None:
        assert client.post("/v1/jobs").status_code == 400

    def test_rejects_an_oversized_upload(self, tmp_path_factory: object) -> None:
        settings = Settings(storage_backend="local", max_upload_bytes=10)
        with TestClient(create_app(settings)) as client:
            response = client.post("/v1/jobs", files={"file": ("song.mp3", b"x" * 100, "audio/mpeg")})
        assert response.status_code == 413

    def test_url_input_can_be_disabled(self) -> None:
        settings = Settings(storage_backend="local", allow_url_input=False)
        with TestClient(create_app(settings)) as client:
            response = client.post("/v1/jobs", data={"url": "https://example.com/a.mp3"})
        assert response.status_code == 400


class TestJobLifecycle:
    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/v1/jobs/does-not-exist").status_code == 404

    def test_created_job_can_be_read_back(self, client: TestClient) -> None:
        job_id = client.post("/v1/jobs", data={"url": "https://example.com/a.mp3"}).json()["id"]
        body = client.get(f"/v1/jobs/{job_id}").json()
        assert body["id"] == job_id
        assert body["source"] == "https://example.com/a.mp3"

    def test_cancelling_marks_the_job_cancelled(self, client: TestClient) -> None:
        job_id = client.post("/v1/jobs", data={"url": "https://example.com/a.mp3"}).json()["id"]
        assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "cancelled"

    def test_cancelling_an_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.delete("/v1/jobs/nope").status_code == 404


class TestApiKey:
    def test_requests_are_rejected_without_the_key(self) -> None:
        settings = Settings(storage_backend="local", api_key="secret")
        with TestClient(create_app(settings)) as client:
            assert client.post("/v1/jobs", data={"url": "https://example.com/a.mp3"}).status_code == 401

    def test_the_key_unlocks_the_endpoint(self) -> None:
        settings = Settings(storage_backend="local", api_key="secret")
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/v1/jobs",
                data={"url": "https://example.com/a.mp3"},
                headers={"X-API-Key": "secret"},
            )
        assert response.status_code == 202

    def test_health_stays_open(self) -> None:
        settings = Settings(storage_backend="local", api_key="secret")
        with TestClient(create_app(settings)) as client:
            assert client.get("/v1/health").status_code == 200


class TestOpenApi:
    def test_schema_documents_the_job_endpoints(self, client: TestClient) -> None:
        """The TypeScript client is generated from this, so drift here breaks kinesthesia."""
        schema = client.get("/openapi.json").json()
        assert "/v1/jobs" in schema["paths"]
        assert "/v1/jobs/{job_id}" in schema["paths"]
        assert schema["paths"]["/v1/jobs"]["post"]["responses"]["202"]
