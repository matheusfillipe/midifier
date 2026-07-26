"""MCP over HTTP, which is how an agent framework actually reaches this.

MCP itself has no headers, but a client speaking it over HTTP does, and agent frameworks
are far easier to configure with a bearer token or an api-key header than with a key
threaded through every tool call. Both are accepted here.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from midifier.api import create_app
from midifier.auth import hash_key
from midifier.config import Settings

KEY = "test-key-value"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
}
ACCEPT = "application/json, text/event-stream"


def _client(secured: bool = True) -> TestClient:
    settings = Settings(storage_backend="local", api_key_hash=hash_key(KEY) if secured else None)
    return TestClient(create_app(settings))


class TestAuth:
    def test_refused_without_a_key(self) -> None:
        with _client() as client:
            response = client.post("/mcp/", json=INITIALIZE, headers={"Accept": ACCEPT})
        assert response.status_code == 401

    def test_refused_with_a_wrong_key(self) -> None:
        with _client() as client:
            response = client.post("/mcp/", json=INITIALIZE, headers={"Accept": ACCEPT, "X-API-Key": "nope"})
        assert response.status_code == 401

    def test_accepts_the_api_key_header(self) -> None:
        with _client() as client:
            response = client.post("/mcp/", json=INITIALIZE, headers={"Accept": ACCEPT, "X-API-Key": KEY})
        assert response.status_code == 200

    def test_accepts_a_bearer_token(self) -> None:
        """CloudBot's bearer_token_config_path sends the key this way."""
        with _client() as client:
            response = client.post(
                "/mcp/",
                json=INITIALIZE,
                headers={"Accept": ACCEPT, "Authorization": f"Bearer {KEY}"},
            )
        assert response.status_code == 200

    def test_open_when_no_key_is_configured(self) -> None:
        with _client(secured=False) as client:
            response = client.post("/mcp/", json=INITIALIZE, headers={"Accept": ACCEPT})
        assert response.status_code == 200


class TestHandshake:
    def test_reports_the_server_and_its_instructions(self) -> None:
        """Clients surface `instructions` to their model as the server's context."""
        with _client() as client:
            response = client.post("/mcp/", json=INITIALIZE, headers={"Accept": ACCEPT, "X-API-Key": KEY})
        body = response.text
        assert "midifier" in body
        assert "instructions" in body
