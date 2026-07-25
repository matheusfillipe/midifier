"""SSRF guards. This service sits inside a home network, so these are load-bearing."""

from __future__ import annotations

import pytest

from midifier.fetch import UnsafeUrlError
from midifier.fetch import assert_public_url


class TestAssertPublicUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/song.mp3",
            "http://localhost:8000/song.mp3",
            "http://192.168.2.213/song.mp3",
            "http://10.0.0.1/song.mp3",
            "http://169.254.169.254/latest/meta-data",  # cloud metadata
            "http://[::1]/song.mp3",
        ],
    )
    def test_refuses_private_and_loopback_targets(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_public_url(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://example.com/a.mp3"])
    def test_refuses_non_http_schemes(self, url: str) -> None:
        with pytest.raises(UnsafeUrlError, match="unsupported scheme"):
            assert_public_url(url)

    def test_refuses_a_url_without_a_host(self) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_public_url("http:///song.mp3")

    def test_refuses_an_unresolvable_host(self) -> None:
        with pytest.raises(UnsafeUrlError, match="cannot resolve"):
            assert_public_url("http://nonexistent.invalid/song.mp3")
