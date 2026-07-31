"""Fetching audio from a caller-supplied URL, without becoming a proxy into the LAN.

This service runs inside a home network, so a naive fetch would let any caller reach
private addresses through it. Every hostname is resolved and checked before connecting,
redirects are followed one at a time with the same check applied to each hop, and the
download stops at a byte ceiling rather than trusting the declared length.
"""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_REDIRECTS = 5
CHUNK_BYTES = 64 * 1024


class UnsafeUrlError(ValueError):
    """The URL points somewhere this service refuses to fetch from."""


class FetchError(RuntimeError):
    """The URL was acceptable but the download did not succeed."""


@dataclass(frozen=True)
class Fetched:
    content: bytes
    content_type: str | None
    final_url: str


def _addresses(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise UnsafeUrlError(f"cannot resolve host: {host}") from error
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def assert_public_url(url: str) -> str:
    """Reject anything that is not a public http(s) endpoint.

    Every resolved address is checked, not just the first: a hostname with one public and
    one private record would otherwise slip through.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"unsupported scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise UnsafeUrlError("url has no host")

    for address in _addresses(parsed.hostname):
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeUrlError(f"host resolves to a non-public address: {address}")
    return url


def fetch_audio(url: str, max_bytes: int, timeout: float = 30.0) -> Fetched:
    """Download an audio file, re-validating the target at every redirect."""
    current = assert_public_url(url)

    try:
        return _fetch(current, max_bytes, timeout)
    except httpx.HTTPError as error:
        # httpx raises its own hierarchy, which the worker does not catch; left unwrapped a
        # dead link kills the worker thread and the job sits "running" forever.
        raise FetchError(f"could not fetch {current}: {error}") from error


def _fetch(current: str, max_bytes: int, timeout: float) -> Fetched:
    with httpx.Client(follow_redirects=False, timeout=timeout) as client:
        for _ in range(MAX_REDIRECTS + 1):
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("redirect without a location header")
                    current = assert_public_url(str(response.url.join(location)))
                    continue

                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise UnsafeUrlError(f"response exceeds {max_bytes} bytes")
                return Fetched(
                    content=bytes(body),
                    content_type=response.headers.get("content-type"),
                    final_url=str(response.url),
                )

    raise UnsafeUrlError("too many redirects")
