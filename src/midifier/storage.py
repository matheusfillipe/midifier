"""Where finished MIDI files go.

Two backends behind one protocol: the local filesystem, so the service runs with no
infrastructure, and S3/MinIO, which is what kinesthesia reads from.
"""

import io
import shutil
from pathlib import Path
from typing import Protocol

from minio import Minio
from minio.error import S3Error

from midifier.config import Settings

MIDI_CONTENT_TYPE = "audio/midi"


class StorageError(RuntimeError):
    """Raised when a backend cannot store or address an object."""


class Storage(Protocol):
    """Stores a MIDI file and returns a URL a browser can fetch it from."""

    def put(self, key: str, data: bytes, content_type: str = MIDI_CONTENT_TYPE) -> str: ...

    def get(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalStorage:
    """Writes under a directory. The URL is a path served by the API itself."""

    def __init__(self, root: Path, url_prefix: str = "/v1/files") -> None:
        self._root = root
        self._url_prefix = url_prefix.rstrip("/")
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are service-generated, but a traversal here would write anywhere on disk,
        # so the resolved path is checked to still sit under the root.
        candidate = (self._root / key).resolve()
        if not candidate.is_relative_to(self._root.resolve()):
            raise StorageError(f"key escapes the storage root: {key!r}")
        return candidate

    def put(self, key: str, data: bytes, content_type: str = MIDI_CONTENT_TYPE) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"{self._url_prefix}/{key}"

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(f"no such object: {key!r}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def clear(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)


class S3Storage:
    """MinIO or any S3-compatible bucket."""

    def __init__(self, client: Minio, bucket: str, public_base: str | None = None) -> None:
        self._client = client
        self._bucket = bucket
        self._public_base = public_base.rstrip("/") if public_base else None

    def put(self, key: str, data: bytes, content_type: str = MIDI_CONTENT_TYPE) -> str:
        self._client.put_object(
            self._bucket,
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        if self._public_base is None:
            raise StorageError("MIDIFIER_MINIO_PUBLIC_BASE must be set to address stored objects")
        return f"{self._public_base}/{key}"

    def get(self, key: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(self._bucket, key)
            return bytes(response.read())
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
        except S3Error:
            return False
        return True


def build_storage(settings: Settings) -> Storage:
    """Pick a backend from configuration, failing loudly on a half-configured bucket."""
    if settings.storage_backend == "local":
        return LocalStorage(settings.local_storage_dir)

    if not settings.s3_configured:
        raise StorageError(
            "storage_backend is 's3' but MINIO_ENDPOINT / ACCESS_KEY / SECRET_KEY / BUCKET are not all set"
        )

    assert settings.minio_endpoint is not None
    assert settings.minio_bucket is not None
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_use_ssl,
        region=settings.minio_region,
    )
    return S3Storage(client, settings.minio_bucket, settings.minio_public_base)
