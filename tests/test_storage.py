"""Local storage, including the traversal guard."""

from pathlib import Path

import pytest

from midifier.config import Settings
from midifier.storage import LocalStorage
from midifier.storage import StorageError
from midifier.storage import build_storage


class TestLocalStorage:
    def test_round_trips_a_file(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        url = storage.put("songs/a.mid", b"MThd")
        assert url == "/v1/files/songs/a.mid"
        assert storage.get("songs/a.mid") == b"MThd"
        assert storage.exists("songs/a.mid")

    def test_missing_object_raises(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError):
            LocalStorage(tmp_path).get("nope.mid")

    def test_refuses_to_escape_the_root(self, tmp_path: Path) -> None:
        """A traversal here would write anywhere the process can reach."""
        with pytest.raises(StorageError):
            LocalStorage(tmp_path).put("../escaped.mid", b"x")

    def test_exists_is_false_for_absent_keys(self, tmp_path: Path) -> None:
        assert LocalStorage(tmp_path).exists("absent.mid") is False


class TestBuildStorage:
    def test_defaults_to_local(self, tmp_path: Path) -> None:
        settings = Settings(storage_backend="local", local_storage_dir=tmp_path)
        assert isinstance(build_storage(settings), LocalStorage)

    def test_refuses_a_half_configured_bucket(self) -> None:
        settings = Settings(storage_backend="s3", minio_endpoint="localhost:9000")
        with pytest.raises(StorageError, match="not all set"):
            build_storage(settings)
