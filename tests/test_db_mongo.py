"""Integration tests for the Mongo LandmarkStore.

Requires the Docker Mongo container from docker-compose.yml to be running.
Uses a dedicated test collection that's dropped at the start of each test.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pymongo = pytest.importorskip("pymongo")

from fitcoach.db.mongo import LandmarkStore  # noqa: E402


def _uri() -> str:
    user = os.environ.get("MONGO_USER", "fitcoach")
    pw = os.environ.get("MONGO_PASSWORD", "fitcoach")
    host = os.environ.get("MONGO_HOST", "localhost")
    port = os.environ.get("MONGO_PORT", "27017")
    return f"mongodb://{user}:{pw}@{host}:{port}"


@pytest.fixture
def store():
    try:
        client = pymongo.MongoClient(_uri(), serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
    except Exception as exc:
        pytest.skip(f"Mongo not reachable: {exc}")
    db = client.get_database("fitcoach_test")
    db.drop_collection("landmarks")
    yield LandmarkStore(db, collection="landmarks")
    client.close()


def test_save_and_load_round_trip(store) -> None:
    arr = np.arange(30 * 33 * 4, dtype=np.float32).reshape(30, 33, 4)
    store.save_landmarks(session_id=1, landmarks=arr)
    loaded = store.load_landmarks(session_id=1)
    assert loaded.dtype == np.float32
    assert loaded.shape == arr.shape
    np.testing.assert_array_equal(loaded, arr)


def test_save_preserves_nan_rows(store) -> None:
    arr = np.full((5, 33, 4), 0.5, dtype=np.float32)
    arr[2] = np.nan
    store.save_landmarks(session_id=2, landmarks=arr)
    loaded = store.load_landmarks(session_id=2)
    assert np.isnan(loaded[2]).all()
    assert not np.isnan(loaded[0]).any()


def test_load_unknown_session_raises(store) -> None:
    with pytest.raises(LookupError):
        store.load_landmarks(session_id=12345)


def test_save_overwrites_existing(store) -> None:
    """Saving twice for the same session_id replaces the prior payload."""
    a = np.zeros((3, 33, 4), dtype=np.float32)
    b = np.ones((4, 33, 4), dtype=np.float32)
    store.save_landmarks(session_id=3, landmarks=a)
    store.save_landmarks(session_id=3, landmarks=b)
    loaded = store.load_landmarks(session_id=3)
    np.testing.assert_array_equal(loaded, b)


def test_save_validates_shape(store) -> None:
    """Reject arrays that don't match the (T, 33, 4) contract."""
    with pytest.raises(ValueError):
        store.save_landmarks(session_id=4, landmarks=np.zeros((3, 33), dtype=np.float32))
    with pytest.raises(ValueError):
        store.save_landmarks(session_id=4, landmarks=np.zeros((3, 30, 4), dtype=np.float32))
