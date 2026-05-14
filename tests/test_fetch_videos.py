"""Tests for the video manifest fetcher — no network access required."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_videos import (  # noqa: E402
    VideoEntry,
    ManifestError,
    load_manifest,
    fetch_entry,
)


def test_load_manifest_parses_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "videos.json"
    manifest.write_text(json.dumps({
        "videos": [
            {"name": "a.mp4", "url": "https://example.test/a.mp4", "exercise": "squat"},
            {"name": "b.mp4", "url": "https://example.test/b.mp4", "exercise": "pushup",
             "sha256": "deadbeef"},
        ]
    }))
    entries = load_manifest(manifest)
    assert [e.name for e in entries] == ["a.mp4", "b.mp4"]
    assert entries[0].exercise == "squat"
    assert entries[0].sha256 is None
    assert entries[1].sha256 == "deadbeef"


def test_load_manifest_skips_todo_entries(tmp_path: Path) -> None:
    """Entries with a null/empty url are placeholders and should be skipped."""
    manifest = tmp_path / "videos.json"
    manifest.write_text(json.dumps({
        "videos": [
            {"name": "todo.mp4", "url": None, "exercise": "curl"},
            {"name": "ok.mp4", "url": "https://example.test/ok.mp4", "exercise": "squat"},
        ]
    }))
    entries = load_manifest(manifest)
    assert [e.name for e in entries] == ["ok.mp4"]


def test_load_manifest_rejects_bad_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"not_videos": []}))
    with pytest.raises(ManifestError):
        load_manifest(manifest)


def test_fetch_entry_skips_when_file_present(tmp_path: Path) -> None:
    dest = tmp_path / "a.mp4"
    dest.write_bytes(b"already here")
    calls: list[str] = []

    def fake_downloader(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(b"new")

    entry = VideoEntry(name="a.mp4", url="https://example.test/a.mp4", exercise="x", sha256=None)
    result = fetch_entry(entry, dest_dir=tmp_path, downloader=fake_downloader)
    assert result == "skipped"
    assert calls == []
    assert dest.read_bytes() == b"already here"


def test_fetch_entry_downloads_when_missing(tmp_path: Path) -> None:
    def fake_downloader(url: str, target: Path) -> None:
        target.write_bytes(b"payload")

    entry = VideoEntry(name="a.mp4", url="https://example.test/a.mp4", exercise="x", sha256=None)
    result = fetch_entry(entry, dest_dir=tmp_path, downloader=fake_downloader)
    assert result == "downloaded"
    assert (tmp_path / "a.mp4").read_bytes() == b"payload"


def test_fetch_entry_verifies_sha256(tmp_path: Path) -> None:
    payload = b"hello world"
    good_hash = hashlib.sha256(payload).hexdigest()

    def good_downloader(url: str, target: Path) -> None:
        target.write_bytes(payload)

    def bad_downloader(url: str, target: Path) -> None:
        target.write_bytes(b"corrupted")

    good = VideoEntry(name="g.mp4", url="u", exercise="x", sha256=good_hash)
    bad = VideoEntry(name="b.mp4", url="u", exercise="x", sha256=good_hash)

    assert fetch_entry(good, dest_dir=tmp_path, downloader=good_downloader) == "downloaded"
    with pytest.raises(ManifestError, match="sha256 mismatch"):
        fetch_entry(bad, dest_dir=tmp_path, downloader=bad_downloader)
    # The bad file should have been removed so a retry can proceed cleanly.
    assert not (tmp_path / "b.mp4").exists()
