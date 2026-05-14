"""Download sample exercise videos listed in data/videos.json into data/raw/videos/.

Usage:
    python scripts/fetch_videos.py                    # use default manifest
    python scripts/fetch_videos.py --manifest other.json
    python scripts/fetch_videos.py --force            # re-download even if present

The manifest is a JSON file shaped like:
    {
      "videos": [
        {"name": "squat.mp4", "url": "https://...", "exercise": "squat", "sha256": "..."},
        {"name": "todo.mp4",  "url": null,          "exercise": "pushup"}
      ]
    }
Entries with `url: null` are TODO placeholders and are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "videos.json"
DEFAULT_DEST = ROOT / "data" / "raw" / "videos"

Downloader = Callable[[str, Path], None]
FetchResult = Literal["downloaded", "skipped"]


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoEntry:
    name: str
    url: str
    exercise: str
    sha256: Optional[str] = None


def load_manifest(path: Path) -> list[VideoEntry]:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "videos" not in data or not isinstance(data["videos"], list):
        raise ManifestError("Manifest must be an object with a 'videos' list")

    entries: list[VideoEntry] = []
    for raw in data["videos"]:
        url = raw.get("url")
        if not url:  # None or empty string => placeholder TODO entry
            continue
        entries.append(VideoEntry(
            name=raw["name"],
            url=url,
            exercise=raw.get("exercise", "unknown"),
            sha256=raw.get("sha256"),
        ))
    return entries


_USER_AGENT = "fitcoach-ai/0.1 (+https://github.com/local)"


def _http_download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(target, "wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 16):
            h.update(chunk)
    return h.hexdigest()


def fetch_entry(
    entry: VideoEntry,
    *,
    dest_dir: Path,
    downloader: Downloader = _http_download,
    force: bool = False,
) -> FetchResult:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / entry.name
    if target.exists() and not force:
        return "skipped"

    downloader(entry.url, target)

    if entry.sha256 is not None:
        got = _sha256_of(target)
        if got.lower() != entry.sha256.lower():
            target.unlink(missing_ok=True)
            raise ManifestError(
                f"sha256 mismatch for {entry.name}: expected {entry.sha256}, got {got}"
            )
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch sample exercise videos")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    if not entries:
        print("Manifest has no fetchable entries (all are TODO placeholders).")
        print(f"Edit {args.manifest} and add URLs.")
        return 0

    print(f"Fetching {len(entries)} video(s) into {args.dest}")
    for entry in entries:
        try:
            result = fetch_entry(entry, dest_dir=args.dest, force=args.force)
        except Exception as exc:
            print(f"  ! {entry.name}: {exc}")
            continue
        marker = "→" if result == "downloaded" else "·"
        print(f"  {marker} {entry.name}  ({entry.exercise})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
