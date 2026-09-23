from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

from .models import AudioRecord


def iter_manifest(path: Path) -> Iterator[AudioRecord]:
    """Read newline-delimited JSON so workers never load a corpus into memory."""
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                yield AudioRecord(
                    audio_id=str(raw["audio_id"]),
                    path=str(raw["path"]),
                    expected_text=raw.get("expected_text"),
                    language=raw.get("language"),
                    generator=raw.get("generator"),
                    generator_version=raw.get("generator_version"),
                    voice_id=raw.get("voice_id"),
                    metadata=raw.get("metadata", {}),
                )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid manifest record at {path}:{line_no}: {exc}") from exc


def iter_audio_directory(root: Path, extensions: list[str]) -> Iterator[AudioRecord]:
    """Yield files in a stable order without materialising a full directory listing."""
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    allowed = {extension.lower() for extension in extensions}
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        for filename in sorted(filenames):
            path = Path(directory, filename)
            if path.suffix.lower() not in allowed:
                continue
            relative_path = path.relative_to(root).as_posix()
            audio_id = hashlib.blake2s(relative_path.encode(), digest_size=12).hexdigest()
            yield AudioRecord(audio_id=audio_id, path=str(path))


def write_jsonl(path: Path, records: Iterator[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def stable_sample(audio_id: str, rate: float, seed: int) -> bool:
    """Stable Bernoulli routing: retries and repartitions yield the same sample."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    digest = hashlib.blake2b(f"{seed}:{audio_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < rate
