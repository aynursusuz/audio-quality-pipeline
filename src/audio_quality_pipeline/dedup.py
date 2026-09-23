from __future__ import annotations

import sqlite3
from pathlib import Path


class ExactDeduplicator:
    """Transactional exact-content registry; use one DB per immutable release."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS exact_hashes "
            "(sha256 TEXT PRIMARY KEY, audio_id TEXT NOT NULL)"
        )

    def first_seen_or_duplicate(self, sha256: str, audio_id: str) -> str | None:
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO exact_hashes(sha256, audio_id) VALUES (?, ?)", (sha256, audio_id)
                )
            return None
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                "SELECT audio_id FROM exact_hashes WHERE sha256 = ?", (sha256,)
            ).fetchone()
            return str(row[0]) if row else None

    def close(self) -> None:
        self.connection.close()
