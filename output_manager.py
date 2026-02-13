"""Output Manager - Persistent storage for generated audio and metadata."""

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)

GENERATIONS_DIR = OUTPUT_DIR / "generations"


@dataclass
class OutputEntry:
    id: str
    model: str
    prompt: str
    negative_prompt: str
    duration_sec: float
    generation_time_sec: float
    sample_rate: int
    format: str
    seed: int
    clap_score: Optional[float]
    params: dict
    post_processing: dict
    timestamp: str  # ISO format
    filename: str   # audio filename (no path)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class OutputManager:
    """Manages persistent output storage on disk.

    Each generation produces two files:
      {timestamp}_{model}_{id}.wav   (audio)
      {timestamp}_{model}_{id}.json  (metadata)
    """

    def __init__(self, generations_dir: Path | None = None):
        self._dir = generations_dir or GENERATIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, OutputEntry] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load all existing metadata JSON files from disk."""
        for json_path in sorted(self._dir.glob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = OutputEntry.from_dict(data)
                self._entries[entry.id] = entry
            except Exception as e:
                logger.warning("Skipping corrupt metadata %s: %s", json_path.name, e)

    def save_output(
        self,
        audio_bytes: bytes,
        model: str,
        prompt: str,
        negative_prompt: str = "",
        duration_sec: float = 0.0,
        generation_time_sec: float = 0.0,
        sample_rate: int = 44100,
        fmt: str = "wav",
        seed: int = -1,
        params: dict | None = None,
        post_processing: dict | None = None,
    ) -> OutputEntry:
        """Save audio bytes and metadata to disk. Returns the new OutputEntry."""
        entry_id = os.urandom(3).hex()
        ts = datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        base_name = f"{ts_str}_{model}_{entry_id}"
        audio_filename = f"{base_name}.{fmt}"
        json_filename = f"{base_name}.json"

        entry = OutputEntry(
            id=entry_id,
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration_sec=duration_sec,
            generation_time_sec=generation_time_sec,
            sample_rate=sample_rate,
            format=fmt,
            seed=seed,
            clap_score=None,
            params=params or {},
            post_processing=post_processing or {},
            timestamp=ts.isoformat(),
            filename=audio_filename,
        )

        # Write files
        audio_path = self._dir / audio_filename
        json_path = self._dir / json_filename
        audio_path.write_bytes(audio_bytes)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entry.to_dict(), f, indent=2)

        with self._lock:
            self._entries[entry.id] = entry

        logger.info("Saved output %s: %s", entry.id, audio_filename)
        return entry

    def get_all_entries(self) -> list[OutputEntry]:
        """Return all entries sorted by timestamp descending (newest first)."""
        with self._lock:
            entries = list(self._entries.values())
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def get_entry(self, entry_id: str) -> OutputEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def get_audio_path(self, entry_id: str) -> Path | None:
        entry = self.get_entry(entry_id)
        if entry:
            p = self._dir / entry.filename
            return p if p.exists() else None
        return None

    def delete_entry(self, entry_id: str) -> bool:
        """Delete audio + metadata files and remove from memory. Returns True if found."""
        with self._lock:
            entry = self._entries.pop(entry_id, None)
        if not entry:
            return False

        # Delete audio file
        audio_path = self._dir / entry.filename
        if audio_path.exists():
            audio_path.unlink()

        # Delete JSON (derive name from audio filename)
        json_path = audio_path.with_suffix(".json")
        if json_path.exists():
            json_path.unlink()

        logger.info("Deleted output %s", entry_id)
        return True

    def update_clap_score(self, entry_id: str, score: float) -> bool:
        """Update CLAP score for an entry. Persists to JSON on disk."""
        with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return False
            entry.clap_score = score

        # Update JSON on disk
        audio_path = self._dir / entry.filename
        json_path = audio_path.with_suffix(".json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist CLAP score for %s: %s", entry_id, e)

        return True
