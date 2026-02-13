"""Tests for output_manager.py — persistent output storage."""

import json
from pathlib import Path

import pytest

from output_manager import OutputManager, OutputEntry


@pytest.fixture
def gen_dir(tmp_path):
    d = tmp_path / "generations"
    d.mkdir()
    return d


@pytest.fixture
def mgr(gen_dir):
    return OutputManager(gen_dir)


class TestOutputManager:
    def test_save_creates_audio_and_json(self, mgr, gen_dir):
        entry = mgr.save_output(b"RIFFWAV", model="musicgen", prompt="test")
        audio = gen_dir / entry.filename
        json_f = audio.with_suffix(".json")
        assert audio.exists()
        assert json_f.exists()
        assert audio.read_bytes() == b"RIFFWAV"

    def test_save_returns_correct_fields(self, mgr):
        entry = mgr.save_output(
            b"audio", model="stable_audio", prompt="ambient",
            negative_prompt="noise", duration_sec=10.5,
            generation_time_sec=3.2, sample_rate=44100, fmt="wav",
            seed=42, params={"steps": 50}, post_processing={"eq": "warm"},
        )
        assert entry.model == "stable_audio"
        assert entry.prompt == "ambient"
        assert entry.negative_prompt == "noise"
        assert entry.duration_sec == 10.5
        assert entry.seed == 42
        assert entry.clap_score is None
        assert len(entry.id) == 6

    def test_get_all_entries_sorted_desc(self, mgr):
        e1 = mgr.save_output(b"a", model="m1", prompt="p1")
        e2 = mgr.save_output(b"b", model="m2", prompt="p2")
        entries = mgr.get_all_entries()
        # Newest first
        assert entries[0].id == e2.id
        assert entries[1].id == e1.id

    def test_get_entry(self, mgr):
        entry = mgr.save_output(b"x", model="m", prompt="p")
        assert mgr.get_entry(entry.id).model == "m"
        assert mgr.get_entry("nonexistent") is None

    def test_get_audio_path(self, mgr, gen_dir):
        entry = mgr.save_output(b"wav", model="m", prompt="p")
        path = mgr.get_audio_path(entry.id)
        assert path is not None
        assert path.exists()
        assert mgr.get_audio_path("nonexistent") is None

    def test_delete_entry_removes_files(self, mgr, gen_dir):
        entry = mgr.save_output(b"data", model="m", prompt="p")
        audio_path = gen_dir / entry.filename
        json_path = audio_path.with_suffix(".json")
        assert audio_path.exists()
        assert json_path.exists()

        result = mgr.delete_entry(entry.id)
        assert result is True
        assert not audio_path.exists()
        assert not json_path.exists()
        assert mgr.get_entry(entry.id) is None

    def test_delete_nonexistent_returns_false(self, mgr):
        assert mgr.delete_entry("nonexistent") is False

    def test_update_clap_score_persists(self, mgr, gen_dir):
        entry = mgr.save_output(b"audio", model="m", prompt="p")
        mgr.update_clap_score(entry.id, 0.85)

        # Check in memory
        updated = mgr.get_entry(entry.id)
        assert updated.clap_score == 0.85

        # Check on disk
        audio_path = gen_dir / entry.filename
        json_path = audio_path.with_suffix(".json")
        data = json.loads(json_path.read_text())
        assert data["clap_score"] == 0.85

    def test_load_existing_on_init(self, gen_dir):
        # Create entries with first manager
        mgr1 = OutputManager(gen_dir)
        e1 = mgr1.save_output(b"a", model="m1", prompt="p1")
        e2 = mgr1.save_output(b"b", model="m2", prompt="p2")

        # Create new manager (should load existing)
        mgr2 = OutputManager(gen_dir)
        entries = mgr2.get_all_entries()
        assert len(entries) == 2
        ids = {e.id for e in entries}
        assert e1.id in ids
        assert e2.id in ids

    def test_corrupt_json_skipped(self, gen_dir):
        # Write a corrupt JSON file
        (gen_dir / "bad_data.json").write_text("not valid json")
        mgr = OutputManager(gen_dir)
        assert len(mgr.get_all_entries()) == 0

    def test_update_clap_nonexistent_returns_false(self, mgr):
        assert mgr.update_clap_score("nonexistent", 0.5) is False
