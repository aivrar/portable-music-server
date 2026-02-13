"""Tests for clap_scorer.py — CLAP scoring FastAPI endpoints."""

import base64
import io

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from clap_scorer import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def reset_model():
    """Ensure model is unloaded between tests."""
    import clap_scorer
    clap_scorer._model = None
    yield
    clap_scorer._model = None


class TestClapHealth:
    def test_health_idle(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_health_ready_after_load(self, client):
        import clap_scorer
        clap_scorer._model = MagicMock()  # fake loaded model
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


class TestClapScore:
    def test_score_503_without_load(self, client):
        """Score should fail if model not loaded."""
        resp = client.post("/score", json={
            "audio_base64": base64.b64encode(b"fake").decode(),
            "prompt": "test",
        })
        assert resp.status_code == 503

    def test_score_returns_float(self, client):
        """Score with mocked model returns a valid float."""
        import clap_scorer

        mock_model = MagicMock()
        # Mock embeddings: unit vectors that are identical → score ≈ 1.0
        mock_model.get_audio_embedding_from_data.return_value = np.array([[1.0, 0.0]])
        mock_model.get_text_embedding.return_value = np.array([[1.0, 0.0]])
        clap_scorer._model = mock_model

        # Create a valid WAV file in memory
        import soundfile as sf
        audio = np.zeros(48000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, 48000, format="WAV")
        wav_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post("/score", json={
            "audio_base64": wav_b64,
            "prompt": "silence",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0


class TestClapLoad:
    def test_load_with_mock(self, client):
        mock_module = MagicMock()
        mock_instance = MagicMock()
        mock_module.CLAP_Module.return_value = mock_instance

        with patch.dict("sys.modules", {"laion_clap": mock_module}):
            resp = client.post("/load")

        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"


class TestClapUnload:
    def test_unload_when_not_loaded(self, client):
        resp = client.post("/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_unloaded"

    def test_unload_clears_model(self, client):
        import clap_scorer
        clap_scorer._model = MagicMock()
        resp = client.post("/unload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"
        assert clap_scorer._model is None
